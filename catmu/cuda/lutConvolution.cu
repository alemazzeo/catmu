#include "lutConvolution.h"


// Kernel de CUDA para la convolución TMU
__global__ void lutKernel2D(sImage2d image, Positions2d pos, sPSF psf, cudaTextureObject_t texPSF,
                            int sub_pixel, int offset_image, int offset_position){

    // Identificación del kernel
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t idy = blockIdx.y * blockDim.y + threadIdx.y;

    int x, y;
    float px, py, pixel;
    float factor_x, factor_y, center_x, center_y;

    // Factor de conversión entre el pixel de la PSF y el de la imagen
    factor_x = image.pixel_width / psf.pixel_width;
    factor_y = image.pixel_height / psf.pixel_height;

    // Centro de la PSF
    center_x = psf.width / 2.0;
    center_y = psf.height / 2.0;

    // División interna del trabajo para más de un pixel por kernel
    for (int j = 0; j < sub_pixel; j++){
        for (int k = 0; k < sub_pixel; k++){

            // Pixel calculado en la imagen (X, Y)
            x = idx * sub_pixel + j;
            y = idy * sub_pixel + k;

            // Condición para calcular el pixel (que pertenezca a la imagen)
            if (idx < image.width && idy < image.height) {

                // Resultado acumulado (inicialmente en cero)
                pixel = 0;

                // Iteración sobre todas las fuentes virtuales
                for (int i = 0; i < pos.n; i++){
                    // Conversión de coordenadas
                    px = (x-pos.data[offset_position + i*2]) * factor_x + center_x;
                    py = (y-pos.data[offset_position + i*2+1]) * factor_y + center_y;
                    // Evaluación realizada por la TMU para las coordenadas dadas
                    pixel += tex2D<float>(texPSF, px, py);
                }

                // Resultado aplicado a la imagen
                image.data[offset_image + y * image.width + x] = pixel;
            }
        }
    }
}

// Configuración del dispositivo (GPU) utilizado
int set_device(int device){
    int count, current_device;
    // Consulta la cantidad de dispositivos disponibles
    cudaGetDeviceCount(&count);

    // Revisa que el dispositivo seleccionado exista
    if (device >= count){
        return 101;
    }
    info_print("Selecting device %d (%d available)\n", device, count);

    // Configura el dispositivo
    cudaSetDevice(device);
    CUDA_CHECK_ERROR(return err);

    // Consulta el dispositivo actual
    cudaGetDevice(&current_device);
    info_print("Current device: %d\n", current_device);

    // Reporta el error en caso de que la asignación falle
    if (current_device != device){
        return -2;
    }

    return 0;
}


int free_device_memory(sImage2d * d_image, Positions2d * d_pos,
                       cudaTextureObject_t * texObj, cudaArray * cuArray){

    if (d_image->data != 0){
        info_print("Free results memory\n");
        cudaFree(d_image->data);
        CUDA_CHECK_ERROR();
    }

    if (d_pos->data != 0){
        info_print("Free positions memory\n");
        cudaFree(d_pos->data);
        CUDA_CHECK_ERROR();
    }

    info_print("Destroy texture object\n");
    cudaDestroyTextureObject(*texObj);
    CUDA_CHECK_ERROR();
    info_print("Free texture memory\n");
    cudaFreeArray(cuArray);
    CUDA_CHECK_ERROR();

    return 0;
}

int set_texture_2d(cudaArray * cuArray, sPSF * psf,
                  cudaTextureObject_t * texObj){

    // Formato del array que contiene la textura (formalismo, acá no hay canales RGBA)
    info_print("Allocate CUDA array in device memory\n");
    cudaChannelFormatDesc channelDesc =
               cudaCreateChannelDesc(32, 0, 0, 0,
                                     cudaChannelFormatKindFloat);

    // Pedido de memoria para el array de la textura
    cudaMallocArray(&cuArray, &channelDesc, psf->width, psf->height);
    CUDA_CHECK_ERROR(return err);

    // Transferencia de CPU a GPU de la PSF en formato array (compatible para texturas)
    info_print("Copy to device memory sPSF data from host\n");
    cudaMemcpy2DToArray(cuArray, 0, 0, psf->data,
                        psf->width * sizeof(float),
                        psf->width * sizeof(float),
                        psf->height,
                        cudaMemcpyHostToDevice);
    CUDA_CHECK_ERROR(return err);

    // Configuración del descriptor de la textura
    info_print("Specify texture\n");
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;
    CUDA_CHECK_ERROR(return err);

    // Parámetros configurables
    info_print("Specify texture object parameters\n");
    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    // Modo de acceso (border -> cualquier acceso fuera de rango devuelve un cero)
    texDesc.addressMode[0]   = cudaAddressModeBorder;
    texDesc.addressMode[1]   = cudaAddressModeBorder;
    // Filtrado (interpolado lineal de valores)
    texDesc.filterMode       = cudaFilterModeLinear;
    // Tipo de acceso (basado en los elementos del array -> float32)
    texDesc.readMode         = cudaReadModeElementType;
    // Tipo de coordenadas no normalizadas -> [0, N)
    texDesc.normalizedCoords = 0;
    CUDA_CHECK_ERROR(return err);

    // Objecto que representa la textura -> texObj
    info_print("Create texture object\n");
    cudaCreateTextureObject(texObj, &resDesc, &texDesc, NULL);
    CUDA_CHECK_ERROR(return err);

    return 0;

}

int set_images_2d(sImage2d * h_image, sImage2d * d_image){
    info_print("Allocating memory for results in device\n");

    // Calcula el tamaño requerido
    size_t size = h_image->width * h_image->height * h_image->N * sizeof(float);
    info_print("Request: %lu bytes\n", size);

    // Revisa si la memoria fue asignada antes
    if (d_image->data != 0){
        info_print("Memory allocated in device: %lu bytes\n", d_image->allocated_size);

        // Revisa que los tamaños coincidan
        if (d_image->allocated_size != size){
            // Libera el bloque de tamaño incorrecto
            info_print("Releasing block\n");
            cudaFree(d_image->data);
            CUDA_CHECK_ERROR(return err);
            info_print("Memory released\n");

            // Pide el bloque correcto
            info_print("Allocating memory block of %lu bytes\n", size);
            cudaMalloc(&d_image->data, size);
            CUDA_CHECK_ERROR(return err);
            d_image->allocated_size = size;
            info_print("Memory allocated at %p\n", d_image->data);
        }

    } else {
        // Pide memoria
        info_print("Allocating memory block of %lu bytes\n", size);
        cudaMalloc(&d_image->data, size);
        CUDA_CHECK_ERROR(return err);
        d_image->allocated_size = size;
        info_print("Memory allocated at %p\n", d_image->data);
    }

    // Actualiza la metadata de las imágenes
    info_print("Updating image metadata\n");
    d_image->width = h_image->width;
    d_image->height = h_image->height;
    d_image->pixel_width = h_image->pixel_width;
    d_image->pixel_height = h_image->pixel_height;
    info_print("width = %d\n", d_image->width);
    info_print("height = %d\n", d_image->height);
    info_print("pixel_width = %f\n", d_image->pixel_width);
    info_print("pixel_height = %f\n", d_image->pixel_height);

    info_print("Memory for results is ready to use\n");
    return 0;
}

int upload_positions_2d(Positions2d * h_pos, sPositions2d * d_pos){
    info_print("Allocating memory for lists of positions in device\n");

    // Calcula el tamaño requerido
    size_t size = h_pos->N * h_pos->n * 2 * sizeof(float);
    info_print("Request: %lu bytes\n", size);

    // Revisa si la memoria fue asignada antes
    if (d_pos->data != 0){
        info_print("Memory allocated in device: %lu bytes\n", d_pos->allocated_size);

        // Revisa si la memoria fue asignada antes
        if (d_pos->allocated_size != size){
            // Libera el bloque de tamaño incorrecto
            info_print("Releasing block\n");
            cudaFree(d_pos->data);
            CUDA_CHECK_ERROR(return err);
            info_print("Memory released\n");

            // Pide el bloque correcto
            info_print("Allocating memory block of %lu bytes\n", size);
            cudaMalloc(&d_pos->data, size);
            CUDA_CHECK_ERROR(return err);
            d_pos->allocated_size = size;
            info_print("Memory allocated at %p\n", d_pos->data);
        }
    } else {
        // Pide memoria
        info_print("Allocating memory block of %lu bytes\n", size);
        cudaMalloc(&d_pos->data, size);
        CUDA_CHECK_ERROR(return err);
        d_pos->allocated_size = size;
        info_print("Memory allocated at %p\n", d_pos->data);
    }

    d_pos->n = h_pos->n;
    info_print("n = %d\n", d_pos->n);
    info_print("Memory for positions is ready to use\n");

    // Tranfiere los datos de CPU a GPU
    info_print("Uploading positions from host CPU to device GPU\n");
    cudaMemcpy(d_pos->data, h_pos->data, size, cudaMemcpyHostToDevice);
    CUDA_CHECK_ERROR(return err);
    info_print("Successfully uploaded\n");

    return 0;
}

int download_results(sImage2d * h_image, sImage2d * d_image){
    // Descarga los resultados de GPU a CPU
    info_print("Downloading results from device GPU to host CPU\n");
    cudaMemcpy(h_image->data, d_image->data, d_image->allocated_size, cudaMemcpyDeviceToHost);
    CUDA_CHECK_ERROR(return err);
    info_print("Successfully downloaded\n");

    return 0;
}


extern "C" {
int lutConvolution2D(callback checkpoint, sImage2d * h_image, Positions2d * h_pos, sPSF * h_psf, sConfig * config){
    info_print("Starting LUT Convolution Module of CaTMU\n");

    // Definición de variables
    int r = 0;
    clock_t start, end;
    double elapsed_time = 0;
    unsigned long loop_counter = 0;

    cudaArray * cuArray = 0;
    cudaTextureObject_t texObj = 0;

    cudaStream_t * stream;

    sImage2d d_image;
    sPositions2d d_pos;

    d_image.data = 0;
    d_pos.data = 0;

    // Configuración del dispositivo
    info_print("Setting device\n");
    r = set_device(config->device);
    if (r != 0) return r;

    // Configuración de la textura
    info_print("Setting PSF as texture object\n");
    r = set_texture_2d(cuArray, h_psf, &texObj);
    if (r != 0) return r;

    info_print("width = %d\n", h_psf->width);
    info_print("height = %d\n", h_psf->height);
    info_print("pixel_width = %f\n", h_psf->pixel_width);
    info_print("pixel_height = %f\n", h_psf->pixel_height);

    // Bucle principal para mantener la GPU en espera entre pedidos
    info_print("Main loop\n");
    while (checkpoint(elapsed_time, loop_counter) == true){
        // Control de tiempos (para rendimiento)
        start = clock();

        // Configuración de la memoria para resultados
        info_print("Setting memory for results\n");
        r = set_images_2d(h_image, &d_image);
        if (r != 0) return r;

        // Configuración de la memoria para posiciones y subida de datos a la GPU
        info_print("Uploading data for positions\n");
        info_print("n -> %d\n", h_pos->n);
        r = upload_positions_2d(h_pos, &d_pos);
        if (r != 0) return r;

        // Configuración de los streams de CUDA (para optimizar el uso de la GPU)
        stream = (cudaStream_t *) malloc(config->n_streams * sizeof(cudaStream_t));
        info_print("Setting %d streams\n", config->n_streams);
        for (int i = 0; i < config->n_streams; i ++)
        {
            cudaStreamCreate(&stream[i]);
            CUDA_CHECK_ERROR(return err);
            info_print("Stream %d created\n", i);
        }

        // Configuración de los bloques y grillas para la paralelización
        info_print("Grid and block sizes:\n");
        dim3 dimBlock(config->block_size, config->block_size);
        dim3 dimGrid((h_image->width / config->sub_pixel  + dimBlock.x - 1) / dimBlock.x,
                     (h_image->height / config->sub_pixel + dimBlock.y - 1) / dimBlock.y);

        info_print("dimGrid: %dx%d\n", dimGrid.x, dimGrid.y);
        info_print("dimBlock: %dx%d\n", dimBlock.x, dimBlock.y);

        // Offsets aplicados a los punteros de imagenes y posiciones para acceder a cada uno de ellos
        int offset_image = 0;
        int offset_position = 0;

        // Carga de tareas para cada stream (iterando entre streams)
        info_print("Launching streams\n");
        for (int i = 0; i < h_pos->N; i++)
        {
            // Instancia de kernels en diferentes streams
            info_print("Convolution %d launched on stream %d\n", i, i % config->n_streams);
            info_print("Image offset: %d, Position offset: %d\n", offset_image, offset_position);
            lutKernel2D <<<dimGrid, dimBlock, 0, stream[i % config->n_streams]>>> \
                 (d_image, d_pos, *h_psf, texObj, config->sub_pixel, offset_image, offset_position);
            CUDA_CHECK_ERROR(return err);

            // Actualización del buffer para la siguiente convolución
            offset_position += 2 * h_pos->n;
            offset_image += h_image->width * h_image->height;
        }

        // Barrera de sincronización (todos los streams deben terminar acá)
        info_print("Waiting for synchronization...\n");
        cudaDeviceSynchronize();
        CUDA_CHECK_ERROR(return err);
        info_print("Synchronization completed\n");

        // Descarga de resultados
        download_results(h_image, &d_image);

        // Cierre de streams generados
        info_print("Destroying %d streams\n", config->n_streams);
        for (int i = 0; i < config->n_streams; i ++)
        {
            cudaStreamDestroy(stream[i]);
            CUDA_CHECK_ERROR(return err);
            info_print("Stream %d destroyed\n", i);
        }
        free(stream);

        // Marca final de tiempo, actualización del tiempo transcurrido e incremento del contador
        end = clock();
        elapsed_time = ((double) (end - start)) / CLOCKS_PER_SEC;
        loop_counter += 1;
    }

    // Memoria liberada en el dispositivo (general)
    info_print("Releasing memory from device\n");
    free_device_memory(&d_image, &d_pos, &texObj, cuArray);

    // Reseteo del entorno creado (para mayor seguridad, no debería ser necesario)
    info_print("Executing reset of device\n");
    cudaDeviceReset();
    CUDA_CHECK_ERROR(return err);
    info_print("\nShutting down\n");

    return 0;
}}

double get_T(sPSF * psf, int i, int j){
    if (j >= 0 && j < psf->width && i >= 0 && i < psf->height){
        return psf->data[i * psf->width + j];
    } else {
        return 0;
    }

}

double cpu_tex2d(sPSF * psf, float x, float y){
    int i, j;
    double a, b;
    double T1, T2, T3, T4;

    i = (int) floor(y - 0.5);
    a = (double) (y - 0.5) - i;

    j = (int) floor(x - 0.5);
    b = (double) (x - 0.5) - j;

    T1 = get_T(psf, i, j);
    T2 = get_T(psf, i+1, j);
    T3 = get_T(psf, i, j+1);
    T4 = get_T(psf, i+1, j+1);

    return (double) ((1-a) * (1-b) * T1 + a * (1-b) * T2 + (1-a) * b * T3 + a * b * T4);

}

extern "C" {
int cpu_convolve(sImage2d * image, sPositions2d * positions, sPSF * psf){
    info_print("Testing CPU convolution\n");
    float px, py, pixel;
    float factor_x, factor_y, center_x, center_y;
    int offset_position = 0;
    int offset_image = 0;

    // Factor de conversión entre el pixel de la PSF y el de la imagen
    factor_x = image->pixel_width / psf->pixel_width;
    factor_y = image->pixel_height / psf->pixel_height;

    // Centro de la PSF
    center_x = psf->width / 2.0;
    center_y = psf->height / 2.0;

    for (int k=0; k<positions->N; k++){
        for (int i=0; i<image->width; i++){
            for (int j=0; j<image->height; j++){
                pixel = 0;
                for (int l=0; l<positions->n; l++){
                    px = (i-positions->data[offset_position + l*2]) * factor_x + center_x;
                    py = (j-positions->data[offset_position + l*2+1]) * factor_y + center_y;

                    // Evaluación realizada por la TMU simulada en CPU para las coordenadas dadas
                    pixel += cpu_tex2d(psf, px, py);
                }
                image->data[offset_image + j * image->width + i] = pixel;
            }
        }
        offset_position += positions->n * 2;
        offset_image += image->width * image->height;
    }

    return 0;
}}

float gaussian_2d(float x, float y, float * params){
    return params[0] * exp(-(x*x + y*y) / params[1] / 2);
}

int evaluate_psf_2d(sPSF * psf, psf_function * f, float * params){
    float x, y;
    for (float i=0; i<psf->width; i++){
        for (float j=0; j<psf->height; j++){
            x = i - (psf->width / 2.0);
            y = j - (psf->height / 2.0);
            psf->data[(int) j * psf->width + (int) i] = (*f)(x, y, params);
        }
    }
    return 0;
}
