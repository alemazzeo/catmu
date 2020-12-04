#include "catmu.h"
#include "ufunctions.h"
#include <omp.h>

// Kernel de CUDA para la convolución TMU
__global__ void exprKernel2D(sImage image, sPositions pos, sExpressionPSF psf,
                             int offset_image, int offset_position){

    // Identificación del kernel
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t idy = blockIdx.y * blockDim.y + threadIdx.y;

    float px, py, pixel;

    // Condición para calcular el pixel (que pertenezca a la imagen)
    if (idx < image.width && idy < image.height) {

        // Resultado acumulado (inicialmente en cero)
        pixel = 0;

        // Iteración sobre todas las fuentes virtuales
        for (int i = 0; i < pos.n; i++){
            // Conversión de coordenadas
            px = (idx - pos.data[offset_position + i*2]) * image.pixel_width ;
            py = (idy - pos.data[offset_position + i*2+1]) * image.pixel_height;
            // Evaluación realizada por la TMU para las coordenadas dadas
            pixel += cuda_u_function[psf.id_function](px, py, psf.params);
        }

        // Resultado aplicado a la imagen
        image.data[offset_image + idy * image.width + idx] = pixel;
    }

}

int upload_params(sExpressionPSF * h_psf, sExpressionPSF * d_psf){
    info_print("Allocating memory for params of expression in device\n");

    // Calcula el tamaño requerido
    size_t size = h_psf->n_params * sizeof(double);
    info_print("Request: %lu bytes\n", size);

    // Revisa si la memoria fue asignada antes
    if (d_psf->params != 0){
        info_print("Memory allocated in device: %lu bytes\n", d_psf->allocated_size);

        // Revisa si la memoria fue asignada antes
        if (d_psf->allocated_size != size){
            // Libera el bloque de tamaño incorrecto
            info_print("Releasing block\n");
            cudaFree(d_psf->params);
            CUDA_CHECK_ERROR(return err);
            info_print("Memory released\n");

            // Pide el bloque correcto
            info_print("Allocating memory block of %lu bytes\n", size);
            cudaMalloc(&d_psf->params, size);
            CUDA_CHECK_ERROR(return err);
            d_psf->allocated_size = size;
            info_print("Memory allocated at %p\n", d_psf->params);
        }
    } else {
        // Pide memoria
        info_print("Allocating memory block of %lu bytes\n", size);
        cudaMalloc(&d_psf->params, size);
        CUDA_CHECK_ERROR(return err);
        d_psf->allocated_size = size;
        info_print("Memory allocated at %p\n", d_psf->params);
    }

    d_psf->id_function = h_psf->id_function;
    d_psf->n_params = h_psf->n_params;
    info_print("id_function = %d\n", d_psf->id_function);
    info_print("n_params = %d\n", d_psf->n_params);
    info_print("Memory for expression params is ready to use\n");

    // Tranfiere los datos de CPU a GPU
    info_print("Uploading params from host CPU to device GPU\n");
    cudaMemcpy(d_psf->params, h_psf->params, size, cudaMemcpyHostToDevice);
    CUDA_CHECK_ERROR(return err);
    info_print("Successfully uploaded\n");

    return 0;
}

extern "C" {
int exprConvolution2D(callback checkpoint, sImage * h_image, sPositions * h_pos,
                            sExpressionPSF * h_psf, sConfig * config){
    info_print("Starting Expression Convolution Module of CaTMU\n");

    // Definición de variables
    int r = 0;
    clock_t start, end;
    double elapsed_time = 0;
    unsigned long loop_counter = 0;

    cudaArray * cuArray = 0;
    cudaTextureObject_t texObj = 0;

    cudaStream_t * stream;

    sImage d_image;
    sPositions d_pos;
    sExpressionPSF d_psf;

    d_image.data = 0;
    d_pos.data = 0;
    d_psf.params = 0;

    // Configuración del dispositivo
    info_print("Setting device\n");
    r = set_device(config->device);
    if (r != 0) return r;

    // Bucle principal para mantener la GPU en espera entre pedidos
    info_print("Main loop\n");
    while (checkpoint(elapsed_time, loop_counter) == true){
        // Control de tiempos (para rendimiento)
        start = clock();

        // Configuración de la memoria para resultados
        info_print("Setting memory for results\n");
        r = set_images(h_image, &d_image);
        if (r != 0) return r;

        // Configuración de la memoria para posiciones y subida de datos a la GPU
        info_print("Uploading data for positions\n");
        if (h_pos->dim != 2) {
            error_print("3D not implemented yet over expression methods\n");
            return -1;
        }
        info_print("n -> %d\n", h_pos->n);
        r = upload_positions(h_pos, &d_pos);
        if (r != 0) return r;

        // Configuración de la memoria para parámetros y subida de datos a la GPU
        info_print("Uploading params for expression\n");
        info_print("n -> %d\n", h_pos->n);
        r = upload_params(h_psf, &d_psf);
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
        dim3 dimGrid((h_image->width + dimBlock.x - 1) / dimBlock.x,
                     (h_image->height + dimBlock.y - 1) / dimBlock.y);

        info_print("dimGrid: %dx%d\n", dimGrid.x, dimGrid.y);
        info_print("dimBlock: %dx%d\n", dimBlock.x, dimBlock.y);

        // Offsets aplicados a los punteros de imagenes
        // y posiciones para acceder a cada uno de ellos
        int offset_image = 0;
        int offset_position = 0;

        // Carga de tareas para cada stream (iterando entre streams)
        info_print("Launching streams\n");
        for (int i = 0; i < h_pos->N; i++)
        {
            // Instancia de kernels en diferentes streams
            info_print("Convolution %d launched on stream %d\n",
                       i, i % config->n_streams);
            info_print("Image offset: %d, Position offset: %d\n",
                       offset_image, offset_position);
            exprKernel2D <<<dimGrid, dimBlock, 0, stream[i % config->n_streams]>>> \
                 (d_image, d_pos, d_psf, offset_image, offset_position);
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

        // Marca final de tiempo
        end = clock();
        // Actualización del tiempo transcurrido
        elapsed_time = ((double) (end - start)) / CLOCKS_PER_SEC;
        // Incremento del contador
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

extern "C" {
int cpu_expr_convolve2D_openmp(sImage * image, sPositions * positions,
                               sExpressionPSF * psf){

    info_print("Starting Expression Convolution over CPU powered by Open MP\n");

    info_print("width = %d\n", image->width);
    info_print("height = %d\n", image->height);
    info_print("depth = %d\n", image->depth);
    info_print("pixel_width = %f\n", image->pixel_width);
    info_print("pixel_height = %f\n", image->pixel_height);
    info_print("pixel_depth = %f\n", image->pixel_depth);

    double px, py, pixel;
    int offset_image = image->width * image->height;
    int offset_pos = 2 * positions->n;
    float pixel_width, pixel_height;
    u_func f = c_u_function[psf->id_function];

    pixel_width = image->pixel_width;
    pixel_height = image->pixel_height;

    if (positions->dim != 2) {
        error_print("3D not implemented yet over expression methods\n");
        return -1;
    }

    #pragma omp parallel for private(pixel) private(px) private(py)
    for(int k=0; k<positions->N; k++){
        info_print("Thread %d assigned to convolution %d of %d\n",
                   omp_get_thread_num(), k, positions->N);
        for(int y=0; y<image->width; y++){
            for (int x=0; x<image->height; x++){
                pixel = 0.0;
                for(int i=0; i<positions->n; i++){
                    px = (x - positions->data[k * offset_pos + i*2]) * pixel_width;
                    py = (y - positions->data[k * offset_pos + i*2 + 1]) * pixel_height;
                    pixel = pixel + f(px, py, psf->params);
                }
                image->data[k * offset_image + y * image->width + x] = pixel;
            }
        }
    }

    return 0;
}}

extern "C" {
int cpu_expr_convolve2D(sImage * image, sPositions * positions, sExpressionPSF * psf){
    info_print("Starting Expression Convolution over CPU without Open MP\n");

    info_print("width = %d\n", image->width);
    info_print("height = %d\n", image->height);
    info_print("depth = %d\n", image->depth);
    info_print("pixel_width = %f\n", image->pixel_width);
    info_print("pixel_height = %f\n", image->pixel_height);
    info_print("pixel_depth = %f\n", image->pixel_depth);

    double px, py, pixel;
    int offset_image = image->width * image->height;
    int offset_pos = 2 * positions->n;
    float pixel_width, pixel_height;
    u_func f = c_u_function[psf->id_function];

    pixel_width = image->pixel_width;
    pixel_height = image->pixel_height;

    for(int k=0; k<positions->N; k++){
        info_print("Convolution %d of %d in progress\n", k, positions->N);
        for(int y=0; y<image->width; y++){
            for (int x=0; x<image->height; x++){
                pixel = 0.0;
                for(int i=0; i<positions->n; i++){
                    px = (x - positions->data[k * offset_pos + i*2]) * pixel_width;
                    py = (y - positions->data[k * offset_pos + i*2 + 1]) * pixel_height;
                    pixel = pixel + f(px, py, psf->params);
                }
                image->data[k * offset_image + y * image->width + x] = pixel;
            }
        }
    }

    return 0;
}}
