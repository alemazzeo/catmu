#include "convolution_lut.h"

__global__ void lutKernel2D(int sub_pixel, sImage2d image, Positions2d pos, sPSF psf,
                            cudaTextureObject_t texPSF){

    // Worker ID
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t idy = blockIdx.y * blockDim.y + threadIdx.y;

    int x, y;
    float px, py, pixel;
    float factor_x, factor_y, center_x, center_y;

    factor_x = image.pixel_width / psf.pixel_width;
    factor_y = image.pixel_height / psf.pixel_height;
    center_x = psf.width / 2.0;
    center_y = psf.height / 2.0;

    for (int j = 0; j < sub_pixel; j++){
        for (int k = 0; k < sub_pixel; k++){
            x = idx * sub_pixel + j;
            y = idy * sub_pixel + k;
            // Condition for valid work
            if (idx < image.width && idy < image.height) {
                pixel = 0;
                for (int i = 0; i < pos.n; i++){
                    px = (x-pos.data[i*2]) * factor_x + center_x;
                    py = (y-pos.data[i*2+1]) * factor_y + center_y;
                    pixel += tex2D<float>(texPSF, px, py);
                }
                image.data[idy * image.width + idx] = pixel;
            }
        }
    }
}

int set_device(int device){
    int count, current_device;
    cudaGetDeviceCount(&count);
    if (device > count){
        return -1;
    }
    info_print("Selecting device: %d\n", device);
    cudaSetDevice(device);
    CUDA_CHECK_ERROR(return err);
    cudaGetDevice(&current_device);
    info_print("Current device: %d\n", current_device);
    if (current_device != device){
        return -2;
    }
    return 0;
}

int free_device_memory(sImage2d * d_image, Positions2d * d_pos,
                       cudaTextureObject_t * texObj, cudaArray * cuArray){

    cudaFree(d_image->data);
    cudaFree(d_pos->data);
    CUDA_CHECK_ERROR();

    return 0;
}

extern "C" {
int lutConvolution2D(sImage2d * h_image, Positions2d * h_pos, sPSF * h_psf, int N,
                     int subpixel, int device){
    int r = 0;

    cudaArray * cuArray = 0;
    cudaTextureObject_t texObj = 0;

    cudaStream_t * stream;

    size_t result_size;
    size_t pos_size;


    sImage2d * d_image;
    sPositions2d * d_pos;

    d_image = (sImage2d *) malloc(N * sizeof(sImage2d));
    d_pos = (sPositions2d *) malloc(N * sizeof(Positions2d));
    stream = (cudaStream_t *) malloc(N * sizeof(cudaStream_t));
    
    info_print("Setting device\n");
    set_device(device);

    info_print("Setting PSF as texture object\n");
    r = set_texture2d(cuArray, h_psf, &texObj, true);
    if (r != 0) return r;

    info_print("psf->width = %d\n", h_psf->width);
    info_print("psf->height = %d\n", h_psf->height);
    info_print("psf->pixel_width = %f\n", h_psf->pixel_width);
    info_print("psf->pixel_height = %f\n", h_psf->pixel_height);
    info_print("psf->data[%d] = %f\n",
               (h_psf->width * h_psf-> height) / 2,
               h_psf->data[(h_psf->width * h_psf-> height) / 2]);

    info_print("Setting %d streams\n", N);
    for (int i = 0; i < N; i ++)
    {
        cudaStreamCreate(&stream[i]);
        CUDA_CHECK_ERROR(return err);
        info_print("Stream %d created\n", i);
    }

    info_print("Allocating images and positions with metadata\n");
    for (int i=0; i<N; i++){
        info_print("Image %d of %d\n", i+1, N);
        info_print("Input parameters:\n");
        info_print("image[%d].width = %d\n", i, h_image[i].width);
        info_print("image[%d].height = %d\n", i, h_image[i].height);
        info_print("image[%d].pixel_width = %f\n", i, h_image[i].pixel_width);
        info_print("image[%d].pixel_height = %f\n", i, h_image[i].pixel_height);

        info_print("pos[%d].n = %d\n", i, h_pos[i].n);
        info_print("pos[%d].data[0] = %f\n", i, h_pos[i].data[0]);
        info_print("pos[%d].data[1] = %f\n", i, h_pos[i].data[1]);

        info_print("Allocating memory for result in device\n");

        d_image[i].width = h_image[i].width;
        d_image[i].height = h_image[i].height;
        d_image[i].pixel_width = h_image[i].pixel_width;
        d_image[i].pixel_height = h_image[i].pixel_height;

        result_size = h_image[i].width * h_image[i].height * sizeof(float);
        info_print("Allocated %ld bytes on %p (device)\n",
                   result_size, d_image[i].data);
        cudaMalloc(&d_image[i].data, result_size);
        CUDA_CHECK_ERROR(return err);
        info_print("Allocated %ld bytes on %p (device)\n",
                   result_size, d_image[i].data);

        info_print("Allocating memory for positions coordinates in device\n");

        d_pos[i].n = h_pos[i].n;
        pos_size = h_pos[i].n * sizeof(float) * 2;
        cudaMalloc(&d_pos[i].data, pos_size);
        CUDA_CHECK_ERROR(return err);
    }

    info_print("Launching streams\n");
    for (int i = 0; i < N; i++)
    {
        pos_size = h_pos[i].n * sizeof(float) * 2;
        result_size = h_image[i].width * h_image[i].height * sizeof(float);

        info_print("Grid and block sizes:\n");
        dim3 dimBlock(8, 8);
        dim3 dimGrid((h_image[i].width  + dimBlock.x - 1) / dimBlock.x,
                     (h_image[i].height + dimBlock.y - 1) / dimBlock.y);

        info_print("dimGrid: %dx%d\n", dimGrid.x, dimGrid.y);
        info_print("dimBlock: %dx%d\n", dimBlock.x, dimBlock.y);

        info_print("Asynchronous memcpy positions to device\n");
        cudaMemcpyAsync(d_pos[i].data, h_pos[i].data, pos_size, cudaMemcpyHostToDevice, stream[i]);
        CUDA_CHECK_ERROR(return err);

        info_print("Asynchronous kernel launch\n");
        lutKernel2D <<<dimGrid, dimBlock, 0, stream[i]>>> (subpixel, d_image[i], d_pos[i], *h_psf, texObj);
        CUDA_CHECK_ERROR(return err);

        info_print("Asynchronous memcpy image to host\n");
        cudaMemcpyAsync(h_image[i].data, d_image[i].data, result_size, cudaMemcpyDeviceToHost, stream[i]);
        CUDA_CHECK_ERROR(return err);
    }

    info_print("Device synchronize\n");
    cudaDeviceSynchronize();
    CUDA_CHECK_ERROR(return err);

    info_print("Destroying %d streams\n", N);
    for (int i = 0; i < N; i ++)
    {
        cudaStreamDestroy(stream[i]);
        CUDA_CHECK_ERROR(return err);
        info_print("Stream %d destroyed\n", i);
    }

    info_print("Free device memory\n");
    free_texture(cuArray, &texObj);

    free(d_image);
    free(d_pos);

    info_print("Executing reset of device\n");
    cudaDeviceReset();
    CUDA_CHECK_ERROR(return err);
    info_print("\nBye\n");

    return 0;
}}
