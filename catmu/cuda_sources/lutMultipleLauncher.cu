#include "convolution_lut.h"

__global__ void lutKernel2D(int sub_pixel, sImage2d image, Positions2d pos, sPSF psf,
                            cudaTextureObject_t texPSF, int offset_image, int offset_position){

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
                    px = (x-pos.data[offset_position + i*2]) * factor_x + center_x;
                    py = (y-pos.data[offset_position + i*2+1]) * factor_y + center_y;
                    pixel += tex2D<float>(texPSF, px, py);
                }
                image.data[offset_image + y * image.width + x] = pixel;
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
                     int subpixel, int block_size, int device){
    int r = 0;

    cudaArray * cuArray = 0;
    cudaTextureObject_t texObj = 0;

    cudaStream_t * stream;

    size_t result_size;
    size_t pos_size;

    sImage2d d_image;
    sPositions2d d_pos;

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

    info_print("Allocating images and positions metadata\n");

    info_print("Input parameters:\n");
    info_print("image->width = %d\n", h_image->width);
    info_print("image->height = %d\n", h_image->height);
    info_print("image->pixel_width = %f\n", h_image->pixel_width);
    info_print("image->pixel_height = %f\n", h_image->pixel_height);

    info_print("pos->n = %d\n", h_pos->n);
    info_print("pos->data[0] = %f\n", h_pos->data[0]);
    info_print("pos->data[1] = %f\n", h_pos->data[1]);

    info_print("Allocating memory for result in device\n");

    d_image.width = h_image->width;
    d_image.height = h_image->height;
    d_image.pixel_width = h_image->pixel_width;
    d_image.pixel_height = h_image->pixel_height;

    result_size = h_image->width * h_image->height * sizeof(float);
    info_print("Allocated %ld bytes on %p (device)\n",
               result_size, d_image.data);
    cudaMalloc(&d_image.data, result_size * N);
    CUDA_CHECK_ERROR(return err);
    info_print("Allocated %ld bytes on %p (device)\n",
               result_size, d_image.data);

    info_print("Allocating memory for positions coordinates in device\n");

    d_pos.n = h_pos->n;
    pos_size = h_pos->n * sizeof(float) * 2;
    cudaMalloc(&d_pos.data, pos_size * N);
    CUDA_CHECK_ERROR(return err);

    info_print("Grid and block sizes:\n");
    dim3 dimBlock(block_size, block_size);
    dim3 dimGrid((h_image->width / subpixel  + dimBlock.x - 1) / dimBlock.x,
                 (h_image->height / subpixel + dimBlock.y - 1) / dimBlock.y);

    info_print("dimGrid: %dx%d\n", dimGrid.x, dimGrid.y);
    info_print("dimBlock: %dx%d\n", dimBlock.x, dimBlock.y);

    int offset_position = h_pos->n * 2;
    int offset_image = h_image->width * h_image->height;

    cudaMemcpy(d_pos.data, h_pos->data, pos_size * N, cudaMemcpyHostToDevice);

    info_print("Launching streams\n");
    for (int i = 0; i < N; i++)
    {

//        info_print("Asynchronous memcpy positions to device\n");
//        cudaMemcpyAsync(&d_pos.data[i * offset_position], &h_pos->data[i * offset_position],
//                        pos_size, cudaMemcpyHostToDevice, stream[i]);
//        CUDA_CHECK_ERROR(return err);

        info_print("Asynchronous kernel launch\n");
        lutKernel2D <<<dimGrid, dimBlock, 0, stream[i]>>> (subpixel, d_image, d_pos, *h_psf, texObj,
                                                           offset_image * i, offset_position * i);
        CUDA_CHECK_ERROR(return err);

//        info_print("Asynchronous memcpy image to host\n");
//        cudaMemcpyAsync(&h_image->data[i * offset_image], &d_image.data[i * offset_image],
//                        result_size, cudaMemcpyDeviceToHost, stream[i]);
//        CUDA_CHECK_ERROR(return err);
    }

    info_print("Device synchronize\n");
    cudaDeviceSynchronize();
    CUDA_CHECK_ERROR(return err);

    cudaMemcpy(h_image->data, d_image.data, result_size * N, cudaMemcpyDeviceToHost);

    info_print("Destroying %d streams\n", N);
    for (int i = 0; i < N; i ++)
    {
        cudaStreamDestroy(stream[i]);
        CUDA_CHECK_ERROR(return err);
        info_print("Stream %d destroyed\n", i);
    }

    info_print("Free device memory\n");
    free_texture(cuArray, &texObj);

    info_print("Executing reset of device\n");
    cudaDeviceReset();
    CUDA_CHECK_ERROR(return err);
    info_print("\nBye\n");

    return 0;
}}
