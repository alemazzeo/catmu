#include "convolution_lut.h"

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

int upload_data(sImage2d * h_image, Positions2d * h_pos, sPSF * h_psf,
                sImage2d * d_image, Positions2d * d_pos,
                cudaTextureObject_t * texObj, cudaArray * cuArray) {

    int r = 0;

    info_print("Input parameters:\n");
    info_print("image->width = %d\n", h_image->width);
    info_print("image->height = %d\n", h_image->height);
    info_print("image->pixel_width = %f\n", h_image->pixel_width);
    info_print("image->pixel_height = %f\n", h_image->pixel_height);

    info_print("psf->width = %d\n", h_psf->width);
    info_print("psf->height = %d\n", h_psf->height);
    info_print("psf->pixel_width = %f\n", h_psf->pixel_width);
    info_print("psf->pixel_height = %f\n", h_psf->pixel_height);
    info_print("psf->data[%d] = %f\n", 
               (h_psf->width * h_psf-> height) / 2, 
               h_psf->data[(h_psf->width * h_psf-> height) / 2]);

    info_print("pos->n = %d\n", h_pos->n);
    info_print("pos->data[0] = %f\n", h_pos->data[0]);
    info_print("pos->data[1] = %f\n", h_pos->data[1]);

    info_print("Setting PSF as texture object\n");
    r = set_texture2d(cuArray, h_psf, texObj, true);
    if (r != 0) return r;

    info_print("Allocating memory for result in device\n");

    d_image->width = h_image->width;
    d_image->height = h_image->height;
    d_image->pixel_width = h_image->pixel_width;
    d_image->pixel_height = h_image->pixel_height;

    size_t result_size = h_image->width * h_image->height * sizeof(float);
    info_print("Allocated %ld bytes on %p (device)\n", 
               result_size, d_image->data);
    cudaMalloc(&d_image->data, result_size);
    CUDA_CHECK_ERROR(return err);
    info_print("Allocated %ld bytes on %p (device)\n", 
               result_size, d_image->data);

    info_print("Allocating memory for positions coordinates in device\n");
    
    d_pos->n = h_pos->n;
    size_t pos_size = h_pos->n * sizeof(float) * 2;
    cudaMalloc(&d_pos->data, pos_size);
    CUDA_CHECK_ERROR(return err);

    info_print("Copying image memory from %p (host) to %p (device)\n", 
               h_pos->data, d_pos->data);
    cudaMemcpy(d_pos->data, h_pos->data, pos_size, cudaMemcpyHostToDevice);
    info_print("Copied %ld bytes\n", pos_size);
    CUDA_CHECK_ERROR(return err);

    return 0;
}

int download_data(sImage2d * h_image, Positions2d * h_pos, sPSF * h_psf,
                  sImage2d * d_image, Positions2d * d_pos,
                  cudaTextureObject_t * texObj, cudaArray * cuArray){
    
    size_t result_size = h_image->width * h_image->height * sizeof(float);

    info_print("Copying image memory from %p (device) to %p (host)\n", 
               d_image->data, h_image->data);
    cudaMemcpy(h_image->data, d_image->data, result_size, 
               cudaMemcpyDeviceToHost);
    CUDA_CHECK_ERROR();
    
    info_print("Free device memory\n");
    free_texture(cuArray, texObj);
    cudaFree(d_image->data);
    cudaFree(d_pos->data);

    return 0;
}

extern "C" {
int lutConvolution2D(sImage2d * h_image, Positions2d * h_pos, sPSF * h_psf, 
                     int device){
    int r = 0;

    cudaArray * cuArray = 0;
    cudaTextureObject_t texObj = 0;
    sImage2d d_image;
    Positions2d d_pos;
    
    info_print("Setting device\n");
    set_device(device);

    info_print("Uploading data to device\n");
    r = upload_data(h_image, h_pos, h_psf, &d_image, &d_pos, &texObj, cuArray);
    if (r != 0) return r;
    CUDA_CHECK_ERROR(return err);

    info_print("Executing kernels\n");
    r = launch_kernels(h_image, h_pos, h_psf, &d_image, &d_pos, &texObj);
    if (r != 0) return r;
    CUDA_CHECK_ERROR(return err);

    info_print("Downloading data from device\n");
    r = download_data(h_image, h_pos, h_psf, &d_image, &d_pos, &texObj, cuArray);
    if (r != 0) return r;
    CUDA_CHECK_ERROR(return err);

    return 0;
}}