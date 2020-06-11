// """
// Este kernel de prueba escribe sobre la imagen objetivo
// la identificación del thread utilizado
// """ 
#include "convolution_lut.h"

__global__ void lutKernel2D(sImage2d image, Positions2d pos, sPSF psf, 
                            cudaTextureObject_t texPSF){

    // Worker ID
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t idy = blockIdx.y * blockDim.y + threadIdx.y;

    // Condition for valid work
    if (idx < image.width && idy < image.height) {
        image.data[idy * image.width + idx] = idy * image.width + idx;
    }
}

int launch_kernels(sImage2d * h_image, Positions2d * h_pos, sPSF * h_psf, 
                   sImage2d * d_image, Positions2d * d_pos, 
                   cudaTextureObject_t * texObj){
    
    info_print("Kernel file: %s\n", __FILE__);

    info_print("Grid and block sizes:\n");
    dim3 dimBlock(16, 16);
    dim3 dimGrid((h_image->width  + dimBlock.x - 1) / dimBlock.x,
                 (h_image->height + dimBlock.y - 1) / dimBlock.y);

    info_print("dimGrid: %dx%d\n", dimGrid.x, dimGrid.y);
    info_print("dimBlock: %dx%d\n", dimBlock.x, dimBlock.y);

    info_print("Executing kernels\n");
    lutKernel2D <<<dimGrid, dimBlock>>> (*d_image, *d_pos, *h_psf, *texObj);
    info_print("Kernels finished\n");
    return 0;
}