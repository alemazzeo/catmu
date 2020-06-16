// """
// Este kernel utiliza la Unidad de Mapeo de Texturas (TMU) para
// ajustar la PSF LUT al tamaño de la imagen.
// Por ejemplo, para una LUT de 25x25 aplicada a una imagen de 64x64
// cada pixel (X, Y) será ocupado por la posición (25/64 * X, 25/64 * Y).
//
// Los valores de la LUT son interpolados por la TMU como se indica en la 
// sección texture-fetching de la documentación de CUDA.
//
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
// """ 
#include "convolution_lut.h"

__global__ void lutKernel2D(sImage2d image, Positions2d pos, sPSF psf, 
                            cudaTextureObject_t texPSF){

    // Worker ID
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t idy = blockIdx.y * blockDim.y + threadIdx.y;

    // Condition for valid work
    if (idx < image.width && idy < image.height) {
        float px = ((float) idx / (image.width - 1) + pos.data[0]) * (psf.width - 1) + 0.5;
        float py = ((float) idy / (image.height - 1) + pos.data[1]) * (psf.height - 1) + 0.5;
        image.data[idy * image.width + idx] = tex2D<float>(texPSF, px, py);
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
    lutKernel2D <<<dimGrid, dimBlock >>> (*d_image, *d_pos, *h_psf, *texObj);
    info_print("Kernels finished\n");
    return 0;
}