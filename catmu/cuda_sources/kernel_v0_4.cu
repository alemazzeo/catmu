// """
// Este kernel combina la utilizacion de la memoria compartida para almacenar
// la lista de posiciones con la interpolación de la PSF LUT que aplica la
// Unidad de Mapeo de Texturas (TMU) para calcular la convolución.
// """
#include "convolution_lut.h"

__global__ void lutKernel2D(int sub_pixel, sImage2d image, Positions2d pos, sPSF psf,
                            cudaTextureObject_t texPSF){

    // Worker ID
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t idy = blockIdx.y * blockDim.y + threadIdx.y;

    int idk = blockDim.x * threadIdx.y + threadIdx.x;

    extern __shared__ float shared[];

    if (blockDim.x * blockDim.y >= pos.n * 2){
        if (idk < pos.n * 2){
            shared[idk] = pos.data[idk];
        }
    }
    else{
        int m = pos.n * 2 / blockDim.x * blockDim.y;
        for (int i = 0; i < m; i++){
            if (idk * m + i < pos.n * 2){
                shared[idk * m + i] = pos.data[idk * m + i];
            }
        }
    }

    __syncthreads();

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
                    px = (idx-shared[i*2]) * factor_x + center_x;
                    py = (idy-shared[i*2+1]) * factor_y + center_y;
                    pixel += tex2D<float>(texPSF, px, py);
                }
                image.data[idy * image.width + idx] = pixel;
            }
        }
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

    // Tamaño para la memoria compartida
    size_t sm = h_pos->n * sizeof(float) * 2;

    info_print("Executing kernels\n");
    lutKernel2D <<<dimGrid, dimBlock, sm>>> (1, *d_image, *d_pos, *h_psf, *texObj);
    info_print("Kernels finished\n");
    CUDA_CHECK_ERROR(return err);
    return 0;
}