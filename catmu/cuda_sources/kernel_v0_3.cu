// """
// Este kernel combina la utilizacion de la memoria compartida para almacenar
// la lista de posiciones con la interpolación de la PSF LUT que aplica la 
// Unidad de Mapeo de Texturas (TMU) para calcular la convolución.
//
// La PSF LUT puede indicar un factor de escala respecto de la unidad utilizada
// para la imagen resultante. Por ejemplo, si cada pixel de la PSF mide A y
// los pixels de la imagen resultante miden B, deberá indicarse:
//
// psf.w_size = 
// """ 
#include "convolution_lut.h"

__global__ void lutKernel2D(sImage2d image, Positions2d pos, sPSF psf, 
                            cudaTextureObject_t texPSF){

    // Worker ID
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t idy = blockIdx.y * blockDim.y + threadIdx.y;

    float px, py, pixel;
    float factor_x, factor_y, center_x, center_y;

    factor_x = image.pixel_width / psf.pixel_width;
    factor_y = image.pixel_height / psf.pixel_height;
    center_x = psf.width / 2.0;
    center_y = psf.height / 2.0;

    // Condition for valid work
    if (idx < image.width && idy < image.height) {
        pixel = 0;
        for (int i = 0; i < pos.n; i++){
            px = (idx-pos.data[i*2]) * factor_x + center_x;
            py = (idy-pos.data[i*2+1]) * factor_y + center_y;
            pixel += tex2D<float>(texPSF, px, py);
        }
        image.data[idy * image.width + idx] = pixel;
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
    CUDA_CHECK_ERROR(return err);
    return 0;
}