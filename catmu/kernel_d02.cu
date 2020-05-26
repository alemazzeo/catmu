// """
// Este kernel de prueba transfiere la lista de posiciones
// de la memoria global a la memoria compartida.
// Marca las posiciones sobre la imagen para verificarlas
// Marca las posiciones sobre la imagen para verificarlas
// utilizando la función de ejemplo f(r) = exp(-r**2).
// """ 
#include "convolution_lut.h"

__global__ void lutKernel2D(sImage2d image, Positions2d pos, sPSF psf, 
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

    float px, py, r2, pixel;

    // Condition for valid work
    if (idx < image.width && idy < image.height) {
        pixel = 0;
        for (int i = 0; i < pos.n; i+=2){
            px = idx-shared[i*2];
            py = idy-shared[i*2+1];
            r2 = px*px + py*py;
            pixel += exp(-r2);
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

    // Tamaño para la memoria compartida
    size_t sm = h_pos->n * sizeof(float) * 2;

    info_print("Executing kernels\n");
    lutKernel2D <<<dimGrid, dimBlock, sm>>> (*d_image, *d_pos, *h_psf, *texObj);
    info_print("Kernels finished\n");
    return 0;
}