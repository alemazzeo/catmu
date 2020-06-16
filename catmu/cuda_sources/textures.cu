#include "convolution_lut.h"

int set_texture2d(cudaArray * cuArray, sPSF * psf, 
                   cudaTextureObject_t * texObj, bool normalized){

    info_print("Allocate CUDA array in device memory\n");
    cudaChannelFormatDesc channelDesc =
               cudaCreateChannelDesc(32, 0, 0, 0,
                                     cudaChannelFormatKindFloat);
    cudaMallocArray(&cuArray, &channelDesc, psf->width, psf->height);

    CUDA_CHECK_ERROR(return err);

    info_print("Copy to device memory sPSF data from host\n");
    /*cudaMemcpyToArray(cuArray, 0, 0, psf->data, 
                      (psf->width * psf->height) * sizeof(float), 
                      cudaMemcpyHostToDevice);*/

    cudaMemcpy2DToArray(cuArray, 0, 0, psf->data,
                        psf->width * sizeof(float),
                        psf->width * sizeof(float),
                        psf->height,
                        cudaMemcpyHostToDevice);

    CUDA_CHECK_ERROR(return err);

    info_print("Specify texture\n");
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;
    CUDA_CHECK_ERROR(return err);

    info_print("Specify texture object parameters\n");
    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0]   = cudaAddressModeBorder;
    texDesc.addressMode[1]   = cudaAddressModeBorder;
    texDesc.filterMode       = cudaFilterModeLinear;
    texDesc.readMode         = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;
    CUDA_CHECK_ERROR(return err);

    info_print("Create texture object\n");
    cudaCreateTextureObject(texObj, &resDesc, &texDesc, NULL);
    CUDA_CHECK_ERROR(return err);

    return 0;

}

int free_texture(cudaArray * cuArray, cudaTextureObject_t * texObj){
    info_print("Destroy texture object\n");
    cudaDestroyTextureObject(*texObj);
    info_print("Free texture memory\n");
    cudaFreeArray(cuArray);
    return 0;
}