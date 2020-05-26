#include <stdio.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#ifdef DEBUG
#define DEBUG_TEST 1
#else
#define DEBUG_TEST 0
#endif

// # MACRO info_print()
#define info_print(...) \
    do { if (DEBUG_TEST) {fprintf(stdout, "INFO : %s(%d) : ", \
                                  __FUNCTION__, __LINE__); \
                          fprintf(stdout, __VA_ARGS__ );} \
    } while (0)

// # MACRO error_print()
#define error_print(...) \
    do { fprintf(stderr, "ERROR : %s(%d) : ", __FUNCTION__, __LINE__); \
         fprintf(stderr, __VA_ARGS__ ); \
    } while (0)

// # MACRO CUDA_CHECK_ERROR()
#define CUDA_CHECK_ERROR(X) \
    do {cudaError_t err; \
        err = cudaGetLastError(); \
        if (err != cudaSuccess) { \
            error_print("CODE %d -> %s\n", err, cudaGetErrorString(err)); \
            X; \
        }} while(0)

// # ESTRUCTURA para la imagen convolucionada (resultado final)
typedef struct sImage2d{
    int width;
    int height;
    float pixel_width;
    float pixel_height;
    float * data;
} sImage2d;

// # ESTRUCTURA para la lista de posiciones 2D
typedef struct sPositions2d{
    int n;
    float * data;
} Positions2d;

// # ESTRUCTURA para la LUT de la PSF
typedef struct sPSF{
    int width;
    int height;
    float pixel_width;
    float pixel_height;
    float * data;
} sPSF;

// # Declaración de funciones utilizadas
int set_texture2d(cudaArray * cuArray, sPSF * psf, 
                  cudaTextureObject_t * texObj, bool normalized);

int free_texture(cudaArray * cuArray, cudaTextureObject_t * texObj);

int set_device(int device);

int upload_data(sImage2d * h_image, Positions2d * h_pos, sPSF * h_psf,
                sImage2d * d_image, Positions2d * d_pos,
                cudaTextureObject_t * texObj, cudaArray * cuArray);

int launch_kernels(sImage2d * h_image, Positions2d * h_pos, sPSF * h_psf, 
                   sImage2d * d_image, Positions2d * d_pos, 
                   cudaTextureObject_t * texObj);
                
int download_data(sImage2d * h_image, Positions2d * h_pos, sPSF * h_psf,
                  sImage2d * d_image, Positions2d * d_pos,
                  cudaTextureObject_t * texObj, cudaArray * cuArray);