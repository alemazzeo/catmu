#include <stdio.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <math.h>

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
typedef struct sImage{
    int N;
    int width;
    int height;
    int depth;
    float pixel_width;
    float pixel_height;
    float pixel_depth;
    float * data;
    size_t allocated_size;
} sImage;

// # ESTRUCTURA para la lista de posiciones 2D
typedef struct sPositions{
    int N;
    int n;
    int dim;
    float * data;
    size_t allocated_size;
} sPositions2d;

// # ESTRUCTURA para la LUT de la PSF
typedef struct sLutPSF{
    int width;
    int height;
    int depth;
    int dim;
    float pixel_width;
    float pixel_height;
    float pixel_depth;
    float * data;
    size_t allocated_size;
} sLutPSF2d;

// # ESTRUCTURA para PSF por expresión
typedef struct sExpressionPSF{
    int id_function;
    int n_params;
    double * params;
    size_t allocated_size;
} sExpressionPSF;

// # ESTRUCTURA para las configuraciones generales
typedef struct sConfig{
    int device;
    int block_size;
    int n_streams;
} sConfig;

typedef struct cudaDevicePropCatmu {
    char name[256];
    int multiProcessorCount;
    size_t totalGlobalMem;
    size_t sharedMemPerBlock;
    int major;
    int minor;
    int regsPerBlock;
    int warpSize;
    size_t memPitch;
    int maxThreadsPerBlock;
    int maxThreadsDim[3];
    int maxGridSize[3];
    int clockRate;
    size_t totalConstMem;
    size_t textureAlignment;
    int deviceOverlap;
    int kernelExecTimeoutEnabled;
} cudaDevicePropCatmu;


// Tipo de función para el callback
typedef bool callback(double elapsed_time, unsigned long loop_counter);

// Función para PSF por expresión
typedef float psf_function(float x, float y, float * params);

// # Declaración de funciones utilizadas

int set_device(int device);

int set_texture_2d(cudaArray * cuArray, sLutPSF * psf,
                   cudaTextureObject_t * texObj, bool normalized);
int set_texture_3d(cudaArray * cuArray, sLutPSF * psf,
                   cudaTextureObject_t * texObj, bool normalized);

int set_images(sImage * h_image, sImage * d_image);
int set_positions(sPositions * h_pos, sPositions * d_pos);
int upload_positions(sPositions * h_pos, sPositions * d_pos);
int upload_params(sExpressionPSF * h_params, sExpressionPSF * d_params);
int download_results(sImage * h_image, sImage * d_image);

int free_device_memory(sImage * d_image, sPositions * d_pos,
                       cudaTextureObject_t * texObj, cudaArray * cuArray);

double get_T(sLutPSF * psf, int x, int y);
double cpu_tex2d(sLutPSF * psf, float x, float y);
int rmse(sImage method1, sImage method2, float * result);

float gaussian_2d(float x, float y, float * params);
int evaluate_psf_2d(sLutPSF * psf, float amplitude, float sigma);