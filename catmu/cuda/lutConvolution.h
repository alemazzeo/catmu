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
typedef struct sImage2d{
    int N;
    int width;
    int height;
    float pixel_width;
    float pixel_height;
    float * data;
    size_t allocated_size;
} sImage2d;

// # ESTRUCTURA para la lista de posiciones 2D
typedef struct sPositions2d{
    int N;
    int n;
    float * data;
    size_t allocated_size;
} Positions2d;

// # ESTRUCTURA para la LUT de la PSF
typedef struct sPSF{
    int width;
    int height;
    float pixel_width;
    float pixel_height;
    float * data;
    size_t allocated_size;
} sPSF;

// # ESTRUCTURA para las configuraciones generales
typedef struct sConfig{
    int device;
    int sub_pixel;
    int block_size;
    int n_streams;
} sConfig;


// Tipo de función para el callback
typedef bool callback(double elapsed_time, unsigned long loop_counter);

// Callback que inmediatamente devuelve true (anula el callback)
bool dummy_callback(double elapsed_time, unsigned long loop_counter){
    return true;
}

// Función para PSF por expresión
typedef float psf_function(float x, float y, float * params);

// # Declaración de funciones utilizadas

int set_device(int device);

int set_texture_2d(cudaArray * cuArray, sPSF * psf, cudaTextureObject_t * texObj, bool normalized);

int set_images_2d(sImage2d * h_image, sImage2d * d_image);
int set_positions_2d(Positions2d * h_pos, sPositions2d * d_pos);

double get_T(sPSF * psf, int x, int y);
double cpu_tex2d(sPSF * psf, float x, float y);
int rmse(sImage2d method1, sImage2d method2, float * result);

float gaussian_2d(float x, float y, float * params);
int evaluate_psf_2d(sPSF * psf, float amplitude, float sigma);