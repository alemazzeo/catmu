#include <math.h>

// This macros allows compatibility with CUDA and C at same time

#define PSF_2D(NAME, EXPRESSION) \
    __device__ double cuda_##NAME(double x, double y, double *p){return EXPRESSION;} \
    double c_##NAME(double x, double y, double *p){return EXPRESSION;}

#define PSF_2D_ML(NAME, EXPRESSION) \
    __device__ double cuda_##NAME(double x, double y, double *p){EXPRESSION;} \
    double c_##NAME(double x, double y, double *p){EXPRESSION;}

// ---------------------------------------------------------------------------
// PSFs FUNCTIONS
// ---------------------------------------------------------------------------

    // Python class name: GaussianPSF
    // p[0] = amplitude
    // p[1] = sigma2 (sigma cuadrado)
    // p[2] = offset
    PSF_2D(gaussian2D, p[0] * exp(-(x*x + y*y) / p[1] / 2) + p[2]);

    // Python class name: EllipticalGaussianPSF
    // p[0] = amplitude
    // p[1] = sigma2_x (sigma cuadrado en el eje X)
    // p[2] = sigma2_y (sigma cuadrado en el eje Y)
    // p[3] = offset
    PSF_2D(elliptical_gaussian2D, \
           p[0] * exp(-((x*x) / p[1] + (y*y) / p[2]) / 2) + p[3]);

    // Python class name: GaussianPSF_RC
    // p[0] = amplitude
    // p[1] = sigma2 (sigma cuadrado)
    // p[2] = offset
    // p[3] = cutoff radius
    PSF_2D_ML(gaussian2D_rc,                                        \
        double r = x*x + y*y;                                       \
        if (r > p[3]){                                              \
            return p[2];                                            \
        } else {                                                    \
            return p[0] * exp(-(x*x + y*y) / p[1] / 2) + p[2];      \
        });                                                         \

    // Python class name: GaussianWithHaloPSF
    // p[0] = amplitude_1
    // p[1] = sigma2_1 (sigma cuadrado)
    // p[2] = amplitude_2
    // p[3] = center_radius_2
    // p[4] = sigma2_2 (sigma cuadrado)
    // p[5] = offset
    PSF_2D(gaussian_with_halo2D,
           p[0] * exp(-(x*x + y*y) / p[1] / 2) + \
           p[2] * (x*x + y*y) * \
           exp(-(powf(sqrtf(x*x + y*y)-p[3],2)) / p[4] / 2) + p[5]);

    // Python class name: FinchPSF
    // p[0] = k
    PSF_2D_ML(finch_psf,					                        \
	double r = sqrtf(x*x + y*y);				                    \
	double rf = p[0]*r;					                            \
        if (rf != 0){		                 		                \
            return j1(rf) / rf;    				                    \
        } else {                           			                \
            return 0.5;       					                    \
        });                                			                \

// ---------------------------------------------------------------------------
// TEMPLATES
// ---------------------------------------------------------------------------

    // New one-line 2D PSF function template
    // -----------------------------------------------------------------------
    // PSF_2D(NAME, EXPRESSION);
    // -----------------------------------------------------------------------

    // New multiple-line 2D PSF function template
    // -----------------------------------------------------------------------
    // PSF_2D_ML(NAME, INSTRUCTION1;
    //                 INSTRUCTION2;
    //                 ...
    //                 return RESULT;);
    // -----------------------------------------------------------------------

typedef double (*u_func)(double, double, double * params);

// List of available functions (ADD NEW FUNCTION NAMES INTO VECTOR)
__device__ u_func cuda_u_function[] = {cuda_gaussian2D,
                                       cuda_elliptical_gaussian2D,
                                       cuda_gaussian2D_rc,
                                       cuda_gaussian_with_halo2D,
                                       cuda_finch_psf};

u_func c_u_function[] = {c_gaussian2D,
                         c_elliptical_gaussian2D,
                         c_gaussian2D_rc,
                         c_gaussian_with_halo2D,
                         c_finch_psf};

// +---------+-----------------------+---------------------------------------+
// | FUNC_ID |          NAME         |      PARAMETERS p[0], p[1], ...       |
// +---------+-----------------------+---------------------------------------+
// |    0    | gaussian2D            | amplitude, sigma2, offset             |
// |    1    | elliptical_gaussian2D | amplitude, sigma2_x, sigma2_y, offset |
// |    2    | gaussian2D_rc         | amplitude, sigma2, offset, cutoff     |
// |    3    | gaussian_with_halo2D  | amplitude_1, sigma2_1, amplitude_2,...|
// |         |                       | center_radius_2, sigma2_2, offset     |
// |    4    | finch_psf	         | k				                     |
// +---------+-----------------------+---------------------------------------+
