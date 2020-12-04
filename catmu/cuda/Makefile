NVCC_FLAGS = -rdc=true -lgomp
NVCC_XFLAGS = -Xcompiler -fPIC -Xcompiler -fopenmp -shared

all: libConvolve.so libConvolveDebug.so

libConvolve.so: lutConvolution.cu exprConvolution.cu
	nvcc $(NVCC_FLAGS) $^ $(NVCC_XFLAGS) -o $@

libConvolveDebug.so: lutConvolution.cu exprConvolution.cu
	nvcc $(NVCC_FLAGS) $^ -DDEBUG $(NVCC_XFLAGS) -o $@
