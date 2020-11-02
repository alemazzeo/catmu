NVCC_FLAGS = -rdc=true
NVCC_XFLAGS = -Xcompiler -fPIC -shared

all: libConvolveLUT.so libConvolveLUTd.so

libConvolveLUT.so: lutConvolution.cu
	nvcc $(NVCC_FLAGS) $^ $(NVCC_XFLAGS) -o $@

libConvolveLUTd.so: lutConvolution.cu
	nvcc $(NVCC_FLAGS) $^ -DDEBUG $(NVCC_XFLAGS) -o $@
