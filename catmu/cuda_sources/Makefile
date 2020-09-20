BIN = ../bin/
NVCC_FLAGS = -rdc=true
NVCC_XFLAGS = -Xcompiler -fPIC -shared
COMMON_FILES = textures.cu lutConvolutionLauncher.cu
KERNELS = $(wildcard kernel_*.cu)
LIBS = $(patsubst kernel_%.cu, $(BIN)libConvolveLUT_%.so, $(KERNELS))
LIBS_DEBUG = $(patsubst kernel_%.cu, $(BIN)libConvolveLUT_D_%.so, $(KERNELS))

all: nodebug debug libConvolveLUT.so libConvolveLUTd.so
	
nodebug: $(LIBS)
	
debug: $(LIBS_DEBUG)

libConvolveLUT.so: lutMultipleLauncher.cu textures.cu
	@mkdir -p $(BIN)
	nvcc $(NVCC_FLAGS) $^ $(NVCC_XFLAGS) -o $(BIN)$@

libConvolveLUTd.so: lutMultipleLauncher.cu textures.cu
	@mkdir -p $(BIN)
	nvcc $(NVCC_FLAGS) $^ -DDEBUG $(NVCC_XFLAGS) -o $(BIN)$@

$(LIBS): $(BIN)libConvolveLUT_%.so: kernel_%.cu $(COMMON_FILES)
	@mkdir -p $(BIN)
	nvcc $(NVCC_FLAGS) $^ $(NVCC_XFLAGS) -o $@

$(LIBS_DEBUG): $(BIN)libConvolveLUT_D_%.so: kernel_%.cu $(COMMON_FILES)
	@mkdir -p $(BIN)
	nvcc $(NVCC_FLAGS) $^ -DDEBUG $(NVCC_XFLAGS) -o $@
