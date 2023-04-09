# Location of the CUDA Toolkit
NVCC := /usr/local/cuda/bin/nvcc
EXTRA_NVCCFLAGS := --cudart=shared
CCFLAGS := -O3 -std=c++11

all : build

build: quamsimV1 quamsimV2 quamsimV3

quamsimV1.o:quamsimV1.cu
	$(NVCC) $(INCLUDES) $(CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

quamsimV2.o:quamsimV2.cu
	$(NVCC) $(INCLUDES) $(CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

quamsimV3.o:quamsimV3.cu
	$(NVCC) $(INCLUDES) $(CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

quamsimV1: quamsimV1.o
	$(NVCC) $(LDFLAGS) $(EXTRA_NVCCFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

quamsimV2: quamsimV2.o
	$(NVCC) $(LDFLAGS) $(EXTRA_NVCCFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

quamsimV3: quamsimV3.o
	$(NVCC) $(LDFLAGS) $(EXTRA_NVCCFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

clean:
	rm -f quamsimV1 quamsimV2 *.o
