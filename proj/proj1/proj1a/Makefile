# Location of the CUDA Toolkit
NVCC := nvcc
EXTRA_NVCCFLAGS := --cudart=shared
CCFLAGS := -O3 -std=c++11

# Enable BENCHMARK macro if BENCHMARK is set to true or 1
ifdef BENCHMARK
CCFLAGS += -DBENCHMARK
endif

all : build

build: quamsimV1 quamsimV2

quamsimV1.o:quamsimV1.cu
	$(NVCC) $(INCLUDES) $(CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

quamsimV2.o:quamsimV2.cu
	$(NVCC) $(INCLUDES) $(CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

quamsimV1: quamsimV1.o
	$(NVCC) $(LDFLAGS) $(EXTRA_NVCCFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

quamsimV2: quamsimV2.o
	$(NVCC) $(LDFLAGS) $(EXTRA_NVCCFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

clean:
	rm -f quamsimV1 quamsimV2 *.o
