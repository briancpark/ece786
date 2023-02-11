# Location of the CUDA Toolkit
NVCC := /usr/local/cuda/bin/nvcc
EXTRA_NVCCFLAGS := --cudart=shared
CCFLAGS := -O3 -std=c++11

all : build quamsimV1_benchmark quamsimV2_benchmark

build: quamsimV1 quamsimV2

quamsimV1.o:quamsimV1.cu
	$(NVCC) $(INCLUDES) $(CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

quamsimV2.o:quamsimV2.cu
	$(NVCC) $(INCLUDES) $(CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

quamsimV1: quamsimV1.o
	$(NVCC) $(LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

quamsimV2: quamsimV2.o
	$(NVCC) $(LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

### Benchmark target recipes

quamsimV1_benchmark.o:quamsimV1.cu
	$(NVCC) $(INCLUDES) $(CCFLAGS) $(GENCODE_FLAGS) -DBENCHMARK -o $@ -c $<

quamsimV2_benchmark.o:quamsimV2.cu
	$(NVCC) $(INCLUDES) $(CCFLAGS) $(GENCODE_FLAGS) -DBENCHMARK -o $@ -c $<

quamsimV1_benchmark: quamsimV1_benchmark.o
	$(NVCC) $(LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

quamsimV2_benchmark: quamsimV2_benchmark.o
	$(NVCC) $(LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

clean:
	rm -f quamsimV1 quamsimV2 quamsimV1_benchmark quamsimV2_benchmark *.o
