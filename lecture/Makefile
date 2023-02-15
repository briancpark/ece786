# Location of the CUDA Toolkit
NVCC := /usr/local/cuda/bin/nvcc
CCFLAGS := -O2

build: memcopy

memcopy.o:memcopy.cu
	$(NVCC) $(INCLUDES) $(CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

memcopy: memcopy.o
	$(NVCC) $(LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

run: build
	$(EXEC) ./memcopy

clean:
	rm -f memcopy *.o
