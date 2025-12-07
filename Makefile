ARCH ?= sm_60
HOST_COMP ?= mpicc

NVCC      ?= nvcc
NVCCFLAGS ?= -O3 -std=c++17 -arch=$(ARCH) -ccbin=$(HOST_COMP)

TARGET3   ?= sol3
SRC3      := 3. MPI + CUDA/solution3.cu

.PHONY: all mpi_cuda clean

all: mpi_cuda

mpi_cuda: $(TARGET3)

$(TARGET3): $(SRC3)
	$(NVCC) $(NVCCFLAGS) -o $@ "$(SRC3)"

clean:
	rm -f $(TARGET3)


