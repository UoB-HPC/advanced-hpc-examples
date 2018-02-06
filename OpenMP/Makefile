EXE=vecadd-openmp

CUDA_PATH=/mnt/storage/easybuild/software/CUDA/8.0.44
CC=clang
CFLAGS= -std=c99 -Wall -O3 -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda --cuda-path=/nfs/modules/cuda/8.0.44/
LIBS = -lm

all: $(EXE)

$(EXE): $(EXE).c
	$(CC) $(CFLAGS) $^ $(LIBS) -o $@

clean:
	rm $(EXE)
