# Advanced HPC examples

Each of the example subdirectories contains a Makefile and a job
submission script (note that the `gpu` partition is used).

The examples each require additional modules to be loaded before they
can be compiled and run.

## OpenCL

    module load CUDA/8.0.44

## OpenMP

    module use /mnt/storage/scratch/jp8463/modules/modulefiles
    module load clang-ykt

## Kokkos

    module use /mnt/storage/scratch/jp8463/modules/modulefiles
    module load kokkos/2.5/gpu
