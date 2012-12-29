cpu: cpu.c
	gcc	cpu.c -O3 -o cpu

gpu: gpu.cu
	nvcc gpu.cu -arch=sm_20 -o gpu

all: cpu gpu
