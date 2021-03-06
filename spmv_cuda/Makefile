#compilers
CC=nvcc
CXX=g++
MPICXX=mpicxx

CFLAGS = -lmpi
CXXFLAGS = -std=c++11
CUFLAGS="-D__LP64__ -ta:tesla"

#MPIFLAGS = -I/apps/pkg/openmpi/4.0.3/rhel7_u5/pbs/gnu/include/ -L/apps/pkg/openmpi/4.0.3/rhel7_u5/pbs/gnu/lib/
# MPIFLAGS = -I/apps/pkg/openmpi/3.1.2/rhel7_u5/pbs/gnu/include/ -L/apps/pkg/openmpi/3.1.2/rhel7_u5/pbs/gnu/lib/
# MPICH FLAGS
MPIFLAGS = -I/apps/pkg/mpich/3.3.1/rhel7_u5/pbs/gnu_mxm/include/ -L/apps/pkg/mpich/3.3.1/rhel7_u5/pbs/gnu_mxm/lib
CFLAGS = -lmpi

#GLOBAL_PARAMETERS
USE_DOUBLE = 0
USE_SVM_ALWAYS = 0
NUM_RUN = 10
TRANSACTION_BYTE = 128
#includes
# CUDA_INSTALL_PATH = /apps/pkg/cuda/9.2
# SAMPLE = samples
CUDA_INSTALL_PATH = /apps/pkg/cuda/10.2
SAMPLE = NVIDIA_CUDA-10.2_Samples
INCLUDES = -I$(CUDA_INSTALL_PATH)/include -I$(CUDA_INSTALL_PATH)/$(SAMPLE)/common/inc
#libs
CUDA_LIBS = -L$(CUDA_INSTALL_PATH)/lib -lcudart -lcusparse
LIBS = $(CUDA_LIBS)

all:
	$(CC) $(CXXFLAGS) $(CUFLAGS) -arch=sm_37 -O3 -Xcompiler -fopenmp main.cu -o spmv $(INCLUDES) $(LIBS) -D NUM_RUN=$(NUM_RUN) -D USE_DOUBLE=$(USE_DOUBLE) -D USE_SVM_ALWAYS=$(USE_SVM_ALWAYS) -D TRANSACTION_BYTE=$(TRANSACTION_BYTE)

MPI2dSpMV: mpimain2d.cu
	$(CC) $(CXXFLAGS) $(MPIFLAGS) $(CFLAGS) $(CUFLAGS) -arch=sm_37 -O3 -Xcompiler -fopenmp mpimain2d.cu -o mpi2dspmv $(INCLUDES) $(LIBS) -D NUM_RUN=$(NUM_RUN) -D USE_DOUBLE=$(USE_DOUBLE) -D USE_SVM_ALWAYS=$(USE_SVM_ALWAYS) -D TRANSACTION_BYTE=$(TRANSACTION_BYTE)

MPI2dSpMV_GPU_Model: SpMV2DModel.cu
	$(CC) $(CXXFLAGS) $(MPIFLAGS) $(CFLAGS) $(CUFLAGS) -arch=sm_37 -O3 -Xcompiler -fopenmp SpMV2DModel.cu -o SpMV2DModel $(INCLUDES) $(LIBS) -D NUM_RUN=$(NUM_RUN) -D USE_DOUBLE=$(USE_DOUBLE) -D USE_SVM_ALWAYS=$(USE_SVM_ALWAYS) -D TRANSACTION_BYTE=$(TRANSACTION_BYTE)

SpMVModel: SpMV_Model.cu
	$(CC) $(CXXFLAGS) $(CUFLAGS) -arch=sm_37 -O3 -Xcompiler -fopenmp SpMV_Model.cu -o SpMVModel $(INCLUDES) $(LIBS) -D NUM_RUN=$(NUM_RUN) -D USE_DOUBLE=$(USE_DOUBLE) -D USE_SVM_ALWAYS=$(USE_SVM_ALWAYS) -D TRANSACTION_BYTE=$(TRANSACTION_BYTE)
