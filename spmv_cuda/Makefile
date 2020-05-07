#compilers
CC=nvcc
CXX=g++
MPICXX=mpicxx

CFLAGS = -lmpi
CXXFLAGS = -std=c++11
CUFLAGS="-D__LP64__ -ta:tesla"

MPIFLAGS = -I/apps/pkg/openmpi/4.0.3/rhel7_u5/pbs/gnu/include/ -L/apps/pkg/openmpi/4.0.3/rhel7_u5/pbs/gnu/lib/
# MPIFLAGS = -I/apps/pkg/openmpi/3.1.2/rhel7_u5/pbs/gnu/include/ -L/apps/pkg/openmpi/3.1.2/rhel7_u5/pbs/gnu/lib/
# MPICH FLAGS
#MPIFLAGS = -I/apps/pkg/mpich-3.2.1/rhel7_u2-x86_64-ib/gnu/include/ -L/apps/pkg/mpich-3.2.1/rhel7_u2-x86_64-ib/gnu/lib/
CFLAGS = -lmpi

#GLOBAL_PARAMETERS
USE_DOUBLE = 0
USE_SVM_ALWAYS = 0
NUM_RUN = 10
TRANSACTION_BYTE = 128
#includes
# CUDA_INSTALL_PATH = /apps/pkg/cuda/9.2
CUDA_INSTALL_PATH = /apps/pkg/cuda/10.2
INCLUDES = -I$(CUDA_INSTALL_PATH)/include -I$(CUDA_INSTALL_PATH)/samples/common/inc

#libs
CUDA_LIBS = -L$(CUDA_INSTALL_PATH)/lib -lcudart -lcusparse
LIBS = $(CUDA_LIBS)

all:
	$(CC) $(CXXFLAGS) $(CUFLAGS) -arch=sm_37 -O3 -Xcompiler -fopenmp main.cu -o spmv $(INCLUDES) $(LIBS) -D NUM_RUN=$(NUM_RUN) -D USE_DOUBLE=$(USE_DOUBLE) -D USE_SVM_ALWAYS=$(USE_SVM_ALWAYS) -D TRANSACTION_BYTE=$(TRANSACTION_BYTE)

mpi2dspmv: mpimain2d.cu
	$(CC) $(CXXFLAGS) $(MPIFLAGS) $(CFLAGS) $(CUFLAGS) -arch=sm_37 -O3 -Xcompiler -fopenmp mpimain2d.cu -o mpi2dspmv $(INCLUDES) $(LIBS) -D NUM_RUN=$(NUM_RUN) -D USE_DOUBLE=$(USE_DOUBLE) -D USE_SVM_ALWAYS=$(USE_SVM_ALWAYS) -D TRANSACTION_BYTE=$(TRANSACTION_BYTE)
