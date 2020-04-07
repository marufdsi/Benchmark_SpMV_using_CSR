#compilers
CC=nvcc
CXX = g++
#GLOBAL_PARAMETERS
USE_DOUBLE = 0
USE_SVM_ALWAYS = 0
NUM_RUN = 100
TRANSACTION_BYTE = 128
#includes
CUDA_INSTALL_PATH = /apps/pkg/cuda/9.2
INCLUDES = -I$(CUDA_INSTALL_PATH)/include -I$(CUDA_INSTALL_PATH)/samples/common/inc

#libs
CUDA_LIBS = -L$(CUDA_INSTALL_PATH)/lib -lcudart -lcusparse
LIBS = $(CUDA_LIBS)

all:
	$(CC)  -arch=sm_32 -O3 -Xcompiler -fopenmp main.cu -o spmv $(INCLUDES) $(LIBS) -D USE_DOUBLE=$(USE_DOUBLE) -D USE_SVM_ALWAYS=$(USE_SVM_ALWAYS)
