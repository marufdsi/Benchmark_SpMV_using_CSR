#include "common.h"
#include "bhsparse_spmv_cuda.h"
#include "mmio.h"
#include <unordered_set>
#include<mpi.h>

//#include "mpi-ext.h" /* Needed for CUDA-aware check */

#define MAX_STRING_LENGTH 128
long strideCounts = 0;
int mpi_rank, nRanks, MASTER = 0, sqrRank, row_rank, col_rank, firstRow, firstCol, total_sparsity = 0,
max_sparsity = 0, transactionByte = 128;
int mat_row = 0, nonZeroElements = 0, nodes = 0, ppn = 0, save_mat = 0, _format = 0;
MPI_Comm commrow;
MPI_Comm commcol;

cusparseStatus_t cusparse_spmv(cusparseHandle_t handle, cusparseMatDescr_t descr, 
                   int m, int n, int nnz, 
                   int *csrRowPtrA, int *csrColIdxA, double *csrValA, 
                   double *x, double *y, double alpha, double beta)
{
    return cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, m, n, nnz, &alpha, descr, csrValA, csrRowPtrA, csrColIdxA, x, &beta, y);
}

cusparseStatus_t cusparse_spmv(cusparseHandle_t handle, cusparseMatDescr_t descr,
                   int m, int n, int nnz,
                   int *csrRowPtrA, int *csrColIdxA, float *csrValA,
                   float *x, float *y, float alpha, float beta)
{
    return cusparseScsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, m, n, nnz, &alpha, descr, csrValA, csrRowPtrA, csrColIdxA, x, &beta, y);
}

template <unsigned int THREADS_PER_VECTOR>
void cusp_spmv(int m, int n, int nnz, int *svm_csrRowPtrA, int *svm_csrColIdxA, double *svm_csrValA, double *svm_x, double *svm_y)
{
    const size_t THREADS_PER_BLOCK  = 128;
    const size_t VECTORS_PER_BLOCK  = THREADS_PER_BLOCK / THREADS_PER_VECTOR;

    const size_t NUM_BLOCKS = ceil((double)m / (double)VECTORS_PER_BLOCK);

    spmv_csr_vector_kernel
            <int, double, VECTORS_PER_BLOCK, THREADS_PER_VECTOR>
            <<<NUM_BLOCKS, THREADS_PER_BLOCK>>>
            (m, svm_csrRowPtrA, svm_csrColIdxA, svm_csrValA, svm_x, svm_y);
}

template <unsigned int THREADS_PER_VECTOR>
void cusp_spmv(int m, int n, int nnz, int *svm_csrRowPtrA, int *svm_csrColIdxA, float *svm_csrValA, float *svm_x, float *svm_y)
{
    const size_t THREADS_PER_BLOCK  = 128;
    const size_t VECTORS_PER_BLOCK  = THREADS_PER_BLOCK / THREADS_PER_VECTOR;

    const size_t NUM_BLOCKS = ceil((double)m / (double)VECTORS_PER_BLOCK);

    spmv_csr_vector_kernel
            <int, float, VECTORS_PER_BLOCK, THREADS_PER_VECTOR>
            <<<NUM_BLOCKS, THREADS_PER_BLOCK>>>
            (m, svm_csrRowPtrA, svm_csrColIdxA, svm_csrValA, svm_x, svm_y);
}

void call_cusparse_ref(int m, int n, int nnz, 
                       int *csrRowPtrA, int *csrColIdxA, value_type *csrValA, 
                       value_type *x, value_type *y, value_type *y_ref)
{
    // prepare shared virtual memory (unified memory)
#if USE_SVM_ALWAYS
    cout << endl << "cuSPARSE is using shared virtual memory (unified memory).";
    int *svm_csrRowPtrA;
    int *svm_csrColIdxA;
    value_type *svm_csrValA;
    value_type *svm_x;
    value_type *svm_y;

    checkCudaErrors(cudaMallocManaged(&svm_csrRowPtrA, (m+1) * sizeof(int)));
    checkCudaErrors(cudaMallocManaged(&svm_csrColIdxA, nnz  * sizeof(int)));
    checkCudaErrors(cudaMallocManaged(&svm_csrValA,    nnz  * sizeof(value_type)));
    memcpy(svm_csrRowPtrA, csrRowPtrA, (m+1) * sizeof(int));
    memcpy(svm_csrColIdxA, csrColIdxA, nnz  * sizeof(int));
    memcpy(svm_csrValA,    csrValA,    nnz  * sizeof(value_type));

    checkCudaErrors(cudaMallocManaged(&svm_x, n  * sizeof(value_type)));
    memcpy(svm_x,    x,    n  * sizeof(value_type));
    checkCudaErrors(cudaMallocManaged(&svm_y, m  * sizeof(value_type)));
    memcpy(svm_y,    y,    m  * sizeof(value_type));
    // prepare device memory
#else
    cout << endl << "cuSPARSE is using dedicated GPU memory.";
    int *d_csrRowPtrA;
    int *d_csrColIdxA;
    value_type *d_csrValA;
    value_type *d_x;
    value_type *d_y;

    checkCudaErrors(cudaMalloc((void **)&d_csrRowPtrA, (m+1) * sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&d_csrColIdxA, nnz  * sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&d_csrValA,    nnz  * sizeof(value_type)));
    checkCudaErrors(cudaMemcpy(d_csrRowPtrA, csrRowPtrA, (m+1) * sizeof(int),   cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_csrColIdxA, csrColIdxA, nnz  * sizeof(int),   cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_csrValA,    csrValA,    nnz  * sizeof(value_type),   cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc((void **)&d_x, n * sizeof(value_type)));
    checkCudaErrors(cudaMemcpy(d_x, x, n * sizeof(value_type), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc((void **)&d_y, m  * sizeof(value_type)));
    checkCudaErrors(cudaMemcpy(d_y, y, m * sizeof(value_type), cudaMemcpyHostToDevice));
#endif

    double gb = (double)((m + 1 + nnz) * sizeof(int) + (2 * nnz + m) * sizeof(value_type));
    double gflop = (double)(2 * nnz);

// run cuSPARSE START
    cusparseHandle_t handle = 0;
    cusparseStatus_t status;
    status = cusparseCreate(&handle);
    if (status != CUSPARSE_STATUS_SUCCESS)
    {
        printf("CUSPARSE initialization error\n");
        //return -1;
    }

    cusparseMatDescr_t descr = 0;
    status = cusparseCreateMatDescr(&descr);
    if (status != CUSPARSE_STATUS_SUCCESS)
    {
        printf("CUSPARSE cusparseCreateMatDescr error\n");
        //return -2;
    }
    cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);

    value_type alpha = 1.0;
    value_type beta = 0.0;

    checkCudaErrors(cudaDeviceSynchronize());
    bhsparse_timer cusparse_timer;
    cusparse_timer.start();
    for (int i = 0; i < NUM_RUN; i++)
    {
#if USE_SVM_ALWAYS
        status = cusparse_spmv(handle, descr, m, n, nnz, svm_csrRowPtrA, svm_csrColIdxA, svm_csrValA, svm_x, svm_y, alpha, beta);
#else
        status = cusparse_spmv(handle, descr, m, n, nnz, d_csrRowPtrA, d_csrColIdxA, d_csrValA, d_x, d_y, alpha, beta);
#endif
    }
    checkCudaErrors(cudaDeviceSynchronize());
    if (status != CUSPARSE_STATUS_SUCCESS)
    {
        printf("CUSPARSE cusparseS/Dcsrmv error\n");
        //return -3;
    }
    double cusparseTime = cusparse_timer.stop() / NUM_RUN;

    cout << endl << "Checking cuSPARSE SpMV Correctness ... ";

#if USE_SVM_ALWAYS == 0
    value_type *y_cusparse_ref = (value_type *)malloc(m * sizeof(value_type));
    checkCudaErrors(cudaMemcpy(y_cusparse_ref, d_y, m * sizeof(value_type), cudaMemcpyDeviceToHost));
#endif

    int error_count = 0;
    for (int i = 0; i < m; i++)
#if USE_SVM_ALWAYS
        if (y_ref[i] != svm_y[i])
            error_count++;
#else
        if (y_ref[i] != y_cusparse_ref[i])
            error_count++;
#endif
    if (error_count)
        cout << "NO PASS. Error count = " << error_count << " out of " << m << " entries.";
    else
        cout << "PASS!";
    cout << endl;

    cout << "cuSPARSE time = " << cusparseTime
         << " ms. Bandwidth = " << gb/(1.0e+6 * cusparseTime)
         << " GB/s. GFlops = " << gflop/(1.0e+6 * cusparseTime)  << " GFlops." << endl << endl;
// run cuSPARSE STOP

    char outputFile[100] = "Results/CSR_MPI_CUDA_2D_SpMV_Model.csv";
    FILE *resultCSV;
    FILE *checkFile;
    if ((checkFile = fopen(outputFile, "r")) != NULL) {
        // file exists
        fclose(checkFile);
        if (!(resultCSV = fopen(outputFile, "a"))) {
            fprintf(stderr, "fopen: failed to open %s file\n", outputFile);
            exit(EXIT_FAILURE);
        }
    } else {
        if (!(resultCSV = fopen(outputFile, "w"))) {
            fprintf(stderr, "fopen: failed to open file %s\n", outputFile);
            exit(EXIT_FAILURE);
        }
        fprintf(resultCSV, "M,N,AvgTime,TotalRun,NonZeroPerRow,NonZeroElements,Bandwidth,Flops,ValueType,Type,Strides,TransactionByte,WordSize\n");
    }

    fprintf(resultCSV, "%s,%d,%d,%10.6lf,%d,%lf,%d,%lf,%lf,%d,%s,%ld,%d,%d\n", m, n, cusparseTime, NUM_RUN, (double) nnz / m,
            nnz, gb / (1.0e+6 * cusparseTime), gflop / (1.0e+6 * cusparseTime), sizeof(value_type), "CUSPARSE", strideCounts,
            TRANSACTION_BYTE, TRANSACTION_BYTE/ sizeof(value_type));
    if (fclose(resultCSV) != 0) {
        fprintf(stderr, "fopen: failed to open file %s\n", outputFile);
        exit(EXIT_FAILURE);
    }

#if USE_SVM_ALWAYS
    checkCudaErrors(cudaFree(svm_csrValA));
    checkCudaErrors(cudaFree(svm_csrRowPtrA));
    checkCudaErrors(cudaFree(svm_csrColIdxA));
    checkCudaErrors(cudaFree(svm_x));
    checkCudaErrors(cudaFree(svm_y));
#else
    free(y_cusparse_ref);
    checkCudaErrors(cudaFree(d_csrRowPtrA));
    checkCudaErrors(cudaFree(d_csrColIdxA));
    checkCudaErrors(cudaFree(d_csrValA));
    checkCudaErrors(cudaFree(d_x));
    checkCudaErrors(cudaFree(d_y));
#endif

    return;
}




void call_cusp_ref(int m, int n, int nnz, int *csrRowPtrA, int *csrColIdxA, value_type *csrValA, value_type *x
        , value_type *y, value_type *y_ref)
{
    int *d_csrRowPtrA;
    int *d_csrColIdxA;
    value_type *d_csrValA;
    value_type *d_x;
    value_type *d_y;

    checkCudaErrors(cudaMalloc((void **)&d_csrRowPtrA, (m+1) * sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&d_csrColIdxA, nnz  * sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&d_csrValA,    nnz  * sizeof(value_type)));
    checkCudaErrors(cudaMemcpy(d_csrRowPtrA, csrRowPtrA, (m+1) * sizeof(int),   cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_csrColIdxA, csrColIdxA, nnz  * sizeof(int),   cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_csrValA,    csrValA,    nnz  * sizeof(value_type),   cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc((void **)&d_x, n * sizeof(value_type)));
    checkCudaErrors(cudaMalloc((void **)&d_y, m  * sizeof(value_type)));
    checkCudaErrors(cudaMemcpy(d_y, y, m * sizeof(value_type), cudaMemcpyHostToDevice));

//        cout << endl << "[" << mpi_rank << "] Checking CUSP SpMV Correctness ... ";
    checkCudaErrors(cudaMemcpy(d_x, x, n * sizeof(value_type), cudaMemcpyHostToDevice));
    cusp_spmv<32>(m, n, nnz, d_csrRowPtrA, d_csrColIdxA, d_csrValA, d_x, d_y);
    value_type *y_cusp_ref = (value_type *)malloc(m * sizeof(value_type));
    checkCudaErrors(cudaMemcpy(y_cusp_ref, d_y, m * sizeof(value_type), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaDeviceSynchronize());

    int error_count = 0;
    for (int i = 0; i < m; i++)
        if (abs(y_ref[i] - y_cusp_ref[i]) > 0.01 * abs(y_ref[i])/*y_ref[i] != y_cusp_ref[i]*/)
            error_count++;
    if (error_count)
        cout << "[" << mpi_rank << "] NO PASS. Error count = " << error_count << " out of " << m << " entries." << endl;
//    else
//        cout << "PASS!" << endl;

    double gb = (double)((m + 1 + nnz) * sizeof(int) + (2 * nnz + m) * sizeof(value_type));
    double gflop = (double)(2 * nnz);

// run CUSP START
    const int nnz_per_row = nnz / m;
    bhsparse_timer cusp_timer;
    bhsparse_timer broadcast_timer;
    bhsparse_timer mult_timer;
    bhsparse_timer reduce_timer;
    cusp_timer.start();
    double b_time, r_time, m_time, avg_b_time = 0, avg_r_time = 0, avg_m_time = 0;
    if (nnz_per_row <=  2)
    {
//        cout<< "THREADS_PER_VECTOR = 2" << endl;
        MPI_Barrier(MPI_COMM_WORLD);
        for (int i = 0; i < NUM_RUN+SKIP; i++) {
//            cout << "[" << mpi_rank << "] 2-iter= " << i+1 << " mat= " << matName << endl;
            broadcast_timer.start();
            MPI_Bcast(x, m, MPI_FLOAT, col_rank, commcol); //col_rank is the one with the correct information
            b_time = broadcast_timer.stop();
            checkCudaErrors(cudaMemcpy(d_x, x, n * sizeof(value_type), cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(d_y, y, m * sizeof(value_type), cudaMemcpyHostToDevice));
            mult_timer.start();
            cusp_spmv<2>(m, n, nnz, d_csrRowPtrA, d_csrColIdxA, d_csrValA, d_x, d_y);
            m_time = mult_timer.stop();
            checkCudaErrors(cudaMemcpy(y, d_y, m * sizeof(value_type), cudaMemcpyDeviceToHost));
            reduce_timer.start();
            MPI_Reduce(y, x, m, MPI_FLOAT, MPI_SUM, row_rank, commrow);
            r_time = reduce_timer.stop();
            if(i>=SKIP){
                avg_b_time += b_time;
                avg_m_time += m_time;
                avg_r_time += r_time;
            }
            MPI_Barrier(commcol);
            MPI_Barrier(commrow);
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }
    else if (nnz_per_row <=  4)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        for (int i = 0; i < NUM_RUN+SKIP; i++) {
//            cout << "[" << mpi_rank << "] 4-iter= " << i+1 << " mat= " << matName << endl;
            broadcast_timer.start();
            MPI_Bcast(x, m, MPI_FLOAT, col_rank, commcol); //col_rank is the one with the correct information
            b_time = broadcast_timer.stop();
            checkCudaErrors(cudaMemcpy(d_x, x, n * sizeof(value_type), cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(d_y, y, m * sizeof(value_type), cudaMemcpyHostToDevice));
            mult_timer.start();
            cusp_spmv<4>(m, n, nnz, d_csrRowPtrA, d_csrColIdxA, d_csrValA, d_x, d_y);
            m_time = mult_timer.stop();
            checkCudaErrors(cudaMemcpy(y, d_y, m * sizeof(value_type), cudaMemcpyDeviceToHost));
            reduce_timer.start();
            MPI_Reduce(y, x, m, MPI_FLOAT, MPI_SUM, row_rank, commrow);
            r_time = reduce_timer.stop();
            if(i>=SKIP){
                avg_b_time += b_time;
                avg_m_time += m_time;
                avg_r_time += r_time;
            }
            MPI_Barrier(commcol);
            MPI_Barrier(commrow);
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }
    else if (nnz_per_row <=  8)
    {
//        cout<< "THREADS_PER_VECTOR = 8" << endl;
        MPI_Barrier(MPI_COMM_WORLD);
        for (int i = 0; i < NUM_RUN+SKIP; i++) {
//            cout << "[" << mpi_rank << "] 8-iter= " << i+1 << " mat= " << matName << endl;
            broadcast_timer.start();
            MPI_Bcast(x, m, MPI_FLOAT, col_rank, commcol); //col_rank is the one with the correct information
            b_time = broadcast_timer.stop();
            checkCudaErrors(cudaMemcpy(d_x, x, n * sizeof(value_type), cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(d_y, y, m * sizeof(value_type), cudaMemcpyHostToDevice));
            mult_timer.start();
            cusp_spmv<8>(m, n, nnz, d_csrRowPtrA, d_csrColIdxA, d_csrValA, d_x, d_y);
            m_time = mult_timer.stop();
            checkCudaErrors(cudaMemcpy(y, d_y, m * sizeof(value_type), cudaMemcpyDeviceToHost));
            reduce_timer.start();
            MPI_Reduce(y, x, m, MPI_FLOAT, MPI_SUM, row_rank, commrow);
            r_time = reduce_timer.stop();
            if(i>=SKIP){
                avg_b_time += b_time;
                avg_m_time += m_time;
                avg_r_time += r_time;
            }
            MPI_Barrier(commcol);
            MPI_Barrier(commrow);
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }
    else if (nnz_per_row <= 16)
    {
//        cout<< "[" << mpi_rank << "] THREADS_PER_VECTOR = 16" << endl;
        MPI_Barrier(MPI_COMM_WORLD);
        for (int i = 0; i < NUM_RUN; i++) {
            /*if(i==19){
                for (int j = 0; j < 10; ++j) {
                    cout<< "[" << mpi_rank << "] 16: " ;
                    cout << x[j] << " ";
                }
                cout<<endl;
            }
            cout << "[" << mpi_rank << "] 16-iter= " << i+1 << " mat= " << matName << endl;*/
            broadcast_timer.start();
            MPI_Bcast(x, m, MPI_FLOAT, col_rank, commcol); //col_rank is the one with the correct information
            b_time = broadcast_timer.stop();
            checkCudaErrors(cudaMemcpy(d_x, x, n * sizeof(value_type), cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(d_y, y, m * sizeof(value_type), cudaMemcpyHostToDevice));
            mult_timer.start();
            cusp_spmv<16>(m, n, nnz, d_csrRowPtrA, d_csrColIdxA, d_csrValA, d_x, d_y);
            m_time = mult_timer.stop();
            checkCudaErrors(cudaMemcpy(y, d_y, m * sizeof(value_type), cudaMemcpyDeviceToHost));
            reduce_timer.start();
            MPI_Reduce(y, x, m, MPI_FLOAT, MPI_SUM, row_rank, commrow);
            r_time = reduce_timer.stop();
            if(i>=SKIP){
                avg_b_time += b_time;
                avg_m_time += m_time;
                avg_r_time += r_time;
            }
            MPI_Barrier(commcol);
            MPI_Barrier(commrow);
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }
    else
    {
//        cout<< "THREADS_PER_VECTOR = 32" << endl;
        MPI_Barrier(MPI_COMM_WORLD);
        for (int i = 0; i < NUM_RUN+SKIP; i++) {
            /*if(i==20){
                for (int j = 0; j < 10; ++j) {
                    cout<< "[" << mpi_rank << "] 32: ";
                    cout << x[j] << " ";
                }
                cout<<endl;
            }
            cout << "[" << mpi_rank << "] 32-iter= " << i+1 << " mat= " << matName << endl;*/
            broadcast_timer.start();
            MPI_Bcast(x, m, MPI_FLOAT, col_rank, commcol); //col_rank is the one with the correct information
            b_time = broadcast_timer.stop();
            checkCudaErrors(cudaMemcpy(d_x, x, n * sizeof(value_type), cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(d_y, y, m * sizeof(value_type), cudaMemcpyHostToDevice));
            mult_timer.start();
            cusp_spmv<32>(m, n, nnz, d_csrRowPtrA, d_csrColIdxA, d_csrValA, d_x, d_y);
            m_time = mult_timer.stop();
            checkCudaErrors(cudaMemcpy(y, d_y, m * sizeof(value_type), cudaMemcpyDeviceToHost));
            reduce_timer.start();
            MPI_Reduce(y, x, m, MPI_FLOAT, MPI_SUM, row_rank, commrow);
            r_time = reduce_timer.stop();
            if(i>=SKIP){
                avg_b_time += b_time;
                avg_m_time += m_time;
                avg_r_time += r_time;
            }
            MPI_Barrier(commcol);
            MPI_Barrier(commrow);
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    checkCudaErrors(cudaDeviceSynchronize());
    if(mpi_rank == MASTER)
        cout<< "Run complete" << endl;
    double cuspTime = cusp_timer.stop() / (NUM_RUN+SKIP);
    int avg_nnz;
    double avg_nnz_per_row, avgTime;
    avg_b_time /= NUM_RUN;
    avg_m_time /= NUM_RUN;
    avg_r_time /= NUM_RUN;
    MPI_Reduce(&nnz, &avg_nnz, 1, MPI_INT, MPI_SUM, MASTER, MPI_COMM_WORLD);
    MPI_Reduce(&nnz_per_row, &avg_nnz_per_row, 1, MPI_DOUBLE, MPI_SUM, MASTER, MPI_COMM_WORLD);
    MPI_Reduce(&avg_b_time, &b_time, 1, MPI_DOUBLE, MPI_SUM, MASTER, MPI_COMM_WORLD);
    MPI_Reduce(&avg_m_time, &m_time, 1, MPI_DOUBLE, MPI_SUM, MASTER, MPI_COMM_WORLD);
    MPI_Reduce(&avg_r_time, &r_time, 1, MPI_DOUBLE, MPI_SUM, MASTER, MPI_COMM_WORLD);
    MPI_Reduce(&cuspTime, &avgTime, 1, MPI_DOUBLE, MPI_SUM, MASTER, MPI_COMM_WORLD);
    avg_nnz /= nRanks;
    avg_nnz_per_row /= nRanks;
    b_time /= nRanks;
    m_time /= nRanks;
    r_time /= nRanks;
    avgTime /= nRanks;
    if(mpi_rank == MASTER) {
        cout << "CUSP time = " << cuspTime
             << " ms. Bandwidth = " << gb / (1.0e+6 * cuspTime)
             << " GB/s. GFlops = " << gflop / (1.0e+6 * cuspTime) << " GFlops." << endl << endl;
// run CUSP STOP

        char outputFile[100] = "Results/CSR_MPI_CUDA_2D_SpMV_Model.csv";
        FILE *resultCSV;
        FILE *checkFile;
        if ((checkFile = fopen(outputFile, "r")) != NULL) {
            // file exists
            fclose(checkFile);
            if (!(resultCSV = fopen(outputFile, "a"))) {
                fprintf(stderr, "fopen: failed to open %s file\n", outputFile);
                exit(EXIT_FAILURE);
            }
        } else {
            if (!(resultCSV = fopen(outputFile, "w"))) {
                fprintf(stderr, "fopen: failed to open file %s\n", outputFile);
                exit(EXIT_FAILURE);
            }
            fprintf(resultCSV,
                    "M,N,AvgTime,AvgBcastTime,AvgMultTime,AvgReduceTime,TotalRun,NonZeroPerRow,NonZeroElements,Bandwidth,Flops,ValueType,Type,Strides,TransactionByte,WordSize\n");
        }

        fprintf(resultCSV, "%s,%d,%d,%10.6lf,%10.6lf,%10.6lf,%10.6lf,%d,%lf,%d,%lf,%lf,%d,%s,%ld,%d,%d\n", m,
                n, avgTime, b_time, m_time, r_time, (NUM_RUN + SKIP), avg_nnz_per_row, avg_nnz, gb/(1.0e+6 * avgTime),
                gflop/(1.0e+6 * avgTime), sizeof(value_type), "CUSP", strideCounts, TRANSACTION_BYTE,
                TRANSACTION_BYTE/sizeof(value_type));
        if (fclose(resultCSV) != 0) {
            fprintf(stderr, "fopen: failed to open file %s\n", outputFile);
            exit(EXIT_FAILURE);
        }
    }
    checkCudaErrors(cudaDeviceSynchronize());
    free(y_cusp_ref);
    checkCudaErrors(cudaFree(d_csrRowPtrA));
    checkCudaErrors(cudaFree(d_csrColIdxA));
    checkCudaErrors(cudaFree(d_csrValA));
    checkCudaErrors(cudaFree(d_x));
    checkCudaErrors(cudaFree(d_y));

    return;
}

void call_omp_ref(int m, int n, int nnz,
                  int *csrRowPtrA, int *csrColIdxA, value_type *csrValA,
                  value_type *x, value_type *y, value_type *y_ref)
{
#if USE_SVM_ALWAYS
    cout << endl << "OpenMP is using shared virtual memory (unified memory).";
    int *svm_csrRowPtrA;
    int *svm_csrColIdxA;
    value_type *svm_csrValA;
    value_type *svm_x;
    value_type *svm_y;

    // prepare shared virtual memory (unified memory)
    checkCudaErrors(cudaMallocManaged(&svm_csrRowPtrA, (m+1) * sizeof(int)));
    checkCudaErrors(cudaMallocManaged(&svm_csrColIdxA, nnz  * sizeof(int)));
    checkCudaErrors(cudaMallocManaged(&svm_csrValA,    nnz  * sizeof(value_type)));
    memcpy(svm_csrRowPtrA, csrRowPtrA, (m+1) * sizeof(int));
    memcpy(svm_csrColIdxA, csrColIdxA, nnz  * sizeof(int));
    memcpy(svm_csrValA,    csrValA,    nnz  * sizeof(value_type));

    checkCudaErrors(cudaMallocManaged(&svm_x, n  * sizeof(value_type)));
    memcpy(svm_x,    x,    n  * sizeof(value_type));
    checkCudaErrors(cudaMallocManaged(&svm_y, m  * sizeof(value_type)));
    memcpy(svm_y,    y,    m  * sizeof(value_type));
#else
//    cout << endl << "OpenMP is using dedicated HOST memory.";
    value_type *y_omp_ref = (value_type *)malloc(m * sizeof(value_type));
#endif

    double gb = (double)((m + 1 + nnz) * sizeof(int) + (2 * nnz + m) * sizeof(value_type));
    double gflop = (double)(2 * nnz);

// run OpenMP START
//    omp_set_num_threads(4);
//    cout << endl << "OpenMP is using 4 threads.";
    checkCudaErrors(cudaDeviceSynchronize());

    bhsparse_timer omp_timer;
    omp_timer.start();

    for (int iter = 0; iter < NUM_RUN; iter++)
    {
        #pragma omp parallel for
        for (int i = 0; i < m; i++)
        {
            value_type sum = 0;
#if USE_SVM_ALWAYS
            for (int j = svm_csrRowPtrA[i]; j < svm_csrRowPtrA[i+1]; j++)
                sum += svm_x[svm_csrColIdxA[j]] * svm_csrValA[j];
            svm_y[i] = sum;
#else
            for (int j = csrRowPtrA[i]; j < csrRowPtrA[i+1]; j++)
                sum += x[csrColIdxA[j]] * csrValA[j];
            y_omp_ref[i] = sum;
#endif
        }
    }

    double ompTime = omp_timer.stop() / NUM_RUN;

    cout << endl << "Checking OpenMP SpMV Correctness ... ";
    int error_count = 0;
    for (int i = 0; i < m; i++)
#if USE_SVM_ALWAYS
        if (y_ref[i] != svm_y[i])
            error_count++;
#else
        if (y_ref[i] != y_omp_ref[i])
            error_count++;
#endif
    if (error_count)
        cout << "NO PASS. Error count = " << error_count << " out of " << m << " entries." << endl;
//    else
//        cout << "PASS!" << endl;

if(mpi_rank == MASTER) {
    cout << "OpenMP time = " << ompTime
         << " ms. Bandwidth = " << gb / (1.0e+6 * ompTime)
         << " GB/s. GFlops = " << gflop / (1.0e+6 * ompTime) << " GFlops." << endl << endl;
}
// run OpenMP STOP

#if USE_SVM_ALWAYS
    checkCudaErrors(cudaFree(svm_csrValA));
    checkCudaErrors(cudaFree(svm_csrRowPtrA));
    checkCudaErrors(cudaFree(svm_csrColIdxA));
    checkCudaErrors(cudaFree(svm_x));
    checkCudaErrors(cudaFree(svm_y));
#else
    free(y_omp_ref);
#endif

    return;
}

int call_bhsparse_small()
{
    int err = 0;

    int m, n, nnzA;

    int *csrColIdxA;
    int *csrRowPtrA;
    value_type *csrValA;

    m = 6;
    n = 6;
    nnzA = 15;

    csrColIdxA = (int *)malloc(nnzA * sizeof(int));
    csrRowPtrA = (int *)malloc((m+1) * sizeof(int));
    csrValA = (value_type *)malloc(nnzA * sizeof(value_type));

    int row_ptr[7]     = {0,       3,                9,    11, 11, 12,      15};
    int col_idx[15]    = {0, 2, 5, 0, 1, 2, 3, 4, 5, 2, 4,      4,  2, 3, 4};
    value_type val[15] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

    memcpy(csrRowPtrA, row_ptr, (m+1) * sizeof(int));
    memcpy(csrColIdxA, col_idx, nnzA * sizeof(int));
    memcpy(csrValA, val, nnzA * sizeof(value_type));

    cout << "row_ptr = [ ";
    for (int i = 0; i < m + 1; i++)
        cout << csrRowPtrA[i] << ", ";
    cout << " ]" << endl;

    cout << "col_idx = [ ";
    for (int i = 0; i < nnzA; i++)
        cout << csrColIdxA[i] << ", ";
    cout << " ]" << endl;

    cout << "value   = [ ";
    for (int i = 0; i < nnzA; i++)
        cout << csrValA[i] << ", ";
    cout << " ]" << endl << endl;

    value_type *x = (value_type *)malloc(n * sizeof(value_type));
    for (int i = 0; i < n; i++)
        x[i] = 1.0;

    value_type *y = (value_type *)malloc(m * sizeof(value_type));
    value_type *y_ref = (value_type *)malloc(m * sizeof(value_type));

    // compute cpu results
    for (int i = 0; i < m; i++)
    {
        value_type sum = 0;
        for (int j = csrRowPtrA[i]; j < csrRowPtrA[i+1]; j++)
            sum += x[csrColIdxA[j]] * csrValA[j];
        y_ref[i] = sum;
    }

    memset(y, 0, m * sizeof(value_type));

    bhsparse_spmv_cuda *bhsparse = new bhsparse_spmv_cuda();
    err = bhsparse->init_platform();
//    cout << "Initializing CUDA platform ... ";
    if (!err) {
//        cout << "Done.";
    }
    else
        cout << "\"Initializing CUDA platform ... Failed. Error code = " << err << endl;
//    cout << endl;

    err = bhsparse->prepare_mem(m, n, nnzA, csrRowPtrA, csrColIdxA, csrValA, x, y);

    err = bhsparse->run_benchmark();

    cout << endl;
    // print y_ref
    cout << "(CPU) y = ";
    for (int i = 0; i < m; i++)
    {
        cout << y_ref[i] << ", ";
        if ((i+1) % 16 == 0)
            cout << endl;
    }
    cout << endl;

    // print y
    cout << "(GPU) y = ";
    for (int i = 0; i < m; i++)
    {
        cout << y[i] << ", ";
        if ((i+1) % 16 == 0)
            cout << endl;
    }
    cout << endl;

    // compare cpu and gpu results
    cout << endl << "Checking bhSPARSE SpMV Correctness ... ";
    int error_count = 0;
    for (int i = 0; i < m; i++)
        if (y_ref[i] != y[i])
        {
            error_count++;
            cout << "ERROR ROW [ " << i << " ] " "cpu = " << y_ref[i] << ", gpu = " << y[i] << endl;
        }
    if (error_count)
        cout << "NO PASS. Error count = " << error_count << " out of " << m << " entries.";
    else
        cout << "PASS!";
    cout << endl;

    free(y_ref);

    err = bhsparse->free_platform();
    err = bhsparse->free_mem();

    return err;
}

int create_random_diagonal_matrix(int **row_ptr, int **col_ptr, value_type **val_ptr, int m,
                                  int nnz_per_row, int startCol, int rank, int isCSR) {
    size_t alignment = 64;

    if (isCSR == 1) {
        posix_memalign((void **) row_ptr, alignment, (m + 1) * sizeof(int));
        memset((*row_ptr), 0, (m+1)* sizeof(int));
    }
    else
        posix_memalign((void **) row_ptr, alignment, m * nnz_per_row * sizeof(int));
    posix_memalign((void **) col_ptr, alignment, m * nnz_per_row * sizeof(int));
    posix_memalign((void **) val_ptr, alignment, m * nnz_per_row * sizeof(value_type));
    srand(time(NULL) * (rank + 1));

    int *trackIndex, idx = 0;
    trackIndex = (int *) malloc(m * sizeof(int));
    for (int l = 0; l < m; ++l) {
        trackIndex[l] = -1;
    }
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < nnz_per_row; ++j) {
            int randColIdx;
            do {
                randColIdx = rand() % m;
            } while (trackIndex[randColIdx] >= i);
            trackIndex[randColIdx] = i;
            if (isCSR == 1)
                (*row_ptr)[i]++;
            else
                (*row_ptr)[idx] = i;
            (*col_ptr)[idx] = randColIdx;
            (*val_ptr)[idx] = ((value_type) (randColIdx % 10) + 1);
            idx++;
        }
    }

    if (isCSR == 1) {
        int old = 0;
        for (int i = 0; i < m + 1; ++i) {
            int current = (*row_ptr)[i] + old;
            (*row_ptr)[i] = old;
            old = current;
        }
    }

    free(trackIndex);
    return 0;
}


int call_CSR_bhsparse(){
    int err = 0;

    // report precision of floating-point
    char  *precision;
    if (sizeof(value_type) == 4)
    {
        precision = "32-bit Single Precision";
    }
    else if (sizeof(value_type) == 8)
    {
        precision = "64-bit Double Precision";
    }
    else
    {
        cout << "Wrong precision. Program exit!" << endl;
        return 0;
    }

    int m, n, nnzA, max_deg;
    int *csrRowPtrA;
    int *csrColIdxA;
    value_type *csrValA;
    m = n = mat_row;
    nnzA = nonZeroElements;
    max_deg = nnzA/m;
    create_random_diagonal_matrix(&csrRowPtrA, &csrColIdxA, &csrValA, m, nnzA/m, save_mat, col_rank * mat_row, mpi_rank, 1);
    double gb = (double)((m + 1 + nnzA) * sizeof(int) + (2 * nnzA + m) * sizeof(value_type));
    double gflop = (double)(2 * nnzA);

    srand(time(NULL));
    for (int i = 0; i < nnzA; i++)
    {
        csrValA[i] = 1.0/(value_type)m;
    }

    value_type *x = (value_type *)malloc(m * sizeof(value_type));
    for (int i = 0; i < m; i++)
        x[i] = 1.0;

    value_type *y = (value_type *)malloc(m * sizeof(value_type));
    value_type *y_ref = (value_type *)malloc(m * sizeof(value_type));

    /***********Access Pattern Based on 128 Threads Per Block *********/
    if(mpi_rank == MASTER)
        cout << "M: " << m << " N: " << n << " nnzA: " << nnzA << " Max degree=" << max_deg << endl;
    int wordSize = TRANSACTION_BYTE/ sizeof(value_type);
    for (int row_i = 0; row_i < m; row_i += wordSize) {
        for (int k = 0; k < max_deg; ++k) {
            int failed = 0;
            int row_check = (row_i + wordSize) > m ? m : (row_i + wordSize);
            unordered_set<long> hashme;
            for (int th = row_i; th < row_check; ++th) {
                if (k < (csrRowPtrA[th + 1] - csrRowPtrA[th])) {
                    hashme.insert((long)(&x[csrColIdxA[csrRowPtrA[th] + k]])/TRANSACTION_BYTE);
                    failed = 1;
                }

            }

            if (failed == 0) {
                break;
            }
            strideCounts += hashme.size();
        }
    }
    if(mpi_rank == MASTER)
        cout << "Strides count: " << strideCounts << " Transaction Byte Size: " << TRANSACTION_BYTE << " Number of Transaction Word: " << wordSize << endl;

    /*****************************************************************/

    // compute cpu results
    bhsparse_timer ref_timer;
    ref_timer.start();

    int ref_iter = 1;
    for (int iter = 0; iter < ref_iter; iter++)
    {
        for (int i = 0; i < m; i++)
        {
            value_type sum = 0;
            for (int j = csrRowPtrA[i]; j < csrRowPtrA[i+1]; j++)
                sum += x[csrColIdxA[j]] * csrValA[j];
            y_ref[i] = sum;
        }
    }

    double ref_time = ref_timer.stop() / (double)ref_iter;
    if(mpi_rank == MASTER) {
        cout << "cpu sequential time = " << ref_time
             << " ms. Bandwidth = " << gb / (1.0e+6 * ref_time)
             << " GB/s. GFlops = " << gflop / (1.0e+6 * ref_time) << " GFlops." << endl << endl;
    }

    memset(y, 0, m * sizeof(value_type));


    bhsparse_spmv_cuda *bhsparse = new bhsparse_spmv_cuda();
    err = bhsparse->init_platform(mpi_rank);


    call_cusp_ref(m, m, nnzA, csrRowPtrA, csrColIdxA, csrValA, x, y, y_ref);
//    call_cusparse_ref(m, m, nnzA, csrRowPtrA, csrColIdxA, csrValA, x, y, y_ref);
//    call_omp_ref(m, n, nnzA, csrRowPtrA, csrColIdxA, csrValA, x, y, y_ref);


    free(csrRowPtrA);
    free(csrColIdxA);
    free(csrValA);
    free(x);
    free(y);
    free(y_ref);

    return err;
}

int call_COO_bhsparse(){
    int err = 0;

    // report precision of floating-point
    char  *precision;
    if (sizeof(value_type) == 4)
    {
        precision = "32-bit Single Precision";
    }
    else if (sizeof(value_type) == 8)
    {
        precision = "64-bit Double Precision";
    }
    else
    {
        cout << "Wrong precision. Program exit!" << endl;
        return 0;
    }
    return 0;
}

int main(int argc, char ** argv)
{
    int mat_row = 0, nnzA = 0, nodes = 0, ppn = 0, save_mat = 0, _format = 0;
    int argi = 1;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    mat_row = atoi(argv[argi++]);
    nonZeroElements = atoi(argv[argi++]);
    nonZeroElements = (nonZeroElements/mat_row) * mat_row;
    if (argc > argi)
        _format = atoi(argv[argi++]);
    if (argc > argi)
        nodes = atoi(argv[argi++]);
    if (argc > argi)
        ppn = atoi(argv[argi++]);
    if (argc > argi)
        save_mat = atoi(argv[argi++]);


    sqrRank = sqrt(nRanks);
    row_rank = mpi_rank / sqrRank; //which col of proc am I
    col_rank = mpi_rank % sqrRank; //which row of proc am I

    //initialize communicators
    MPI_Comm_split(MPI_COMM_WORLD, row_rank, mpi_rank, &commrow);
    MPI_Comm_split(MPI_COMM_WORLD, col_rank, mpi_rank, &commcol);

    int err = 0;
    if (mpi_rank == MASTER)
        std::cout<<"M: " << mat_row << " nnzA: " << nnzA << " format: " << (_format == 0 ? "CSR" : "COO") << std::endl;

    if (mat_row == 0 || nnzA == 0)
        err = call_bhsparse_small();
    else if(_format == 0)
        err = call_CSR_bhsparse();
    else
        err = call_COO_bhsparse();
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return err;
}

