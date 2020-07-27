#include "common.h"
#include "bhsparse_spmv_cuda.h"
#include "mmio.h"
#include <unordered_set>

long strideCounts = 0;
int testType = 0;
typedef int Idx;
int nnz_per_block, mat_row;
int save_mat=0;
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

    char outputFile[100] = "Results/CSR_CUDA_SpMV_Model.csv";
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
        fprintf(resultCSV, "M,N,AvgTime,TotalRun,NonZeroPerRow,NonZeroElements,Bandwidth,Flops,value_type,Type,Strides,TransactionByte,WordSize\n");
    }

    fprintf(resultCSV, "%d,%d,%10.6lf,%d,%lf,%d,%lf,%lf,%d,%s,%ld,%d,%d\n", m, n, cusparseTime, NUM_RUN, (double) nnz / m,
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




void call_cusp_ref(int m, int n, int nnz,  
                   int *csrRowPtrA, int *csrColIdxA, value_type *csrValA,
                   value_type *x, value_type *y, value_type *y_ref)
{
#if USE_SVM_ALWAYS
    cout << endl << "CUSP is using shared virtual memory (unified memory).";
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
    cout << endl << "CUSP is using dedicated GPU memory.";
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

// run CUSP START
    const int nnz_per_row = nnz / m;

    bhsparse_timer cusp_timer;
    cusp_timer.start();

    if (nnz_per_row <=  2)
    {
        for (int i = 0; i < NUM_RUN; i++)
#if USE_SVM_ALWAYS
            cusp_spmv<2>(m, n, nnz, svm_csrRowPtrA, svm_csrColIdxA, svm_csrValA, svm_x, svm_y);
#else
            cusp_spmv<2>(m, n, nnz, d_csrRowPtrA, d_csrColIdxA, d_csrValA, d_x, d_y);
#endif
    }
    else if (nnz_per_row <=  4)
    {
        for (int i = 0; i < NUM_RUN; i++)
#if USE_SVM_ALWAYS
            cusp_spmv<4>(m, n, nnz, svm_csrRowPtrA, svm_csrColIdxA, svm_csrValA, svm_x, svm_y);
#else
            cusp_spmv<4>(m, n, nnz, d_csrRowPtrA, d_csrColIdxA, d_csrValA, d_x, d_y);
#endif
    }
    else if (nnz_per_row <=  8)
    {
        for (int i = 0; i < NUM_RUN; i++)
#if USE_SVM_ALWAYS
            cusp_spmv<8>(m, n, nnz, svm_csrRowPtrA, svm_csrColIdxA, svm_csrValA, svm_x, svm_y);
#else
            cusp_spmv<8>(m, n, nnz, d_csrRowPtrA, d_csrColIdxA, d_csrValA, d_x, d_y);
#endif
    }
    else if (nnz_per_row <= 16)
    {
        for (int i = 0; i < NUM_RUN; i++)
#if USE_SVM_ALWAYS
            cusp_spmv<16>(m, n, nnz, svm_csrRowPtrA, svm_csrColIdxA, svm_csrValA, svm_x, svm_y);
#else
            cusp_spmv<16>(m, n, nnz, d_csrRowPtrA, d_csrColIdxA, d_csrValA, d_x, d_y);
#endif
    }
    else
    {
        for (int i = 0; i < NUM_RUN; i++)
#if USE_SVM_ALWAYS
            cusp_spmv<32>(m, n, nnz, svm_csrRowPtrA, svm_csrColIdxA, svm_csrValA, svm_x, svm_y);
#else
            cusp_spmv<32>(m, n, nnz, d_csrRowPtrA, d_csrColIdxA, d_csrValA, d_x, d_y);
#endif
    }

    checkCudaErrors(cudaDeviceSynchronize());
    double cuspTime = cusp_timer.stop() / NUM_RUN;

    cout << endl << "Checking CUSP SpMV Correctness ... ";

#if USE_SVM_ALWAYS == 0
    value_type *y_cusp_ref = (value_type *)malloc(m * sizeof(value_type));
    checkCudaErrors(cudaMemcpy(y_cusp_ref, d_y, m * sizeof(value_type), cudaMemcpyDeviceToHost));
#endif

    int error_count = 0;
    for (int i = 0; i < m; i++)
#if USE_SVM_ALWAYS
        if (y_ref[i] != svm_y[i])
            error_count++;
#else
        if (y_ref[i] != y_cusp_ref[i])
            error_count++;
#endif
    if (error_count)
        cout << "NO PASS. Error count = " << error_count << " out of " << m << " entries.";
    else
        cout << "PASS!";
    cout << endl;

    cout << "CUSP time = " << cuspTime
         << " ms. Bandwidth = " << gb/(1.0e+6 * cuspTime)
         << " GB/s. GFlops = " << gflop/(1.0e+6 * cuspTime)  << " GFlops." << endl << endl;
// run CUSP STOP

    char outputFile[100] = "Results/CSR_CUDA_SpMV_Model.csv";
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
        fprintf(resultCSV, "M,N,AvgTime,TotalRun,NonZeroPerRow,NonZeroElements,Bandwidth,Flops,value_type,Type,Strides,TransactionByte,WordSize\n");
    }

    fprintf(resultCSV, "%d,%d,%10.6lf,%d,%lf,%d,%lf,%lf,%d,%s,%ld,%d,%d\n", m, n, cuspTime, NUM_RUN, (double) nnz / m,
            nnz, gb / (1.0e+6 * cuspTime), gflop / (1.0e+6 * cuspTime), sizeof(value_type), "CUSP", strideCounts,
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
    free(y_cusp_ref);
    checkCudaErrors(cudaFree(d_csrRowPtrA));
    checkCudaErrors(cudaFree(d_csrColIdxA));
    checkCudaErrors(cudaFree(d_csrValA));
    checkCudaErrors(cudaFree(d_x));
    checkCudaErrors(cudaFree(d_y));
#endif

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
    cout << endl << "OpenMP is using dedicated HOST memory.";
    value_type *y_omp_ref = (value_type *)malloc(m * sizeof(value_type));
#endif

    double gb = (double)((m + 1 + nnz) * sizeof(int) + (2 * nnz + m) * sizeof(value_type));
    double gflop = (double)(2 * nnz);

// run OpenMP START
    omp_set_num_threads(4);
    cout << endl << "OpenMP is using 4 threads.";
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
        cout << "NO PASS. Error count = " << error_count << " out of " << m << " entries.";
    else
        cout << "PASS!";
    cout << endl;

    cout << "OpenMP time = " << ompTime
         << " ms. Bandwidth = " << gb/(1.0e+6 * ompTime)
         << " GB/s. GFlops = " << gflop/(1.0e+6 * ompTime)  << " GFlops." << endl << endl;
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

Idx create_random_diagonal_matrix(Idx **row_ptr, Idx **col_ptr, value_type **val_ptr, Idx m, Idx nnz_per_row,
        Idx save_mat) {
    std::ofstream createMat;
    std::string folderName = "SavedMatrix/";
    std::string outMat = "Random_Diag_Mat_" + std::to_string(m) + "_" + std::to_string(nnz_per_row) + ".mtx";
    if (mkdir(folderName.c_str(), 0777) == -1)
        std::cout << "Directory " << folderName << " is already exist" << std::endl;
    else
        std::cout << "Directory " << folderName << " created" << std::endl;
    std::string fileName = folderName + outMat;
    std::ifstream infile(fileName);
    if (infile.good()) {
        GraphReader reader = GraphReader();
        reader.readMatrix(fileName, &m, &nnz_per_row, row_ptr, col_ptr, val_ptr);
        infile.close();
        return _SUCCESS;
    }

    (*row_ptr) = (Idx *) calloc(m + 1, sizeof(Idx));
    (*col_ptr) = (Idx *) malloc(m * nnz_per_row * sizeof(Idx));
    (*val_ptr) = (value_type *) malloc(m * nnz_per_row * sizeof(value_type));
    srand(time(NULL) * (rank + 1));

    Idx *trackIndex, idx = 0;
    trackIndex = (Idx *) malloc(m * sizeof(Idx));
    for (int l = 0; l < m; ++l) {
        trackIndex[l] = -1;
    }
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < nnz_per_row; ++j) {
            Idx randColIdx;
            do {
                randColIdx = rand() % m;
            } while (trackIndex[randColIdx] >= i);
            trackIndex[randColIdx] = i;
            (*row_ptr)[i]++;
            (*col_ptr)[idx] = startCol + randColIdx;
            (*val_ptr)[idx] = ((value_type) (randColIdx % 10) + 1);
            idx++;
        }
    }

    Idx old = 0;
    for (int i = 0; i < m + 1; ++i) {
        Idx current = (*row_ptr)[i] + old;
        (*row_ptr)[i] = old;
        old = current;
    }
    if (save_mat == 1) {
        std::ofstream createMat;
        createMat.open(fileName, std::ios_base::out);
        createMat << "%%MatrixMarket matrix coordinate real general" << std::endl;
        createMat << m << " " << m << " " << (nnz_per_row * m) << std::endl;
        for (int i = 0; i < m; ++i) {
            for (int j = (*row_ptr)[i]; j < (*row_ptr)[i + 1]; ++j) {
                createMat << i << " " << (*col_ptr)[j] << " " << (*val_ptr)[j] << std::endl;
            }
        }
        createMat.close();
    }
    free(trackIndex);
    return _SUCCESS;
}


int call_bhsparse()
{
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

    cout << "PRECISION = " << precision << endl;
    cout << "RUN SpMV " << NUM_RUN << " times" << endl;

    int m, n, nnzA, max_deg = 0;
    int *csrRowPtrA;
    int *csrColIdxA;
    value_type *csrValA;
    m = n = mat_row;
    nnzA = nnz_per_block;

    create_random_diagonal_matrix(&csrRowPtrA, &csrColIdxA, &csrValA, m, nnzA/m, save_mat);
    double gb = (double)((m + 1 + nnzA) * sizeof(int) + (2 * nnzA + m) * sizeof(value_type));
    double gflop = (double)(2 * nnzA);

    cout << " ( " << m << ", " << n << " ) nnz = " << nnzA << endl;

    /*srand(time(NULL));
    for (int i = 0; i < nnzA; i++)
    {
        csrValA[i] = rand() % 10;
    }*/

    value_type *x = (value_type *)malloc(n * sizeof(value_type));
    for (int i = 0; i < n; i++)
        x[i] = rand() % 10;

    value_type *y = (value_type *)malloc(m * sizeof(value_type));
    value_type *y_ref = (value_type *)malloc(m * sizeof(value_type));

    /***********Access Pattern Based on 128 Threads Per Block *********/
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
    cout << "cpu sequential time = " << ref_time
         << " ms. Bandwidth = " << gb/(1.0e+6 * ref_time)
         << " GB/s. GFlops = " << gflop/(1.0e+6 * ref_time)  << " GFlops." << endl << endl;

    memset(y, 0, m * sizeof(value_type));


    bhsparse_spmv_cuda *bhsparse = new bhsparse_spmv_cuda();
    err = bhsparse->init_platform();


    // test OpenMP, cuSPARSE and CUSP v0.4.0
    call_cusp_ref(m, n, nnzA, csrRowPtrA, csrColIdxA, csrValA, x, y, y_ref);
    call_cusparse_ref(m, n, nnzA, csrRowPtrA, csrColIdxA, csrValA, x, y, y_ref);
//    call_omp_ref(m, n, nnzA, csrRowPtrA, csrColIdxA, csrValA, x, y, y_ref);

    // run bhSPARSE
    err = bhsparse->prepare_mem(m, n, nnzA, csrRowPtrA, csrColIdxA, csrValA, x, y);

    double time = 0.0;
    err = bhsparse->run_benchmark();
    bhsparse->get_y();

    // compare ref and our results
    cout << endl << "Checking bhSPARSE SpMV Correctness ... ";
    int error_count = 0;
    for (int i = 0; i < m; i++)
        if (y_ref[i] != y[i])
        {
            error_count++;
//            cout << "ROW [ " << i << " ] "
//                 << csrRowPtrA[i] << " - "
//                 << csrRowPtrA[i+1]
//                 << " warp = " <<  csrRowPtrA[i+1]/(31*256)
//                 << "\t cpu = " << y_ref[i]
//                 << ", \t gpu = " << y[i]
//                 << ", \t error = " << y_ref[i] - y[i]
//                 << endl;
        }

    if (error_count)
        cout << "NO PASS. Error count = " << error_count << " out of " << m << " entries.";
    else
    {
        cout << "PASS!";

        bhsparse_timer spmv_timer;
        spmv_timer.start();        
        for (int i = 0; i < NUM_RUN; i++)
        {
            err = bhsparse->run_benchmark();
        }
        time = spmv_timer.stop()/(double)NUM_RUN;

        cout << endl << "bhSPARSE time = " << time
             << " ms. Bandwidth = " << gb/(1.0e+6 * time)
             << " GB/s. GFlops = " << gflop/(1.0e+6 * time) << " GFlops." << endl;
    }

    err = bhsparse->free_platform();
    err = bhsparse->free_mem();

    free(csrRowPtrA);
    free(csrColIdxA);
    free(csrValA);
    free(x);
    free(y);
    free(y_ref);

    return err;
}

int main(int argc, char ** argv)
{
    int argi = 1;
    mat_row = atoi(argv[argi++]);
    nnz_per_block = atoi(argv[argi++]);
    if (argc > argi)
        save_mat = atoi(argv[argi++]);
    int err = 0;
    cout << "----------------mat row: "<< mat_row <<  " non-zero per block: " << nnz_per_block << " --------------" << endl;
    err = call_bhsparse(filename);
    cout << "------------------------------------------------------" << endl;

    return err;
}

