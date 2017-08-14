/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#pragma once
#ifndef _TESTING_UTILITY_H_
#define _TESTING_UTILITY_H_

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include "hipblas.h"
#include <sys/time.h>

using namespace std;

/*!\file
 * \brief provide data initialization, timing, hipblas type <-> lapack char conversion utilities.
 */

#define CHECK_HIP_ERROR(error) \
    if (error != hipSuccess) { \
      fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(error), error,__FILE__, __LINE__); \
      exit(EXIT_FAILURE);\
    }

#define BLAS_1_RESULT_PRINT                                                      \
    if(argus.timing){                                                            \
        cout << "N, hipblas (us), ";                                             \
        if(argus.norm_check){                                                    \
            cout << "CPU (us), error" ;                                          \
        }                                                                        \
        cout << endl;                                                            \
        cout << N <<',' << gpu_time_used << ',';                                 \
        if(argus.norm_check){                                                    \
            cout << cpu_time_used <<',';                                         \
            cout << hipblas_error;                                               \
        }                                                                        \
        cout << endl;                                                            \
    }

    /* ============================================================================================ */
    /* generate random number :*/

    /*! \brief  generate a random number between [0, 0.999...] . */
    template<typename T>
    T random_generator(){
        //return rand()/( (T)RAND_MAX + 1);
        return (T)(rand() % 10 + 1); //generate a integer number between [1, 10]
    };


    /* ============================================================================================ */
    /*! \brief  matrix/vector initialization: */
    // for vector x (M=1, N=lengthX, lda=incx);
    // for complex number, the real/imag part would be initialized with the same value
    template<typename T>
    void hipblas_init(vector<T> &A, int M, int N, int lda){
        for (int i = 0; i < M; ++i){
            for (int j = 0; j < N; ++j){
                A[i+j*lda] = random_generator<T>();
            }
        }
    };
    template<typename T>
    void hipblas_init(T* A, int M, int N, int lda){
        for (int i = 0; i < M; ++i){
            for (int j = 0; j < N; ++j){
                A[i+j*lda] = random_generator<T>();
            }
        }
    };

    /*! \brief  symmetric matrix initialization: */
    // for real matrix only
    template<typename T>
    void hipblas_init_symmetric(vector<T> &A, int N, int lda){
        for (int i = 0; i < N; ++i){
            for (int j = 0; j <= i; ++j){
                A[j+i*lda] = A[i+j*lda] = random_generator<T>();
            }
        }
    };

    /*! \brief  hermitian matrix initialization: */
    // for complex matrix only, the real/imag part would be initialized with the same value
    // except the diagonal elment must be real
    template<typename T>
    void hipblas_init_hermitian(vector<T> &A, int N, int lda){
        for (int i = 0; i < N; ++i){
            for (int j = 0; j <= i; ++j){
                A[j+i*lda] = A[i+j*lda] = random_generator<T>();
                if(i==j) A[j+i*lda].y = 0.0;
            }
        }
    };

    /* ============================================================================================ */
    /*! \brief  turn float -> 's', double -> 'd', hipblas_float_complex -> 'c', hipblas_double_complex -> 'z' */
    template<typename T>
    char type2char();

    /* ============================================================================================ */
    /*! \brief  Debugging purpose, print out CPU and GPU result matrix, not valid in complex number  */
    template<typename T>
    void print_matrix(vector<T> CPU_result, vector<T> GPU_result, int m, int n, int lda){  
        for(int i=0;i<m;i++)
            for(int j=0;j<n;j++)
            {
                printf("matrix  col %d, row %d, CPU result=%f, GPU result=%f\n", i, j, CPU_result[j+i*lda], GPU_result[j+i*lda]);
            }
    }

#ifdef __cplusplus
extern "C" {
#endif

    /* ============================================================================================ */
    /*  device query and print out their ID and name */
    int query_device_property();

    /*  set current device to device_id */
    void set_device(int device_id);

    /* ============================================================================================ */
    /*  timing: HIP only provides very limited timers function clock() and not general;
                hipblas sync CPU and device and use more accurate CPU timer*/

    /*! \brief  CPU Timer(in microsecond): synchronize with the default device and return wall time */
    double get_time_us( void );


    /*! \brief  CPU Timer(in microsecond): synchronize with given queue/stream and return wall time */
    double get_time_us_sync( hipStream_t stream );

    /* ============================================================================================ */
    /*  Convert hipblas constants to lapack char. */

    char
    hipblas2char_operation(hipblasOperation_t value);

    char
    hipblas2char_fill(hipblasFillMode_t value);

    char
    hipblas2char_diagonal(hipblasDiagType_t value);

    char
    hipblas2char_side(hipblasSideMode_t value);

    /* ============================================================================================ */
    /*  Convert lapack char constants to hipblas type. */

    hipblasOperation_t
    char2hipblas_operation(char value);

    hipblasFillMode_t
    char2hipblas_fill(char value);

    hipblasDiagType_t
    char2hipblas_diagonal(char value);

    hipblasSideMode_t
    char2hipblas_side(char value);

#ifdef __cplusplus
}
#endif


/* ============================================================================================ */


/*! \brief Class used to parse command arguments in both client & gtest   */

// has to compile with option "-std=c++11", and this hipblas library uses c++11 everywhere
// c++11 allows intilization of member of a struct

class Arguments {
    public:
    int M = 128;
    int N = 128;
    int K = 128;

    int rows = 128;
    int cols = 128;

    int lda = 128;
    int ldb = 128;
    int ldc = 128;

    int incx = 1 ;
    int incy = 1 ;
    int incd = 1 ;

    int start = 1024;
    int end   = 10240;
    int step  = 1000;

    double alpha = 1.0;
    double beta  = 0.0;

    char transA_option = 'N';
    char transB_option = 'N';
    char side_option = 'L';
    char uplo_option = 'L';
    char diag_option = 'N';

    int apiCallCount = 1;
    int batch_count = 10;

    int norm_check = 0;
    int unit_check = 1;
    int timing = 0;

    Arguments & operator=(const Arguments &rhs)
    {
        M = rhs.M;
        N = rhs.N;
        K = rhs.K;

        lda = rhs.lda;
        ldb = rhs.ldb;
        ldc = rhs.ldc;

        incx = rhs.incx;
        incy = rhs.incy;
        incd = rhs.incd;

        start = rhs.start;
        end = rhs.end;
        step = rhs.step;

        alpha = rhs.alpha;
        beta = rhs.beta;

        transA_option = rhs.transA_option;
        transB_option = rhs.transB_option;
        side_option = rhs.side_option;
        uplo_option = rhs.uplo_option;
        diag_option = rhs.diag_option;

        apiCallCount = rhs.apiCallCount;
        batch_count = rhs.batch_count;

        norm_check = rhs.norm_check;
        unit_check = rhs.unit_check;
        timing = rhs.timing;

        return *this;
    }

};



#endif
