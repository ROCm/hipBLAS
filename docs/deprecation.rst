#######################
Deprecations by version
#######################

Announced in hipBLAS 0.49
*************************

Replace inplace hipblasXtrmm with out of place hipblasXtrmm
===========================================================

The hipblasXtrmm API, along with batched versions, will be changing in hipBLAS 1.0
release to allow in-place and out-of-place behavior. This change will introduce an
output matrix 'C', matching the rocblas_xtrmm_outofplace API and the cublasXtrmm API.

Announced in hipBLAS 0.53
*************************

Remove packed_int8x4 datatype
=============================

The packed_int8x4 datatype will be removed in hipBLAS 1.0. There are two int8 datatypes:

* int8_t
* packed_int8x4

int8_t is the C99 unsigned 8 bit integer. packed_int8x4 has 4 consecutive int8_t numbers
in the k dimension packed into 32 bits. packed_int8x4 is only used in hipblasGemmEx.
int8_t will continue to be available in hipblasGemmEx.

Announced in hipBLAS 1.0
^^^^^^^^^^^^^^^^^^^^^^^^

Replace Legacy BLAS in-place trmm functions with trmm functions that support both in-place and out-of-place functionality
=========================================================================================================================
Use of the deprecated Legacy BLAS in-place trmm functions will give deprecation warnings telling
you to compile with -DHIPBLAS_V1 and use the new in-place and out-of-place trmm functions.

Note that there are no deprecation warnings for the hipBLAS Fortran API.

The Legacy BLAS in-place trmm calculates B <- alpha * op(A) * B. Matrix B is replaced in-place by
triangular matrix A multiplied by matrix B. The prototype in the include file rocblas-functions.h is:

::

    hipblasStatus_t hipblasStrmm(hipblasHandle_t    handle,
                                 hipblasSideMode_t  side,
                                 hipblasFillMode_t  uplo,
                                 hipblasOperation_t transA,
                                 hipblasDiagType_t  diag,
                                 int                m,
                                 int                n,
                                 const float*       alpha,
                                 const float*       AP,
                                 int                lda,
                                 float*             BP,
                                 int                ldb);

hipBLAS 1.0 deprecates the legacy BLAS trmm functionality and replaces it with C <- alpha * op(A) * B. The prototype is:

::

    hipblasStatus_t hipblasStrmmOutofplace(hipblasHandle_t    handle,
                                           hipblasSideMode_t  side,
                                           hipblasFillMode_t  uplo,
                                           hipblasOperation_t transA,
                                           hipblasDiagType_t  diag,
                                           int                m,
                                           int                n,
                                           const float*       alpha,
                                           const float*       AP,
                                           int                lda,
                                           const float*       BP,
                                           int                ldb,
                                           float*             CP,
                                           int                ldc);

The new API provides the legacy BLAS in-place functionality if you set pointer C equal to pointer B and
ldc equal to ldb.

There are similar deprecations for the _batched and _strided_batched versions of trmm.

Removed in hipBLAS 1.0
^^^^^^^^^^^^^^^^^^^^^^

HIPBLAS_INT8_DATATYPE_PACK_INT8x4 hipblasGemmEx support removed
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

Packed int8x4 was removed as support for arbitrary dimensioned int8_t data is a superset of this functionality:

* enum hipblasInt8Datatype_t was removed
* function hipblasSetInt8Datatype was removed
* function hipblasGetInt8Datatype was removed

Announced in hipBLAS 2.0
^^^^^^^^^^^^^^^^^^^^^^^^

Remove hipblasDatatype_t and replace with hipDataType
=====================================================
Use of the deprecated hipblasDatatype_t will give deprecation warnings telling you to compile with -DHIPBLAS_V2
and to use hipDataType instead.

Note that there are no deprecation warnings for the hipBLAS Fortran API.

hipblasDatatype_t will be removed in a future release, and the use of this type in the API will be replaced with hipDataType.

Removed in hipBLAS 2.0
^^^^^^^^^^^^^^^^^^^^^^

Legacy BLAS in-place trmm
'''''''''''''''''''''''''
The legacay BLAS in-place hipblasXtrmm that calculates B <- alpha * op(A) * B is removed and replaced with the
out-of-place hipblasXtrmm that calculates C <- alpha * op(A) * B.

The prototype for the removed legacy BLAS in-place functionality was

::

    hipblasStatus_t hipblasStrmm(hipblasHandle_t    handle,
                                 hipblasSideMode_t  side,
                                 hipblasFillMode_t  uplo,
                                 hipblasOperation_t transA,
                                 hipblasDiagType_t  diag,
                                 int                m,
                                 int                n,
                                 const float*       alpha,
                                 const float*       A,
                                 int                lda,
                                 float*             B,
                                 int                ldb);

The prototype for the replacement in-place and out-of-place functionality is

::

    hipblasStatus_t hipblasStrmmOutofplace(hipblasHandle_t    handle,
                                           hipblasSideMode_t  side,
                                           hipblasFillMode_t  uplo,
                                           hipblasOperation_t transA,
                                           hipblasDiagType_t  diag,
                                           int                m,
                                           int                n,
                                           const float*       alpha,
                                           const float*       A,
                                           int                lda,
                                           const float*       B,
                                           int                ldb,
                                           float*             C,
                                           int                ldc);

The legacy BLAS in-place functionality can be obtained with the new function if you set pointer C equal to pointer B and
ldc equal to ldb.

The out-of-place functionality is from setting pointer B distinct from pointer C.
