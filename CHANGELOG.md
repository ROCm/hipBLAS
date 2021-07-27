# Change Log for hipBLAS

## (Unreleased) hipBLAS 0.48.0
### Added
- Added more support for hipblas-bench

### Fixed
- Avoid large offset overflow for gemv and hemv in hipblas-test

### Changed
- Packaging split into a runtime package called hipblas and a development package called hipblas-devel. The development package depends on runtime. The runtime package suggests the development package for all supported OSes except CentOS 7 to aid in the transition. The suggests feature in packaging is introduced as a deprecated feature and will be removed in a future rocm release.

## [hipBLAS 0.47.0 for ROCm 4.4.0]
## Added
- Added HIPBLAS_STATUS_UNKNOWN for unsupported backend status codes

## [hipBLAS 0.46.0 for ROCm 4.3.0]
### Added
- Added hipblasStatusToString

### Fixed
- Added catch() blocks around API calls to prevent the leak of C++ exceptions

## [hipBLAS 0.44.0 for ROCm 4.2.0]
### Added
- Made necessary changes to work with rocBLAS' gemm_ex changes. When using rocBLAS backend, hipBLAS will query the preferable
  layout of int8 data to be passed to gemm_ex, and will pass in the resulting flag. Users must be sure to use the preferable
  data format when calling gemm_ex with a rocBLAS backend.
- Added hipblas-bench with support for:
    - copy, swap, scal

## [hipBLAS 0.42.0 for ROCm 4.1.0]
### Added
- Added the following functions. All added functions include batched and strided-batched support with rocBLAS backend:
    - axpy_ex
    - dot_ex
    - nrm2_ex
    - rot_ex
    - scal_ex

### Fixed
- Fixed complex unit test bug caused by incorrect caxpy and zaxpy function signatures

## [hipBLAS 0.40.0 for ROCm 4.0.0]
### Added
- Added changelog
- Added hipblas-bench with support for:
    - gemv, trsm, gemm
- Added rocSOLVER as a cpack dependency

## [hipBLAS 0.38.0 for ROCm 3.10.0]
### Added
- Added hipblasSetAtomicsMode and hipblasGetAtomicsMode
- No longer look for CUDA backend unless --cuda build flag is passed

## [hipBLAS 0.36.0 for ROCm 3.9.0]
### Added
- Make device memory reallocate on demand

## [hipBLAS 0.34.0 for ROCm 3.8.0]
### Added
- Added --static build flag to allow for creating a static library

## [hipBLAS 0.32.0 for ROCm 3.7.0]
### Added
- Added --rocblas-path command line option to choose path to pre-built rocBLAS
- Added sgetriBatched, dgetriBatched, cgetriBatched, and zgetriBatched
- Added TrsmEx, TrsmBatchedEx, and TrsmStridedBatchedEx
- Added hipblasSetVectorAsync and hipblasGetVectorAsync
- Added hipblasSetMatrixAsync and hipblasGetMatrixAsync
- Added Fortran support for getrf, getrs, geqrf and all variants thereof

## [hipBLAS 0.30.0 for ROCm 3.6.0]
### Added
- Added the following functions. All added functions include batched and strided-batched support with rocBLAS backend:
    - stbsv, dtbsv, ctbsv, ztbsv
    - ssymm, dsymm, csymm, zsymm
    - cgeam, zgeam
    - chemm, zhemm
    - strtri, dtrtri, ctrtri, ztrtri
    - sdgmm, ddgmm, cdgmm, zdgmm
- Added GemmBatchedEx and GemmStridedBatchedEx
- Added Fortran support for BLAS functions

## [hipBLAS 0.28.0 for ROCm 3.5.0]
### Added
- Added the following functions. All added functions include batched and strided-batched support with rocBLAS backend:
    - sgbmv, dgbmv, cgbmv, zgbmv
    - chemv, zhemv
    - stbmv, dtbmv, ctbmv, ztbmv
    - strmv, trmv, ctrmv, ztrmv
    - chbmv, zhbmv
    - cher, zher
    - cher2, zher2
    - chpmv, zhpmv
    - chpr, zhpr
    - chpr2, zhpr2
    - ssbmv, dsbmv
    - sspmv, dspmv
    - ssymv, dsymv, csymv, zsymv
    - stpmv, dtpmv, ctpmv, ztpmv
    - cgeru, cgerc, zgeru, zgerc
    - sspr, dspr, cspr, zspr
    - sspr2, dspr2
    - csyr, zsyr
    - ssyr2, dsyr2, csyr2, zsyr2
    - stpsv, dtpsv, ctpsv, ztpsv
    - ctrsv, ztrsv
    - cherk, zherk
    - cherkx, zherkx
    - cher2k, zher2k
    - ssyrk, dsyrk, csyrk, zsyrk
    - ssyr2k, dsyr2k, csyr2k, zsyr2k
    - ssyrkx, dsyrkx, csyrkx, zsyrkx
    - ctrmm, ztrmm
    - ctrsm, ztrsm
