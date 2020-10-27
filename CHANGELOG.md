# Change Log for hipBLAS

## [(Unreleased) hipBLAS 0.40.0 for ROCm 4.0.0]
### Added
__anchor__a

__anchor__t
- Added hipblas-bench

__anchor__w

__anchor__d
- Added changelog

__anchor__n

__anchor__l

__ahchor__o


## [(Unreleased) hipBLAS 0.38.0 for ROCm 3.10.0]
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
