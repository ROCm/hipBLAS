.. _api_label:


*************
Functions
*************

Naming conventions
==================

    hipBLAS follows the following naming conventions,
    - Big case for matrix, e.g. matrix A, B, C   GEMM (C = A*B)
    - Lower case for vector, e.g. vector x, y    GEMV (y = A*x)


Notations
=========

    hipBLAS functions uses the following notations to denote precisions,
    - h  = half
    - bf = 16 bit brian floating point
    - s  = single
    - d  = double
    - c  = single complex
    - z  = double complex

*************
hipBLAS Types
*************

Definitions
===========

hipblasHandle_t
--------------
.. doxygentypedef:: hipblasHandle_t

hipblasHalf
------------
.. doxygentypedef:: hipblasHalf

hipblasInt8
-----------
.. doxygentypedef:: hipblasInt8

hipblasStride
--------------
.. doxygentypedef:: hipblasStride

hipblasBfloat16
----------------
.. doxygenstruct:: hipblasBfloat16

hipblasComplex
---------------------
.. doxygenstruct:: hipblasComplex

hipblasDoubleComplex
-----------------------
.. doxygenstruct:: hipblasDoubleComplex

Enums
=====
Enumeration constants have numbering that is consistent with CBLAS, ACML and most standard C BLAS libraries.

hipblasStatus_t
-----------------
.. doxygenenum:: hipblasStatus_t

hipblasOperation_t
------------
.. doxygenenum:: hipblasOperation_t

hipblasPointerMode_t
----------------
.. doxygenenum:: hipblasPointerMode_t

hipblasFillMode_t
------------
.. doxygenenum:: hipblasFillMode_t

hipblasDiagType_t
--------------
.. doxygenenum:: hipblasDiagType_t

hipblasSideMode_t
----------------
.. doxygenenum:: hipblasSideMode_t

hipblasDatatype_t
--------------------
.. doxygenenum:: hipblasDatatype_t

hipblasGemmAlgo_t
--------------------
.. doxygenenum:: hipblasGemmAlgo_t

hipblasAtomicsMode_t
------------------
.. doxygenenum:: hipblasAtomicsMode_t

*****************
hipBLAS Functions
*****************

Level 1 BLAS
============

hipblasIXamax + Batched, StridedBatched
-----------------------------------------
.. doxygenfunction:: hipblasIsamax
.. doxygenfunction:: hipblasIdamax
.. doxygenfunction:: hipblasIcamax
.. doxygenfunction:: hipblasIzamax

.. doxygenfunction:: hipblasIsamaxBatched
.. doxygenfunction:: hipblasIdamaxBatched
.. doxygenfunction:: hipblasIcamaxBatched
.. doxygenfunction:: hipblasIzamaxBatched

.. doxygenfunction:: hipblasIsamaxStridedBatched
.. doxygenfunction:: hipblasIdamaxStridedBatched
.. doxygenfunction:: hipblasIcamaxStridedBatched
.. doxygenfunction:: hipblasIzamaxStridedBatched


hipblasIXamin + Batched, StridedBatched
-----------------------------------------
.. doxygenfunction:: hipblasIsamin
.. doxygenfunction:: hipblasIdamin
.. doxygenfunction:: hipblasIcamin
.. doxygenfunction:: hipblasIzamin

.. doxygenfunction:: hipblasIsaminBatched
.. doxygenfunction:: hipblasIdaminBatched
.. doxygenfunction:: hipblasIcaminBatched
.. doxygenfunction:: hipblasIzaminBatched

.. doxygenfunction:: hipblasIsaminStridedBatched
.. doxygenfunction:: hipblasIdaminStridedBatched
.. doxygenfunction:: hipblasIcaminStridedBatched
.. doxygenfunction:: hipblasIzaminStridedBatched

hipblasXasum + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasSasum
.. doxygenfunction:: hipblasDasum
.. doxygenfunction:: hipblasScasum
.. doxygenfunction:: hipblasDzasum

.. doxygenfunction:: hipblasSasumBatched
.. doxygenfunction:: hipblasDasumBatched
.. doxygenfunction:: hipblasScasumBatched
.. doxygenfunction:: hipblasDzasumBatched

.. doxygenfunction:: hipblasSasumStridedBatched
.. doxygenfunction:: hipblasDasumStridedBatched
.. doxygenfunction:: hipblasScasumStridedBatched
.. doxygenfunction:: hipblasDzasumStridedBatched

hipblasXaxpy + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasHaxpy
.. doxygenfunction:: hipblasSaxpy
.. doxygenfunction:: hipblasDaxpy
.. doxygenfunction:: hipblasCaxpy
.. doxygenfunction:: hipblasZaxpy

.. doxygenfunction:: hipblasHaxpyBatched
.. doxygenfunction:: hipblasSaxpyBatched
.. doxygenfunction:: hipblasDaxpyBatched
.. doxygenfunction:: hipblasCaxpyBatched
.. doxygenfunction:: hipblasZaxpyBatched

.. doxygenfunction:: hipblasHaxpyStridedBatched
.. doxygenfunction:: hipblasSaxpyStridedBatched
.. doxygenfunction:: hipblasDaxpyStridedBatched
.. doxygenfunction:: hipblasCaxpyStridedBatched
.. doxygenfunction:: hipblasZaxpyStridedBatched

hipblasXcopy + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasScopy
.. doxygenfunction:: hipblasDcopy
.. doxygenfunction:: hipblasCcopy
.. doxygenfunction:: hipblasZcopy

.. doxygenfunction:: hipblasScopyBatched
.. doxygenfunction:: hipblasDcopyBatched
.. doxygenfunction:: hipblasCcopyBatched
.. doxygenfunction:: hipblasZcopyBatched

.. doxygenfunction:: hipblasScopyStridedBatched
.. doxygenfunction:: hipblasDcopyStridedBatched
.. doxygenfunction:: hipblasCcopyStridedBatched
.. doxygenfunction:: hipblasZcopyStridedBatched

hipblasXdot + Batched, StridedBatched
---------------------------------------
.. doxygenfunction:: hipblasHdot
.. doxygenfunction:: hipblasBfdot
.. doxygenfunction:: hipblasSdot
.. doxygenfunction:: hipblasDdot
.. doxygenfunction:: hipblasCdotc
.. doxygenfunction:: hipblasCdotu
.. doxygenfunction:: hipblasZdotc
.. doxygenfunction:: hipblasZdotu

.. doxygenfunction:: hipblasHdotBatched
.. doxygenfunction:: hipblasBfdotBatched
.. doxygenfunction:: hipblasSdotBatched
.. doxygenfunction:: hipblasDdotBatched
.. doxygenfunction:: hipblasCdotcBatched
.. doxygenfunction:: hipblasCdotuBatched
.. doxygenfunction:: hipblasZdotcBatched
.. doxygenfunction:: hipblasZdotuBatched

.. doxygenfunction:: hipblasHdotStridedBatched
.. doxygenfunction:: hipblasBfdotStridedBatched
.. doxygenfunction:: hipblasSdotStridedBatched
.. doxygenfunction:: hipblasDdotStridedBatched
.. doxygenfunction:: hipblasCdotcStridedBatched
.. doxygenfunction:: hipblasCdotuStridedBatched
.. doxygenfunction:: hipblasZdotcStridedBatched
.. doxygenfunction:: hipblasZdotuStridedBatched

hipblasXnrm2 + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasSnrm2
.. doxygenfunction:: hipblasDnrm2
.. doxygenfunction:: hipblasScnrm2
.. doxygenfunction:: hipblasDznrm2

.. doxygenfunction:: hipblasSnrm2Batched
.. doxygenfunction:: hipblasDnrm2Batched
.. doxygenfunction:: hipblasScnrm2Batched
.. doxygenfunction:: hipblasDznrm2Batched

.. doxygenfunction:: hipblasSnrm2StridedBatched
.. doxygenfunction:: hipblasDnrm2StridedBatched
.. doxygenfunction:: hipblasScnrm2StridedBatched
.. doxygenfunction:: hipblasDznrm2StridedBatched

hipblasXrot + Batched, StridedBatched
---------------------------------------
.. doxygenfunction:: hipblasSrot
.. doxygenfunction:: hipblasDrot
.. doxygenfunction:: hipblasCrot
.. doxygenfunction:: hipblasCsrot
.. doxygenfunction:: hipblasZrot
.. doxygenfunction:: hipblasZdrot

.. doxygenfunction:: hipblasSrotBatched
.. doxygenfunction:: hipblasDrotBatched
.. doxygenfunction:: hipblasCrotBatched
.. doxygenfunction:: hipblasCsrotBatched
.. doxygenfunction:: hipblasZrotBatched
.. doxygenfunction:: hipblasZdrotBatched

.. doxygenfunction:: hipblasSrotStridedBatched
.. doxygenfunction:: hipblasDrotStridedBatched
.. doxygenfunction:: hipblasCsrotStridedBatched
.. doxygenfunction:: hipblasCsrotStridedBatched
.. doxygenfunction:: hipblasZrotStridedBatched
.. doxygenfunction:: hipblasZdrotStridedBatched

hipblasXrotg + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasSrotg
.. doxygenfunction:: hipblasDrotg
.. doxygenfunction:: hipblasCrotg
.. doxygenfunction:: hipblasZrotg

.. doxygenfunction:: hipblasSrotgBatched
.. doxygenfunction:: hipblasDrotgBatched
.. doxygenfunction:: hipblasCrotgBatched
.. doxygenfunction:: hipblasZrotgBatched

.. doxygenfunction:: hipblasSrotgStridedBatched
.. doxygenfunction:: hipblasDrotgStridedBatched
.. doxygenfunction:: hipblasCrotgStridedBatched
.. doxygenfunction:: hipblasZrotgStridedBatched

hipblasXrotm + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasSrotm
.. doxygenfunction:: hipblasDrotm

.. doxygenfunction:: hipblasSrotmBatched
.. doxygenfunction:: hipblasDrotmBatched

.. doxygenfunction:: hipblasSrotmStridedBatched
.. doxygenfunction:: hipblasDrotmStridedBatched

hipblasXrotmg + Batched, StridedBatched
-----------------------------------------
.. doxygenfunction:: hipblasSrotmg
.. doxygenfunction:: hipblasDrotmg

.. doxygenfunction:: hipblasSrotmgBatched
.. doxygenfunction:: hipblasDrotmgBatched

.. doxygenfunction:: hipblasSrotmgStridedBatched
.. doxygenfunction:: hipblasDrotmgStridedBatched

hipblasXscal + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasSscal
.. doxygenfunction:: hipblasDscal
.. doxygenfunction:: hipblasCscal
.. doxygenfunction:: hipblasCsscal
.. doxygenfunction:: hipblasZscal
.. doxygenfunction:: hipblasZdscal

.. doxygenfunction:: hipblasSscalBatched
.. doxygenfunction:: hipblasDscalBatched
.. doxygenfunction:: hipblasCscalBatched
.. doxygenfunction:: hipblasZscalBatched
.. doxygenfunction:: hipblasCsscalBatched
.. doxygenfunction:: hipblasZdscalBatched

.. doxygenfunction:: hipblasSscalStridedBatched
.. doxygenfunction:: hipblasDscalStridedBatched
.. doxygenfunction:: hipblasCscalStridedBatched
.. doxygenfunction:: hipblasZscalStridedBatched
.. doxygenfunction:: hipblasCsscalStridedBatched
.. doxygenfunction:: hipblasZdscalStridedBatched

hipblasXswap + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasSswap
.. doxygenfunction:: hipblasDswap
.. doxygenfunction:: hipblasCswap
.. doxygenfunction:: hipblasZswap

.. doxygenfunction:: hipblasSswapBatched
.. doxygenfunction:: hipblasDswapBatched
.. doxygenfunction:: hipblasCswapBatched
.. doxygenfunction:: hipblasZswapBatched

.. doxygenfunction:: hipblasSswapStridedBatched
.. doxygenfunction:: hipblasDswapStridedBatched
.. doxygenfunction:: hipblasCswapStridedBatched
.. doxygenfunction:: hipblasZswapStridedBatched


Level 2 BLAS
============
hipblasXgbmv + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasSgbmv
.. doxygenfunction:: hipblasDgbmv
.. doxygenfunction:: hipblasCgbmv
.. doxygenfunction:: hipblasZgbmv

.. doxygenfunction:: hipblasSgbmvBatched
.. doxygenfunction:: hipblasDgbmvBatched
.. doxygenfunction:: hipblasCgbmvBatched
.. doxygenfunction:: hipblasZgbmvBatched

.. doxygenfunction:: hipblasSgbmvStridedBatched
.. doxygenfunction:: hipblasDgbmvStridedBatched
.. doxygenfunction:: hipblasCgbmvStridedBatched
.. doxygenfunction:: hipblasZgbmvStridedBatched

hipblasXgemv + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasSgemv
.. doxygenfunction:: hipblasDgemv
.. doxygenfunction:: hipblasCgemv
.. doxygenfunction:: hipblasZgemv

.. doxygenfunction:: hipblasSgemvBatched
.. doxygenfunction:: hipblasDgemvBatched
.. doxygenfunction:: hipblasCgemvBatched
.. doxygenfunction:: hipblasZgemvBatched

.. doxygenfunction:: hipblasSgemvStridedBatched
.. doxygenfunction:: hipblasDgemvStridedBatched
.. doxygenfunction:: hipblasCgemvStridedBatched
.. doxygenfunction:: hipblasZgemvStridedBatched

hipblasXger + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasSger
.. doxygenfunction:: hipblasDger
.. doxygenfunction:: hipblasCgeru
.. doxygenfunction:: hipblasCgerc
.. doxygenfunction:: hipblasZgeru
.. doxygenfunction:: hipblasZgerc

.. doxygenfunction:: hipblasSgerBatched
.. doxygenfunction:: hipblasDgerBatched
.. doxygenfunction:: hipblasCgeruBatched
.. doxygenfunction:: hipblasCgercBatched
.. doxygenfunction:: hipblasZgeruBatched
.. doxygenfunction:: hipblasZgercBatched

.. doxygenfunction:: hipblasSgerStridedBatched
.. doxygenfunction:: hipblasDgerStridedBatched
.. doxygenfunction:: hipblasCgeruStridedBatched
.. doxygenfunction:: hipblasCgercStridedBatched
.. doxygenfunction:: hipblasZgeruStridedBatched
.. doxygenfunction:: hipblasZgercStridedBatched

hipblasXhbmv + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasChbmv
.. doxygenfunction:: hipblasZhbmv

.. doxygenfunction:: hipblasChbmvBatched
.. doxygenfunction:: hipblasZhbmvBatched

.. doxygenfunction:: hipblasChbmvStridedBatched
.. doxygenfunction:: hipblasZhbmvStridedBatched

hipblasXhemv + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasChemv
.. doxygenfunction:: hipblasZhemv

.. doxygenfunction:: hipblasChemvBatched
.. doxygenfunction:: hipblasZhemvBatched

.. doxygenfunction:: hipblasChemvStridedBatched
.. doxygenfunction:: hipblasZhemvStridedBatched

hipblasXher + Batched, StridedBatched
---------------------------------------
.. doxygenfunction:: hipblasCher
.. doxygenfunction:: hipblasZher

.. doxygenfunction:: hipblasCherBatched
.. doxygenfunction:: hipblasZherBatched

.. doxygenfunction:: hipblasCherStridedBatched
.. doxygenfunction:: hipblasZherStridedBatched

hipblasXher2 + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasCher2
.. doxygenfunction:: hipblasZher2

.. doxygenfunction:: hipblasCher2Batched
.. doxygenfunction:: hipblasZher2Batched

.. doxygenfunction:: hipblasCher2StridedBatched
.. doxygenfunction:: hipblasZher2StridedBatched

hipblasXhpmv + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasChpmv
.. doxygenfunction:: hipblasZhpmv

.. doxygenfunction:: hipblasChpmvBatched
.. doxygenfunction:: hipblasZhpmvBatched

.. doxygenfunction:: hipblasChpmvStridedBatched
.. doxygenfunction:: hipblasZhpmvStridedBatched

hipblasXhpr + Batched, StridedBatched
---------------------------------------
.. doxygenfunction:: hipblasChpr
.. doxygenfunction:: hipblasZhpr

.. doxygenfunction:: hipblasChprBatched
.. doxygenfunction:: hipblasZhprBatched

.. doxygenfunction:: hipblasChprStridedBatched
.. doxygenfunction:: hipblasZhprStridedBatched

hipblasXhpr2 + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasChpr2
.. doxygenfunction:: hipblasZhpr2

.. doxygenfunction:: hipblasChpr2Batched
.. doxygenfunction:: hipblasZhpr2Batched

.. doxygenfunction:: hipblasChpr2StridedBatched
.. doxygenfunction:: hipblasZhpr2StridedBatched

hipblasXsbmv + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasSsbmv
.. doxygenfunction:: hipblasDsbmv

.. doxygenfunction:: hipblasSsbmvBatched
.. doxygenfunction:: hipblasDsbmvBatched

.. doxygenfunction:: hipblasSsbmvStridedBatched
.. doxygenfunction:: hipblasDsbmvStridedBatched

hipblasXspmv + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasSspmv
.. doxygenfunction:: hipblasDspmv

.. doxygenfunction:: hipblasSspmvBatched
.. doxygenfunction:: hipblasDspmvBatched

.. doxygenfunction:: hipblasSspmvStridedBatched
.. doxygenfunction:: hipblasDspmvStridedBatched


hipblasXspr + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasSspr
.. doxygenfunction:: hipblasDspr
.. doxygenfunction:: hipblasCspr
.. doxygenfunction:: hipblasZspr

.. doxygenfunction:: hipblasSsprBatched
.. doxygenfunction:: hipblasDsprBatched
.. doxygenfunction:: hipblasCsprBatched
.. doxygenfunction:: hipblasZsprBatched

.. doxygenfunction:: hipblasSsprStridedBatched
.. doxygenfunction:: hipblasDsprStridedBatched
.. doxygenfunction:: hipblasCsprStridedBatched
.. doxygenfunction:: hipblasZsprStridedBatched

hipblasXspr2 + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasSspr2
.. doxygenfunction:: hipblasDspr2

.. doxygenfunction:: hipblasSspr2Batched
.. doxygenfunction:: hipblasDspr2Batched

.. doxygenfunction:: hipblasSspr2StridedBatched
.. doxygenfunction:: hipblasDspr2StridedBatched

hipblasXsymv + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasSsymv
.. doxygenfunction:: hipblasDsymv
.. doxygenfunction:: hipblasCsymv
.. doxygenfunction:: hipblasZsymv

.. doxygenfunction:: hipblasSsymvBatched
.. doxygenfunction:: hipblasDsymvBatched
.. doxygenfunction:: hipblasCsymvBatched
.. doxygenfunction:: hipblasZsymvBatched

.. doxygenfunction:: hipblasSsymvStridedBatched
.. doxygenfunction:: hipblasDsymvStridedBatched
.. doxygenfunction:: hipblasCsymvStridedBatched
.. doxygenfunction:: hipblasZsymvStridedBatched

hipblasXsyr + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasSsyr
.. doxygenfunction:: hipblasDsyr
.. doxygenfunction:: hipblasCsyr
.. doxygenfunction:: hipblasZsyr

.. doxygenfunction:: hipblasSsyrBatched
.. doxygenfunction:: hipblasDsyrBatched
.. doxygenfunction:: hipblasCsyrBatched
.. doxygenfunction:: hipblasZsyrBatched

.. doxygenfunction:: hipblasSsyrStridedBatched
.. doxygenfunction:: hipblasDsyrStridedBatched
.. doxygenfunction:: hipblasCsyrStridedBatched
.. doxygenfunction:: hipblasZsyrStridedBatched

hipblasXsyr2 + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasSsyr2
.. doxygenfunction:: hipblasDsyr2
.. doxygenfunction:: hipblasCsyr2
.. doxygenfunction:: hipblasZsyr2

.. doxygenfunction:: hipblasSsyr2Batched
.. doxygenfunction:: hipblasDsyr2Batched
.. doxygenfunction:: hipblasCsyr2Batched
.. doxygenfunction:: hipblasZsyr2Batched

.. doxygenfunction:: hipblasSsyr2StridedBatched
.. doxygenfunction:: hipblasDsyr2StridedBatched
.. doxygenfunction:: hipblasCsyr2StridedBatched
.. doxygenfunction:: hipblasZsyr2StridedBatched

hipblasXtbmv + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasStbmv
.. doxygenfunction:: hipblasDtbmv
.. doxygenfunction:: hipblasCtbmv
.. doxygenfunction:: hipblasZtbmv

.. doxygenfunction:: hipblasStbmvBatched
.. doxygenfunction:: hipblasDtbmvBatched
.. doxygenfunction:: hipblasCtbmvBatched
.. doxygenfunction:: hipblasZtbmvBatched

.. doxygenfunction:: hipblasStbmvStridedBatched
.. doxygenfunction:: hipblasCtbmvStridedBatched
.. doxygenfunction:: hipblasCtbmvStridedBatched
.. doxygenfunction:: hipblasZtbmvStridedBatched

hipblasXtbsv + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasStbsv
.. doxygenfunction:: hipblasDtbsv
.. doxygenfunction:: hipblasCtbsv
.. doxygenfunction:: hipblasZtbsv

.. doxygenfunction:: hipblasStbsvBatched
.. doxygenfunction:: hipblasDtbsvBatched
.. doxygenfunction:: hipblasCtbsvBatched
.. doxygenfunction:: hipblasZtbsvBatched

.. doxygenfunction:: hipblasStbsvStridedBatched
.. doxygenfunction:: hipblasDtbsvStridedBatched
.. doxygenfunction:: hipblasCtbsvStridedBatched
.. doxygenfunction:: hipblasZtbsvStridedBatched

hipblasXtpmv + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasStpmv
.. doxygenfunction:: hipblasDtpmv
.. doxygenfunction:: hipblasCtpmv
.. doxygenfunction:: hipblasZtpmv

.. doxygenfunction:: hipblasStpmvBatched
.. doxygenfunction:: hipblasDtpmvBatched
.. doxygenfunction:: hipblasCtpmvBatched
.. doxygenfunction:: hipblasZtpmvBatched

.. doxygenfunction:: hipblasStpmvStridedBatched
.. doxygenfunction:: hipblasDtpmvStridedBatched
.. doxygenfunction:: hipblasCtpmvStridedBatched
.. doxygenfunction:: hipblasZtpmvStridedBatched

hipblasXtpsv + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasStpsv
.. doxygenfunction:: hipblasDtpsv
.. doxygenfunction:: hipblasCtpsv
.. doxygenfunction:: hipblasZtpsv

.. doxygenfunction:: hipblasStpsvBatched
.. doxygenfunction:: hipblasDtpsvBatched
.. doxygenfunction:: hipblasCtpsvBatched
.. doxygenfunction:: hipblasZtpsvBatched

.. doxygenfunction:: hipblasStpsvStridedBatched
.. doxygenfunction:: hipblasDtpsvStridedBatched
.. doxygenfunction:: hipblasCtpsvStridedBatched
.. doxygenfunction:: hipblasZtpsvStridedBatched

hipblasXtrmv + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasStrmv
.. doxygenfunction:: hipblasDtrmv
.. doxygenfunction:: hipblasCtrmv
.. doxygenfunction:: hipblasZtrmv

.. doxygenfunction:: hipblasStrmvBatched
.. doxygenfunction:: hipblasDtrmvBatched
.. doxygenfunction:: hipblasCtrmvBatched
.. doxygenfunction:: hipblasZtrmvBatched

.. doxygenfunction:: hipblasStrmvStridedBatched
.. doxygenfunction:: hipblasDtrmvStridedBatched
.. doxygenfunction:: hipblasCtrmvStridedBatched
.. doxygenfunction:: hipblasZtrmvStridedBatched

hipblasXtrsv + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasStrsv
.. doxygenfunction:: hipblasDtrsv
.. doxygenfunction:: hipblasCtrsv
.. doxygenfunction:: hipblasZtrsv

.. doxygenfunction:: hipblasStrsvBatched
.. doxygenfunction:: hipblasDtrsvBatched
.. doxygenfunction:: hipblasCtrsvBatched
.. doxygenfunction:: hipblasZtrsvBatched

.. doxygenfunction:: hipblasStrsvStridedBatched
.. doxygenfunction:: hipblasDtrsvStridedBatched
.. doxygenfunction:: hipblasCtrsvStridedBatched
.. doxygenfunction:: hipblasZtrsvStridedBatched

Level 3 BLAS
============

hipblasXgemm + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasHgemm
.. doxygenfunction:: hipblasSgemm
.. doxygenfunction:: hipblasDgemm
.. doxygenfunction:: hipblasCgemm
.. doxygenfunction:: hipblasZgemm

.. doxygenfunction:: hipblasHgemmBatched
.. doxygenfunction:: hipblasSgemmBatched
.. doxygenfunction:: hipblasDgemmBatched
.. doxygenfunction:: hipblasCgemmBatched
.. doxygenfunction:: hipblasZgemmBatched

.. doxygenfunction:: hipblasHgemmStridedBatched
.. doxygenfunction:: hipblasSgemmStridedBatched
.. doxygenfunction:: hipblasDgemmStridedBatched
.. doxygenfunction:: hipblasCgemmStridedBatched
.. doxygenfunction:: hipblasZgemmStridedBatched

hipblasXherk + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasCherk
.. doxygenfunction:: hipblasZherk

.. doxygenfunction:: hipblasCherkBatched
.. doxygenfunction:: hipblasZherkBatched

.. doxygenfunction:: hipblasCherkStridedBatched
.. doxygenfunction:: hipblasZherkStridedBatched

hipblasXherkx + Batched, StridedBatched
-----------------------------------------
.. doxygenfunction:: hipblasCherkx
.. doxygenfunction:: hipblasZherkx

.. doxygenfunction:: hipblasCherkxBatched
.. doxygenfunction:: hipblasZherkxBatched

.. doxygenfunction:: hipblasCherkxStridedBatched
.. doxygenfunction:: hipblasZherkxStridedBatched

hipblasXher2k + Batched, StridedBatched
-----------------------------------------
.. doxygenfunction:: hipblasCher2k
.. doxygenfunction:: hipblasZher2k

.. doxygenfunction:: hipblasCher2kBatched
.. doxygenfunction:: hipblasZher2kBatched

.. doxygenfunction:: hipblasCher2kStridedBatched
.. doxygenfunction:: hipblasZher2kStridedBatched


hipblasXsymm + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasSsymm
.. doxygenfunction:: hipblasDsymm
.. doxygenfunction:: hipblasCsymm
.. doxygenfunction:: hipblasZsymm

.. doxygenfunction:: hipblasSsymmBatched
.. doxygenfunction:: hipblasDsymmBatched
.. doxygenfunction:: hipblasCsymmBatched
.. doxygenfunction:: hipblasZsymmBatched

.. doxygenfunction:: hipblasSsymmStridedBatched
.. doxygenfunction:: hipblasDsymmStridedBatched
.. doxygenfunction:: hipblasCsymmStridedBatched
.. doxygenfunction:: hipblasZsymmStridedBatched

hipblasXsyrk + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasSsyrk
.. doxygenfunction:: hipblasDsyrk
.. doxygenfunction:: hipblasCsyrk
.. doxygenfunction:: hipblasZsyrk

.. doxygenfunction:: hipblasSsyrkBatched
.. doxygenfunction:: hipblasDsyrkBatched
.. doxygenfunction:: hipblasCsyrkBatched
.. doxygenfunction:: hipblasZsyrkBatched

.. doxygenfunction:: hipblasSsyrkStridedBatched
.. doxygenfunction:: hipblasDsyrkStridedBatched
.. doxygenfunction:: hipblasCsyrkStridedBatched
.. doxygenfunction:: hipblasZsyrkStridedBatched

hipblasXsyr2k + Batched, StridedBatched
-----------------------------------------
.. doxygenfunction:: hipblasSsyr2k
.. doxygenfunction:: hipblasDsyr2k
.. doxygenfunction:: hipblasCsyr2k
.. doxygenfunction:: hipblasZsyr2k

.. doxygenfunction:: hipblasSsyr2kBatched
.. doxygenfunction:: hipblasDsyr2kBatched
.. doxygenfunction:: hipblasCsyr2kBatched
.. doxygenfunction:: hipblasZsyr2kBatched

.. doxygenfunction:: hipblasSsyr2kStridedBatched
.. doxygenfunction:: hipblasDsyr2kStridedBatched
.. doxygenfunction:: hipblasCsyr2kStridedBatched
.. doxygenfunction:: hipblasZsyr2kStridedBatched

hipblasXsyrkx + Batched, StridedBatched
-----------------------------------------
.. doxygenfunction:: hipblasSsyrkx
.. doxygenfunction:: hipblasDsyrkx
.. doxygenfunction:: hipblasCsyrkx
.. doxygenfunction:: hipblasZsyrkx

.. doxygenfunction:: hipblasSsyrkxBatched
.. doxygenfunction:: hipblasDsyrkxBatched
.. doxygenfunction:: hipblasCsyrkxBatched
.. doxygenfunction:: hipblasZsyrkxBatched

.. doxygenfunction:: hipblasSsyrkxStridedBatched
.. doxygenfunction:: hipblasDsyrkxStridedBatched
.. doxygenfunction:: hipblasCsyrkxStridedBatched
.. doxygenfunction:: hipblasZsyrkxStridedBatched

hipblasXgeam + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasSgeam
.. doxygenfunction:: hipblasDgeam
.. doxygenfunction:: hipblasCgeam
.. doxygenfunction:: hipblasZgeam

.. doxygenfunction:: hipblasSgeamBatched
.. doxygenfunction:: hipblasDgeamBatched
.. doxygenfunction:: hipblasCgeamBatched
.. doxygenfunction:: hipblasZgeamBatched

.. doxygenfunction:: hipblasSgeamStridedBatched
.. doxygenfunction:: hipblasDgeamStridedBatched
.. doxygenfunction:: hipblasCgeamStridedBatched
.. doxygenfunction:: hipblasZgeamStridedBatched

hipblasXhemm + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasChemm
.. doxygenfunction:: hipblasZhemm

.. doxygenfunction:: hipblasChemmBatched
.. doxygenfunction:: hipblasZhemmBatched

.. doxygenfunction:: hipblasChemmStridedBatched
.. doxygenfunction:: hipblasZhemmStridedBatched

hipblasXtrmm + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasStrmm
.. doxygenfunction:: hipblasDtrmm
.. doxygenfunction:: hipblasCtrmm
.. doxygenfunction:: hipblasZtrmm

.. doxygenfunction:: hipblasStrmmBatched
.. doxygenfunction:: hipblasDtrmmBatched
.. doxygenfunction:: hipblasCtrmmBatched
.. doxygenfunction:: hipblasZtrmmBatched

.. doxygenfunction:: hipblasStrmmStridedBatched
.. doxygenfunction:: hipblasDtrmmStridedBatched
.. doxygenfunction:: hipblasCtrmmStridedBatched
.. doxygenfunction:: hipblasZtrmmStridedBatched

hipblasXtrsm + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasStrsm
.. doxygenfunction:: hipblasDtrsm
.. doxygenfunction:: hipblasCtrsm
.. doxygenfunction:: hipblasZtrsm

.. doxygenfunction:: hipblasStrsmBatched
.. doxygenfunction:: hipblasDtrsmBatched
.. doxygenfunction:: hipblasCtrsmBatched
.. doxygenfunction:: hipblasZtrsmBatched

.. doxygenfunction:: hipblasStrsmStridedBatched
.. doxygenfunction:: hipblasDtrsmStridedBatched
.. doxygenfunction:: hipblasCtrsmStridedBatched
.. doxygenfunction:: hipblasZtrsmStridedBatched

hipblasXtrtri + Batched, StridedBatched
-----------------------------------------
.. doxygenfunction:: hipblasStrtri
.. doxygenfunction:: hipblasDtrtri
.. doxygenfunction:: hipblasCtrtri
.. doxygenfunction:: hipblasZtrtri

.. doxygenfunction:: hipblasStrtriBatched
.. doxygenfunction:: hipblasDtrtriBatched
.. doxygenfunction:: hipblasCtrtriBatched
.. doxygenfunction:: hipblasZtrtriBatched

.. doxygenfunction:: hipblasStrtriStridedBatched
.. doxygenfunction:: hipblasDtrtriStridedBatched
.. doxygenfunction:: hipblasCtrtriStridedBatched
.. doxygenfunction:: hipblasZtrtriStridedBatched

hipblasXdgmm + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasSdgmm
.. doxygenfunction:: hipblasDdgmm
.. doxygenfunction:: hipblasCdgmm
.. doxygenfunction:: hipblasZdgmm

.. doxygenfunction:: hipblasSdgmmBatched
.. doxygenfunction:: hipblasDdgmmBatched
.. doxygenfunction:: hipblasCdgmmBatched
.. doxygenfunction:: hipblasZdgmmBatched

.. doxygenfunction:: hipblasSdgmmStridedBatched
.. doxygenfunction:: hipblasDdgmmStridedBatched
.. doxygenfunction:: hipblasCdgmmStridedBatched
.. doxygenfunction:: hipblasZdgmmStridedBatched

SOLVER API
===========

hipblasXgetrf + Batched, stridedBatched
----------------------------------------
.. doxygenfunction:: hipblasSgetrf
.. doxygenfunction:: hipblasDgetrf
.. doxygenfunction:: hipblasCgetrf
.. doxygenfunction:: hipblasZgetrf

.. doxygenfunction:: hipblasSgetrfBatched
.. doxygenfunction:: hipblasDgetrfBatched
.. doxygenfunction:: hipblasCgetrfBatched
.. doxygenfunction:: hipblasZgetrfBatched

.. doxygenfunction:: hipblasSgetrfStridedBatched
.. doxygenfunction:: hipblasDgetrfStridedBatched
.. doxygenfunction:: hipblasCgetrfStridedBatched
.. doxygenfunction:: hipblasZgetrfStridedBatched


hipblasXgetrs + Batched, stridedBatched
----------------------------------------
.. doxygenfunction:: hipblasSgetrs
.. doxygenfunction:: hipblasDgetrs
.. doxygenfunction:: hipblasCgetrs
.. doxygenfunction:: hipblasZgetrs

.. doxygenfunction:: hipblasSgetrsBatched
.. doxygenfunction:: hipblasDgetrsBatched
.. doxygenfunction:: hipblasCgetrsBatched
.. doxygenfunction:: hipblasZgetrsBatched

.. doxygenfunction:: hipblasSgetrsStridedBatched
.. doxygenfunction:: hipblasDgetrsStridedBatched
.. doxygenfunction:: hipblasCgetrsStridedBatched
.. doxygenfunction:: hipblasZgetrsStridedBatched

hipblasXgetri + Batched, stridedBatched
----------------------------------------

.. doxygenfunction:: hipblasSgetriBatched
.. doxygenfunction:: hipblasDgetriBatched
.. doxygenfunction:: hipblasCgetriBatched
.. doxygenfunction:: hipblasZgetriBatched

hipblasXgeqrf + Batched, stridedBatched
----------------------------------------
.. doxygenfunction:: hipblasSgeqrf
.. doxygenfunction:: hipblasDgeqrf
.. doxygenfunction:: hipblasCgeqrf
.. doxygenfunction:: hipblasZgeqrf

.. doxygenfunction:: hipblasSgeqrfBatched
.. doxygenfunction:: hipblasDgeqrfBatched
.. doxygenfunction:: hipblasCgeqrfBatched
.. doxygenfunction:: hipblasZgeqrfBatched

.. doxygenfunction:: hipblasSgeqrfStridedBatched
.. doxygenfunction:: hipblasDgeqrfStridedBatched
.. doxygenfunction:: hipblasCgeqrfStridedBatched
.. doxygenfunction:: hipblasZgeqrfStridedBatched

BLAS Extensions
===============

hipblasGemmEx + Batched, StridedBatched
------------------------------------------
.. doxygenfunction:: hipblasGemmEx
.. doxygenfunction:: hipblasGemmBatchedEx
.. doxygenfunction:: hipblasGemmStridedBatchedEx

hipblasTrsmEx + Batched, StridedBatched
------------------------------------------
.. doxygenfunction:: hipblasTrsmEx
.. doxygenfunction:: hipblasTrsmBatchedEx
.. doxygenfunction:: hipblasTrsmStridedBatchedEx

hipblasAxpyEx + Batched, StridedBatched
------------------------------------------
.. doxygenfunction:: hipblasAxpyEx
.. doxygenfunction:: hipblasAxpyBatchedEx
.. doxygenfunction:: hipblasAxpyStridedBatchedEx

hipblasDotEx + Batched, StridedBatched
------------------------------------------
.. doxygenfunction:: hipblasDotEx
.. doxygenfunction:: hipblasDotBatchedEx
.. doxygenfunction:: hipblasDotStridedBatchedEx

hipblasDotcEx + Batched, StridedBatched
------------------------------------------
.. doxygenfunction:: hipblasDotcEx
.. doxygenfunction:: hipblasDotcBatchedEx
.. doxygenfunction:: hipblasDotcStridedBatchedEx

hipblasNrm2Ex + Batched, StridedBatched
------------------------------------------
.. doxygenfunction:: hipblasNrm2Ex
.. doxygenfunction:: hipblasNrm2BatchedEx
.. doxygenfunction:: hipblasNrm2StridedBatchedEx

hipblasRotEx + Batched, StridedBatched
------------------------------------------
.. doxygenfunction:: hipblasRotEx
.. doxygenfunction:: hipblasRotBatchedEx
.. doxygenfunction:: hipblasRotStridedBatchedEx

hipblasScalEx + Batched, StridedBatched
------------------------------------------
.. doxygenfunction:: hipblasScalEx
.. doxygenfunction:: hipblasScalBatchedEx
.. doxygenfunction:: hipblasScalStridedBatchedEx


Auxiliary
=========

hipblasCreate
-----------------------
.. doxygenfunction:: hipblasCreate

hipblasDestroy
---------------------
.. doxygenfunction:: hipblasDestroy

hipblasSetStream
----------------------
.. doxygenfunction:: hipblasSetStream

hipblasGetStream
------------------
.. doxygenfunction:: hipblasGetStream

hipblasSetPointerMode
------------------
.. doxygenfunction:: hipblasSetPointerMode

hipblasGetPointerMode
------------------------
.. doxygenfunction:: hipblasGetPointerMode

hipblasSetVector
------------------------
.. doxygenfunction:: hipblasSetVector

hipblasGetVector
------------------------
.. doxygenfunction:: hipblasGetVector

hipblasSetMatrix
------------------------
.. doxygenfunction:: hipblasSetMatrix

hipblasGetMatrix
------------------
.. doxygenfunction:: hipblasGetMatrix

hipblasSetVectorAsync
------------------------
.. doxygenfunction:: hipblasSetVectorAsync

hipblasGetVectorAsync
------------------
.. doxygenfunction:: hipblasGetVectorAsync

hipblasSetMatrixAsync
------------------------
.. doxygenfunction:: hipblasSetMatrixAsync

hipblasGetMatrixAsync
------------------
.. doxygenfunction:: hipblasGetMatrixAsync

hipblasSetAtomicsMode
------------------------
.. doxygenfunction:: hipblasSetAtomicsMode

hipblasGetAtomicsMode
------------------
.. doxygenfunction:: hipblasGetAtomicsMode

hipblasStatusToString
---------------------
.. doxygenfunction:: hipblasStatusToString
