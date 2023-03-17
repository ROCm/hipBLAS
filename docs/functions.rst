.. _api_label:


*************
Guidelines
*************

Naming conventions
==================

hipBLAS follows the following naming conventions,

- Big case for matrix, e.g. matrix A, B, C   GEMM (C = A*B)
- Lower case for vector, e.g. vector x, y    GEMV (y = A*x)


Notations
=========

hipBLAS function uses the following notations to denote precisions,

- h  = half
- bf = 16 bit floating point
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
---------------
.. doxygentypedef:: hipblasHandle_t

hipblasHalf
------------
.. doxygentypedef:: hipblasHalf

hipblasInt8
------------
.. doxygentypedef:: hipblasInt8

hipblasStride
--------------
.. doxygentypedef:: hipblasStride

hipblasBfloat16
----------------
.. doxygenstruct:: hipblasBfloat16

hipblasComplex
---------------
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
------------------
.. doxygenenum:: hipblasOperation_t

hipblasPointerMode_t
--------------------
.. doxygenenum:: hipblasPointerMode_t

hipblasFillMode_t
------------------
.. doxygenenum:: hipblasFillMode_t

hipblasDiagType_t
-----------------
.. doxygenenum:: hipblasDiagType_t

hipblasSideMode_t
-----------------
.. doxygenenum:: hipblasSideMode_t

hipblasDatatype_t
------------------
.. doxygenenum:: hipblasDatatype_t

hipblasGemmAlgo_t
------------------
.. doxygenenum:: hipblasGemmAlgo_t

hipblasAtomicsMode_t
---------------------
.. doxygenenum:: hipblasAtomicsMode_t

*****************
hipBLAS Functions
*****************

Level 1 BLAS
============

.. contents:: List of Level-1 BLAS Functions
   :local:
   :backlinks: top

hipblasIXamax + Batched, StridedBatched
-----------------------------------------
.. doxygenfunction:: hipblasIsamax
    :outline:
.. doxygenfunction:: hipblasIdamax
    :outline:
.. doxygenfunction:: hipblasIcamax
    :outline:
.. doxygenfunction:: hipblasIzamax

.. doxygenfunction:: hipblasIsamaxBatched
    :outline:
.. doxygenfunction:: hipblasIdamaxBatched
    :outline:
.. doxygenfunction:: hipblasIcamaxBatched
    :outline:
.. doxygenfunction:: hipblasIzamaxBatched

.. doxygenfunction:: hipblasIsamaxStridedBatched
    :outline:
.. doxygenfunction:: hipblasIdamaxStridedBatched
    :outline:
.. doxygenfunction:: hipblasIcamaxStridedBatched
    :outline:
.. doxygenfunction:: hipblasIzamaxStridedBatched


hipblasIXamin + Batched, StridedBatched
-----------------------------------------
.. doxygenfunction:: hipblasIsamin
    :outline:
.. doxygenfunction:: hipblasIdamin
    :outline:
.. doxygenfunction:: hipblasIcamin
    :outline:
.. doxygenfunction:: hipblasIzamin

.. doxygenfunction:: hipblasIsaminBatched
    :outline:
.. doxygenfunction:: hipblasIdaminBatched
    :outline:
.. doxygenfunction:: hipblasIcaminBatched
    :outline:
.. doxygenfunction:: hipblasIzaminBatched

.. doxygenfunction:: hipblasIsaminStridedBatched
    :outline:
.. doxygenfunction:: hipblasIdaminStridedBatched
    :outline:
.. doxygenfunction:: hipblasIcaminStridedBatched
    :outline:
.. doxygenfunction:: hipblasIzaminStridedBatched

hipblasXasum + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasSasum
    :outline:
.. doxygenfunction:: hipblasDasum
    :outline:
.. doxygenfunction:: hipblasScasum
    :outline:
.. doxygenfunction:: hipblasDzasum

.. doxygenfunction:: hipblasSasumBatched
    :outline:
.. doxygenfunction:: hipblasDasumBatched
    :outline:
.. doxygenfunction:: hipblasScasumBatched
    :outline:
.. doxygenfunction:: hipblasDzasumBatched

.. doxygenfunction:: hipblasSasumStridedBatched
    :outline:
.. doxygenfunction:: hipblasDasumStridedBatched
    :outline:
.. doxygenfunction:: hipblasScasumStridedBatched
    :outline:
.. doxygenfunction:: hipblasDzasumStridedBatched

hipblasXaxpy + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasHaxpy
    :outline:
.. doxygenfunction:: hipblasSaxpy
    :outline:
.. doxygenfunction:: hipblasDaxpy
    :outline:
.. doxygenfunction:: hipblasCaxpy
    :outline:
.. doxygenfunction:: hipblasZaxpy

.. doxygenfunction:: hipblasHaxpyBatched
    :outline:
.. doxygenfunction:: hipblasSaxpyBatched
    :outline:
.. doxygenfunction:: hipblasDaxpyBatched
    :outline:
.. doxygenfunction:: hipblasCaxpyBatched
    :outline:
.. doxygenfunction:: hipblasZaxpyBatched

.. doxygenfunction:: hipblasHaxpyStridedBatched
    :outline:
.. doxygenfunction:: hipblasSaxpyStridedBatched
    :outline:
.. doxygenfunction:: hipblasDaxpyStridedBatched
    :outline:
.. doxygenfunction:: hipblasCaxpyStridedBatched
    :outline:
.. doxygenfunction:: hipblasZaxpyStridedBatched

hipblasXcopy + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasScopy
    :outline:
.. doxygenfunction:: hipblasDcopy
    :outline:
.. doxygenfunction:: hipblasCcopy
    :outline:
.. doxygenfunction:: hipblasZcopy

.. doxygenfunction:: hipblasScopyBatched
    :outline:
.. doxygenfunction:: hipblasDcopyBatched
    :outline:
.. doxygenfunction:: hipblasCcopyBatched
    :outline:
.. doxygenfunction:: hipblasZcopyBatched

.. doxygenfunction:: hipblasScopyStridedBatched
    :outline:
.. doxygenfunction:: hipblasDcopyStridedBatched
    :outline:
.. doxygenfunction:: hipblasCcopyStridedBatched
    :outline:
.. doxygenfunction:: hipblasZcopyStridedBatched

hipblasXdot + Batched, StridedBatched
---------------------------------------
.. doxygenfunction:: hipblasHdot
    :outline:
.. doxygenfunction:: hipblasBfdot
    :outline:
.. doxygenfunction:: hipblasSdot
    :outline:
.. doxygenfunction:: hipblasDdot
    :outline:
.. doxygenfunction:: hipblasCdotc
    :outline:
.. doxygenfunction:: hipblasCdotu
    :outline:
.. doxygenfunction:: hipblasZdotc
    :outline:
.. doxygenfunction:: hipblasZdotu

.. doxygenfunction:: hipblasHdotBatched
    :outline:
.. doxygenfunction:: hipblasBfdotBatched
    :outline:
.. doxygenfunction:: hipblasSdotBatched
    :outline:
.. doxygenfunction:: hipblasDdotBatched
    :outline:
.. doxygenfunction:: hipblasCdotcBatched
    :outline:
.. doxygenfunction:: hipblasCdotuBatched
    :outline:
.. doxygenfunction:: hipblasZdotcBatched
    :outline:
.. doxygenfunction:: hipblasZdotuBatched

.. doxygenfunction:: hipblasHdotStridedBatched
    :outline:
.. doxygenfunction:: hipblasBfdotStridedBatched
    :outline:
.. doxygenfunction:: hipblasSdotStridedBatched
    :outline:
.. doxygenfunction:: hipblasDdotStridedBatched
    :outline:
.. doxygenfunction:: hipblasCdotcStridedBatched
    :outline:
.. doxygenfunction:: hipblasCdotuStridedBatched
    :outline:
.. doxygenfunction:: hipblasZdotcStridedBatched
    :outline:
.. doxygenfunction:: hipblasZdotuStridedBatched

hipblasXnrm2 + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasSnrm2
    :outline:
.. doxygenfunction:: hipblasDnrm2
    :outline:
.. doxygenfunction:: hipblasScnrm2
    :outline:
.. doxygenfunction:: hipblasDznrm2

.. doxygenfunction:: hipblasSnrm2Batched
    :outline:
.. doxygenfunction:: hipblasDnrm2Batched
    :outline:
.. doxygenfunction:: hipblasScnrm2Batched
    :outline:
.. doxygenfunction:: hipblasDznrm2Batched

.. doxygenfunction:: hipblasSnrm2StridedBatched
    :outline:
.. doxygenfunction:: hipblasDnrm2StridedBatched
    :outline:
.. doxygenfunction:: hipblasScnrm2StridedBatched
    :outline:
.. doxygenfunction:: hipblasDznrm2StridedBatched

hipblasXrot + Batched, StridedBatched
---------------------------------------
.. doxygenfunction:: hipblasSrot
    :outline:
.. doxygenfunction:: hipblasDrot
    :outline:
.. doxygenfunction:: hipblasCrot
    :outline:
.. doxygenfunction:: hipblasCsrot
    :outline:
.. doxygenfunction:: hipblasZrot
    :outline:
.. doxygenfunction:: hipblasZdrot

.. doxygenfunction:: hipblasSrotBatched
    :outline:
.. doxygenfunction:: hipblasDrotBatched
    :outline:
.. doxygenfunction:: hipblasCrotBatched
    :outline:
.. doxygenfunction:: hipblasCsrotBatched
    :outline:
.. doxygenfunction:: hipblasZrotBatched
    :outline:
.. doxygenfunction:: hipblasZdrotBatched

.. doxygenfunction:: hipblasSrotStridedBatched
    :outline:
.. doxygenfunction:: hipblasDrotStridedBatched
    :outline:
.. doxygenfunction:: hipblasCrotStridedBatched
    :outline:
.. doxygenfunction:: hipblasCsrotStridedBatched
    :outline:
.. doxygenfunction:: hipblasZrotStridedBatched
    :outline:
.. doxygenfunction:: hipblasZdrotStridedBatched

hipblasXrotg + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasSrotg
    :outline:
.. doxygenfunction:: hipblasDrotg
    :outline:
.. doxygenfunction:: hipblasCrotg
    :outline:
.. doxygenfunction:: hipblasZrotg

.. doxygenfunction:: hipblasSrotgBatched
    :outline:
.. doxygenfunction:: hipblasDrotgBatched
    :outline:
.. doxygenfunction:: hipblasCrotgBatched
    :outline:
.. doxygenfunction:: hipblasZrotgBatched

.. doxygenfunction:: hipblasSrotgStridedBatched
    :outline:
.. doxygenfunction:: hipblasDrotgStridedBatched
    :outline:
.. doxygenfunction:: hipblasCrotgStridedBatched
    :outline:
.. doxygenfunction:: hipblasZrotgStridedBatched

hipblasXrotm + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasSrotm
    :outline:
.. doxygenfunction:: hipblasDrotm

.. doxygenfunction:: hipblasSrotmBatched
    :outline:
.. doxygenfunction:: hipblasDrotmBatched

.. doxygenfunction:: hipblasSrotmStridedBatched
    :outline:
.. doxygenfunction:: hipblasDrotmStridedBatched

hipblasXrotmg + Batched, StridedBatched
-----------------------------------------
.. doxygenfunction:: hipblasSrotmg
    :outline:
.. doxygenfunction:: hipblasDrotmg

.. doxygenfunction:: hipblasSrotmgBatched
    :outline:
.. doxygenfunction:: hipblasDrotmgBatched

.. doxygenfunction:: hipblasSrotmgStridedBatched
    :outline:
.. doxygenfunction:: hipblasDrotmgStridedBatched

hipblasXscal + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasSscal
    :outline:
.. doxygenfunction:: hipblasDscal
    :outline:
.. doxygenfunction:: hipblasCscal
    :outline:
.. doxygenfunction:: hipblasCsscal
    :outline:
.. doxygenfunction:: hipblasZscal
    :outline:
.. doxygenfunction:: hipblasZdscal

.. doxygenfunction:: hipblasSscalBatched
    :outline:
.. doxygenfunction:: hipblasDscalBatched
    :outline:
.. doxygenfunction:: hipblasCscalBatched
    :outline:
.. doxygenfunction:: hipblasZscalBatched
    :outline:
.. doxygenfunction:: hipblasCsscalBatched
    :outline:
.. doxygenfunction:: hipblasZdscalBatched

.. doxygenfunction:: hipblasSscalStridedBatched
    :outline:
.. doxygenfunction:: hipblasDscalStridedBatched
    :outline:
.. doxygenfunction:: hipblasCscalStridedBatched
    :outline:
.. doxygenfunction:: hipblasZscalStridedBatched
    :outline:
.. doxygenfunction:: hipblasCsscalStridedBatched
    :outline:
.. doxygenfunction:: hipblasZdscalStridedBatched

hipblasXswap + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasSswap
    :outline:
.. doxygenfunction:: hipblasDswap
    :outline:
.. doxygenfunction:: hipblasCswap
    :outline:
.. doxygenfunction:: hipblasZswap

.. doxygenfunction:: hipblasSswapBatched
    :outline:
.. doxygenfunction:: hipblasDswapBatched
    :outline:
.. doxygenfunction:: hipblasCswapBatched
    :outline:
.. doxygenfunction:: hipblasZswapBatched

.. doxygenfunction:: hipblasSswapStridedBatched
    :outline:
.. doxygenfunction:: hipblasDswapStridedBatched
    :outline:
.. doxygenfunction:: hipblasCswapStridedBatched
    :outline:
.. doxygenfunction:: hipblasZswapStridedBatched


Level 2 BLAS
============
.. contents:: List of Level-2 BLAS Functions
   :local:
   :backlinks: top

hipblasXgbmv + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasSgbmv
    :outline:
.. doxygenfunction:: hipblasDgbmv
    :outline:
.. doxygenfunction:: hipblasCgbmv
    :outline:
.. doxygenfunction:: hipblasZgbmv

.. doxygenfunction:: hipblasSgbmvBatched
    :outline:
.. doxygenfunction:: hipblasDgbmvBatched
    :outline:
.. doxygenfunction:: hipblasCgbmvBatched
    :outline:
.. doxygenfunction:: hipblasZgbmvBatched

.. doxygenfunction:: hipblasSgbmvStridedBatched
    :outline:
.. doxygenfunction:: hipblasDgbmvStridedBatched
    :outline:
.. doxygenfunction:: hipblasCgbmvStridedBatched
    :outline:
.. doxygenfunction:: hipblasZgbmvStridedBatched

hipblasXgemv + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasSgemv
    :outline:
.. doxygenfunction:: hipblasDgemv
    :outline:
.. doxygenfunction:: hipblasCgemv
    :outline:
.. doxygenfunction:: hipblasZgemv

.. doxygenfunction:: hipblasSgemvBatched
    :outline:
.. doxygenfunction:: hipblasDgemvBatched
    :outline:
.. doxygenfunction:: hipblasCgemvBatched
    :outline:
.. doxygenfunction:: hipblasZgemvBatched

.. doxygenfunction:: hipblasSgemvStridedBatched
    :outline:
.. doxygenfunction:: hipblasDgemvStridedBatched
    :outline:
.. doxygenfunction:: hipblasCgemvStridedBatched
    :outline:
.. doxygenfunction:: hipblasZgemvStridedBatched

hipblasXger + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasSger
    :outline:
.. doxygenfunction:: hipblasDger
    :outline:
.. doxygenfunction:: hipblasCgeru
    :outline:
.. doxygenfunction:: hipblasCgerc
    :outline:
.. doxygenfunction:: hipblasZgeru
    :outline:
.. doxygenfunction:: hipblasZgerc

.. doxygenfunction:: hipblasSgerBatched
    :outline:
.. doxygenfunction:: hipblasDgerBatched
    :outline:
.. doxygenfunction:: hipblasCgeruBatched
    :outline:
.. doxygenfunction:: hipblasCgercBatched
    :outline:
.. doxygenfunction:: hipblasZgeruBatched
    :outline:
.. doxygenfunction:: hipblasZgercBatched

.. doxygenfunction:: hipblasSgerStridedBatched
    :outline:
.. doxygenfunction:: hipblasDgerStridedBatched
    :outline:
.. doxygenfunction:: hipblasCgeruStridedBatched
    :outline:
.. doxygenfunction:: hipblasCgercStridedBatched
    :outline:
.. doxygenfunction:: hipblasZgeruStridedBatched
    :outline:
.. doxygenfunction:: hipblasZgercStridedBatched

hipblasXhbmv + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasChbmv
    :outline:
.. doxygenfunction:: hipblasZhbmv

.. doxygenfunction:: hipblasChbmvBatched
    :outline:
.. doxygenfunction:: hipblasZhbmvBatched

.. doxygenfunction:: hipblasChbmvStridedBatched
    :outline:
.. doxygenfunction:: hipblasZhbmvStridedBatched

hipblasXhemv + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasChemv
    :outline:
.. doxygenfunction:: hipblasZhemv

.. doxygenfunction:: hipblasChemvBatched
    :outline:
.. doxygenfunction:: hipblasZhemvBatched

.. doxygenfunction:: hipblasChemvStridedBatched
    :outline:
.. doxygenfunction:: hipblasZhemvStridedBatched

hipblasXher + Batched, StridedBatched
---------------------------------------
.. doxygenfunction:: hipblasCher
    :outline:
.. doxygenfunction:: hipblasZher

.. doxygenfunction:: hipblasCherBatched
    :outline:
.. doxygenfunction:: hipblasZherBatched

.. doxygenfunction:: hipblasCherStridedBatched
    :outline:
.. doxygenfunction:: hipblasZherStridedBatched

hipblasXher2 + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasCher2
    :outline:
.. doxygenfunction:: hipblasZher2

.. doxygenfunction:: hipblasCher2Batched
    :outline:
.. doxygenfunction:: hipblasZher2Batched

.. doxygenfunction:: hipblasCher2StridedBatched
    :outline:
.. doxygenfunction:: hipblasZher2StridedBatched

hipblasXhpmv + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasChpmv
    :outline:
.. doxygenfunction:: hipblasZhpmv

.. doxygenfunction:: hipblasChpmvBatched
    :outline:
.. doxygenfunction:: hipblasZhpmvBatched

.. doxygenfunction:: hipblasChpmvStridedBatched
    :outline:
.. doxygenfunction:: hipblasZhpmvStridedBatched

hipblasXhpr + Batched, StridedBatched
---------------------------------------
.. doxygenfunction:: hipblasChpr
    :outline:
.. doxygenfunction:: hipblasZhpr

.. doxygenfunction:: hipblasChprBatched
    :outline:
.. doxygenfunction:: hipblasZhprBatched

.. doxygenfunction:: hipblasChprStridedBatched
    :outline:
.. doxygenfunction:: hipblasZhprStridedBatched

hipblasXhpr2 + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasChpr2
    :outline:
.. doxygenfunction:: hipblasZhpr2

.. doxygenfunction:: hipblasChpr2Batched
    :outline:
.. doxygenfunction:: hipblasZhpr2Batched

.. doxygenfunction:: hipblasChpr2StridedBatched
    :outline:
.. doxygenfunction:: hipblasZhpr2StridedBatched

hipblasXsbmv + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasSsbmv
    :outline:
.. doxygenfunction:: hipblasDsbmv

.. doxygenfunction:: hipblasSsbmvBatched
    :outline:
.. doxygenfunction:: hipblasDsbmvBatched

.. doxygenfunction:: hipblasSsbmvStridedBatched
    :outline:
.. doxygenfunction:: hipblasDsbmvStridedBatched

hipblasXspmv + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasSspmv
    :outline:
.. doxygenfunction:: hipblasDspmv

.. doxygenfunction:: hipblasSspmvBatched
    :outline:
.. doxygenfunction:: hipblasDspmvBatched

.. doxygenfunction:: hipblasSspmvStridedBatched
    :outline:
.. doxygenfunction:: hipblasDspmvStridedBatched


hipblasXspr + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasSspr
    :outline:
.. doxygenfunction:: hipblasDspr
    :outline:
.. doxygenfunction:: hipblasCspr
    :outline:
.. doxygenfunction:: hipblasZspr

.. doxygenfunction:: hipblasSsprBatched
    :outline:
.. doxygenfunction:: hipblasDsprBatched
    :outline:
.. doxygenfunction:: hipblasCsprBatched
    :outline:
.. doxygenfunction:: hipblasZsprBatched

.. doxygenfunction:: hipblasSsprStridedBatched
    :outline:
.. doxygenfunction:: hipblasDsprStridedBatched
    :outline:
.. doxygenfunction:: hipblasCsprStridedBatched
    :outline:
.. doxygenfunction:: hipblasZsprStridedBatched

hipblasXspr2 + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasSspr2
    :outline:
.. doxygenfunction:: hipblasDspr2

.. doxygenfunction:: hipblasSspr2Batched
    :outline:
.. doxygenfunction:: hipblasDspr2Batched

.. doxygenfunction:: hipblasSspr2StridedBatched
    :outline:
.. doxygenfunction:: hipblasDspr2StridedBatched

hipblasXsymv + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasSsymv
    :outline:
.. doxygenfunction:: hipblasDsymv
    :outline:
.. doxygenfunction:: hipblasCsymv
    :outline:
.. doxygenfunction:: hipblasZsymv

.. doxygenfunction:: hipblasSsymvBatched
    :outline:
.. doxygenfunction:: hipblasDsymvBatched
    :outline:
.. doxygenfunction:: hipblasCsymvBatched
    :outline:
.. doxygenfunction:: hipblasZsymvBatched

.. doxygenfunction:: hipblasSsymvStridedBatched
    :outline:
.. doxygenfunction:: hipblasDsymvStridedBatched
    :outline:
.. doxygenfunction:: hipblasCsymvStridedBatched
    :outline:
.. doxygenfunction:: hipblasZsymvStridedBatched

hipblasXsyr + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasSsyr
    :outline:
.. doxygenfunction:: hipblasDsyr
    :outline:
.. doxygenfunction:: hipblasCsyr
    :outline:
.. doxygenfunction:: hipblasZsyr

.. doxygenfunction:: hipblasSsyrBatched
    :outline:
.. doxygenfunction:: hipblasDsyrBatched
    :outline:
.. doxygenfunction:: hipblasCsyrBatched
    :outline:
.. doxygenfunction:: hipblasZsyrBatched

.. doxygenfunction:: hipblasSsyrStridedBatched
    :outline:
.. doxygenfunction:: hipblasDsyrStridedBatched
    :outline:
.. doxygenfunction:: hipblasCsyrStridedBatched
    :outline:
.. doxygenfunction:: hipblasZsyrStridedBatched

hipblasXsyr2 + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasSsyr2
    :outline:
.. doxygenfunction:: hipblasDsyr2
    :outline:
.. doxygenfunction:: hipblasCsyr2
    :outline:
.. doxygenfunction:: hipblasZsyr2

.. doxygenfunction:: hipblasSsyr2Batched
    :outline:
.. doxygenfunction:: hipblasDsyr2Batched
    :outline:
.. doxygenfunction:: hipblasCsyr2Batched
    :outline:
.. doxygenfunction:: hipblasZsyr2Batched

.. doxygenfunction:: hipblasSsyr2StridedBatched
    :outline:
.. doxygenfunction:: hipblasDsyr2StridedBatched
    :outline:
.. doxygenfunction:: hipblasCsyr2StridedBatched
    :outline:
.. doxygenfunction:: hipblasZsyr2StridedBatched

hipblasXtbmv + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasStbmv
    :outline:
.. doxygenfunction:: hipblasDtbmv
    :outline:
.. doxygenfunction:: hipblasCtbmv
    :outline:
.. doxygenfunction:: hipblasZtbmv

.. doxygenfunction:: hipblasStbmvBatched
    :outline:
.. doxygenfunction:: hipblasDtbmvBatched
    :outline:
.. doxygenfunction:: hipblasCtbmvBatched
    :outline:
.. doxygenfunction:: hipblasZtbmvBatched

.. doxygenfunction:: hipblasStbmvStridedBatched
    :outline:
.. doxygenfunction:: hipblasDtbmvStridedBatched
    :outline:
.. doxygenfunction:: hipblasCtbmvStridedBatched
    :outline:
.. doxygenfunction:: hipblasZtbmvStridedBatched

hipblasXtbsv + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasStbsv
    :outline:
.. doxygenfunction:: hipblasDtbsv
    :outline:
.. doxygenfunction:: hipblasCtbsv
    :outline:
.. doxygenfunction:: hipblasZtbsv

.. doxygenfunction:: hipblasStbsvBatched
    :outline:
.. doxygenfunction:: hipblasDtbsvBatched
    :outline:
.. doxygenfunction:: hipblasCtbsvBatched
    :outline:
.. doxygenfunction:: hipblasZtbsvBatched

.. doxygenfunction:: hipblasStbsvStridedBatched
    :outline:
.. doxygenfunction:: hipblasDtbsvStridedBatched
    :outline:
.. doxygenfunction:: hipblasCtbsvStridedBatched
    :outline:
.. doxygenfunction:: hipblasZtbsvStridedBatched

hipblasXtpmv + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasStpmv
    :outline:
.. doxygenfunction:: hipblasDtpmv
    :outline:
.. doxygenfunction:: hipblasCtpmv
    :outline:
.. doxygenfunction:: hipblasZtpmv

.. doxygenfunction:: hipblasStpmvBatched
    :outline:
.. doxygenfunction:: hipblasDtpmvBatched
    :outline:
.. doxygenfunction:: hipblasCtpmvBatched
    :outline:
.. doxygenfunction:: hipblasZtpmvBatched

.. doxygenfunction:: hipblasStpmvStridedBatched
    :outline:
.. doxygenfunction:: hipblasDtpmvStridedBatched
    :outline:
.. doxygenfunction:: hipblasCtpmvStridedBatched
    :outline:
.. doxygenfunction:: hipblasZtpmvStridedBatched

hipblasXtpsv + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasStpsv
    :outline:
.. doxygenfunction:: hipblasDtpsv
    :outline:
.. doxygenfunction:: hipblasCtpsv
    :outline:
.. doxygenfunction:: hipblasZtpsv

.. doxygenfunction:: hipblasStpsvBatched
    :outline:
.. doxygenfunction:: hipblasDtpsvBatched
    :outline:
.. doxygenfunction:: hipblasCtpsvBatched
    :outline:
.. doxygenfunction:: hipblasZtpsvBatched

.. doxygenfunction:: hipblasStpsvStridedBatched
    :outline:
.. doxygenfunction:: hipblasDtpsvStridedBatched
    :outline:
.. doxygenfunction:: hipblasCtpsvStridedBatched
    :outline:
.. doxygenfunction:: hipblasZtpsvStridedBatched

hipblasXtrmv + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasStrmv
    :outline:
.. doxygenfunction:: hipblasDtrmv
    :outline:
.. doxygenfunction:: hipblasCtrmv
    :outline:
.. doxygenfunction:: hipblasZtrmv

.. doxygenfunction:: hipblasStrmvBatched
    :outline:
.. doxygenfunction:: hipblasDtrmvBatched
    :outline:
.. doxygenfunction:: hipblasCtrmvBatched
    :outline:
.. doxygenfunction:: hipblasZtrmvBatched

.. doxygenfunction:: hipblasStrmvStridedBatched
    :outline:
.. doxygenfunction:: hipblasDtrmvStridedBatched
    :outline:
.. doxygenfunction:: hipblasCtrmvStridedBatched
    :outline:
.. doxygenfunction:: hipblasZtrmvStridedBatched

hipblasXtrsv + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasStrsv
    :outline:
.. doxygenfunction:: hipblasDtrsv
    :outline:
.. doxygenfunction:: hipblasCtrsv
    :outline:
.. doxygenfunction:: hipblasZtrsv

.. doxygenfunction:: hipblasStrsvBatched
    :outline:
.. doxygenfunction:: hipblasDtrsvBatched
    :outline:
.. doxygenfunction:: hipblasCtrsvBatched
    :outline:
.. doxygenfunction:: hipblasZtrsvBatched

.. doxygenfunction:: hipblasStrsvStridedBatched
    :outline:
.. doxygenfunction:: hipblasDtrsvStridedBatched
    :outline:
.. doxygenfunction:: hipblasCtrsvStridedBatched
    :outline:
.. doxygenfunction:: hipblasZtrsvStridedBatched

Level 3 BLAS
============
.. contents:: List of Level-3 BLAS Functions
   :local:
   :backlinks: top


hipblasXgemm + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasHgemm
    :outline:
.. doxygenfunction:: hipblasSgemm
    :outline:
.. doxygenfunction:: hipblasDgemm
    :outline:
.. doxygenfunction:: hipblasCgemm
    :outline:
.. doxygenfunction:: hipblasZgemm

.. doxygenfunction:: hipblasHgemmBatched
    :outline:
.. doxygenfunction:: hipblasSgemmBatched
    :outline:
.. doxygenfunction:: hipblasDgemmBatched
    :outline:
.. doxygenfunction:: hipblasCgemmBatched
    :outline:
.. doxygenfunction:: hipblasZgemmBatched

.. doxygenfunction:: hipblasHgemmStridedBatched
    :outline:
.. doxygenfunction:: hipblasSgemmStridedBatched
    :outline:
.. doxygenfunction:: hipblasDgemmStridedBatched
    :outline:
.. doxygenfunction:: hipblasCgemmStridedBatched
    :outline:
.. doxygenfunction:: hipblasZgemmStridedBatched

hipblasXherk + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasCherk
    :outline:
.. doxygenfunction:: hipblasZherk

.. doxygenfunction:: hipblasCherkBatched
    :outline:
.. doxygenfunction:: hipblasZherkBatched

.. doxygenfunction:: hipblasCherkStridedBatched
    :outline:
.. doxygenfunction:: hipblasZherkStridedBatched

hipblasXherkx + Batched, StridedBatched
-----------------------------------------
.. doxygenfunction:: hipblasCherkx
    :outline:
.. doxygenfunction:: hipblasZherkx

.. doxygenfunction:: hipblasCherkxBatched
    :outline:
.. doxygenfunction:: hipblasZherkxBatched

.. doxygenfunction:: hipblasCherkxStridedBatched
    :outline:
.. doxygenfunction:: hipblasZherkxStridedBatched

hipblasXher2k + Batched, StridedBatched
-----------------------------------------
.. doxygenfunction:: hipblasCher2k
    :outline:
.. doxygenfunction:: hipblasZher2k

.. doxygenfunction:: hipblasCher2kBatched
    :outline:
.. doxygenfunction:: hipblasZher2kBatched

.. doxygenfunction:: hipblasCher2kStridedBatched
    :outline:
.. doxygenfunction:: hipblasZher2kStridedBatched


hipblasXsymm + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasSsymm
    :outline:
.. doxygenfunction:: hipblasDsymm
    :outline:
.. doxygenfunction:: hipblasCsymm
    :outline:
.. doxygenfunction:: hipblasZsymm

.. doxygenfunction:: hipblasSsymmBatched
    :outline:
.. doxygenfunction:: hipblasDsymmBatched
    :outline:
.. doxygenfunction:: hipblasCsymmBatched
    :outline:
.. doxygenfunction:: hipblasZsymmBatched

.. doxygenfunction:: hipblasSsymmStridedBatched
    :outline:
.. doxygenfunction:: hipblasDsymmStridedBatched
    :outline:
.. doxygenfunction:: hipblasCsymmStridedBatched
    :outline:
.. doxygenfunction:: hipblasZsymmStridedBatched

hipblasXsyrk + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasSsyrk
    :outline:
.. doxygenfunction:: hipblasDsyrk
    :outline:
.. doxygenfunction:: hipblasCsyrk
    :outline:
.. doxygenfunction:: hipblasZsyrk

.. doxygenfunction:: hipblasSsyrkBatched
    :outline:
.. doxygenfunction:: hipblasDsyrkBatched
    :outline:
.. doxygenfunction:: hipblasCsyrkBatched
    :outline:
.. doxygenfunction:: hipblasZsyrkBatched

.. doxygenfunction:: hipblasSsyrkStridedBatched
    :outline:
.. doxygenfunction:: hipblasDsyrkStridedBatched
    :outline:
.. doxygenfunction:: hipblasCsyrkStridedBatched
    :outline:
.. doxygenfunction:: hipblasZsyrkStridedBatched

hipblasXsyr2k + Batched, StridedBatched
-----------------------------------------
.. doxygenfunction:: hipblasSsyr2k
    :outline:
.. doxygenfunction:: hipblasDsyr2k
    :outline:
.. doxygenfunction:: hipblasCsyr2k
    :outline:
.. doxygenfunction:: hipblasZsyr2k

.. doxygenfunction:: hipblasSsyr2kBatched
    :outline:
.. doxygenfunction:: hipblasDsyr2kBatched
    :outline:
.. doxygenfunction:: hipblasCsyr2kBatched
    :outline:
.. doxygenfunction:: hipblasZsyr2kBatched

.. doxygenfunction:: hipblasSsyr2kStridedBatched
    :outline:
.. doxygenfunction:: hipblasDsyr2kStridedBatched
    :outline:
.. doxygenfunction:: hipblasCsyr2kStridedBatched
    :outline:
.. doxygenfunction:: hipblasZsyr2kStridedBatched

hipblasXsyrkx + Batched, StridedBatched
-----------------------------------------
.. doxygenfunction:: hipblasSsyrkx
    :outline:
.. doxygenfunction:: hipblasDsyrkx
    :outline:
.. doxygenfunction:: hipblasCsyrkx
    :outline:
.. doxygenfunction:: hipblasZsyrkx

.. doxygenfunction:: hipblasSsyrkxBatched
    :outline:
.. doxygenfunction:: hipblasDsyrkxBatched
    :outline:
.. doxygenfunction:: hipblasCsyrkxBatched
    :outline:
.. doxygenfunction:: hipblasZsyrkxBatched

.. doxygenfunction:: hipblasSsyrkxStridedBatched
    :outline:
.. doxygenfunction:: hipblasDsyrkxStridedBatched
    :outline:
.. doxygenfunction:: hipblasCsyrkxStridedBatched
    :outline:
.. doxygenfunction:: hipblasZsyrkxStridedBatched

hipblasXgeam + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasSgeam
    :outline:
.. doxygenfunction:: hipblasDgeam
    :outline:
.. doxygenfunction:: hipblasCgeam
    :outline:
.. doxygenfunction:: hipblasZgeam

.. doxygenfunction:: hipblasSgeamBatched
    :outline:
.. doxygenfunction:: hipblasDgeamBatched
    :outline:
.. doxygenfunction:: hipblasCgeamBatched
    :outline:
.. doxygenfunction:: hipblasZgeamBatched

.. doxygenfunction:: hipblasSgeamStridedBatched
    :outline:
.. doxygenfunction:: hipblasDgeamStridedBatched
    :outline:
.. doxygenfunction:: hipblasCgeamStridedBatched
    :outline:
.. doxygenfunction:: hipblasZgeamStridedBatched

hipblasXhemm + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasChemm
    :outline:
.. doxygenfunction:: hipblasZhemm

.. doxygenfunction:: hipblasChemmBatched
    :outline:
.. doxygenfunction:: hipblasZhemmBatched

.. doxygenfunction:: hipblasChemmStridedBatched
    :outline:
.. doxygenfunction:: hipblasZhemmStridedBatched

hipblasXtrmm + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasStrmm
    :outline:
.. doxygenfunction:: hipblasDtrmm
    :outline:
.. doxygenfunction:: hipblasCtrmm
    :outline:
.. doxygenfunction:: hipblasZtrmm

.. doxygenfunction:: hipblasStrmmBatched
    :outline:
.. doxygenfunction:: hipblasDtrmmBatched
    :outline:
.. doxygenfunction:: hipblasCtrmmBatched
    :outline:
.. doxygenfunction:: hipblasZtrmmBatched

.. doxygenfunction:: hipblasStrmmStridedBatched
    :outline:
.. doxygenfunction:: hipblasDtrmmStridedBatched
    :outline:
.. doxygenfunction:: hipblasCtrmmStridedBatched
    :outline:
.. doxygenfunction:: hipblasZtrmmStridedBatched

hipblasXtrsm + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasStrsm
    :outline:
.. doxygenfunction:: hipblasDtrsm
    :outline:
.. doxygenfunction:: hipblasCtrsm
    :outline:
.. doxygenfunction:: hipblasZtrsm

.. doxygenfunction:: hipblasStrsmBatched
    :outline:
.. doxygenfunction:: hipblasDtrsmBatched
    :outline:
.. doxygenfunction:: hipblasCtrsmBatched
    :outline:
.. doxygenfunction:: hipblasZtrsmBatched

.. doxygenfunction:: hipblasStrsmStridedBatched
    :outline:
.. doxygenfunction:: hipblasDtrsmStridedBatched
    :outline:
.. doxygenfunction:: hipblasCtrsmStridedBatched
    :outline:
.. doxygenfunction:: hipblasZtrsmStridedBatched

hipblasXtrtri + Batched, StridedBatched
-----------------------------------------
.. doxygenfunction:: hipblasStrtri
    :outline:
.. doxygenfunction:: hipblasDtrtri
    :outline:
.. doxygenfunction:: hipblasCtrtri
    :outline:
.. doxygenfunction:: hipblasZtrtri

.. doxygenfunction:: hipblasStrtriBatched
    :outline:
.. doxygenfunction:: hipblasDtrtriBatched
    :outline:
.. doxygenfunction:: hipblasCtrtriBatched
    :outline:
.. doxygenfunction:: hipblasZtrtriBatched

.. doxygenfunction:: hipblasStrtriStridedBatched
    :outline:
.. doxygenfunction:: hipblasDtrtriStridedBatched
    :outline:
.. doxygenfunction:: hipblasCtrtriStridedBatched
    :outline:
.. doxygenfunction:: hipblasZtrtriStridedBatched

hipblasXdgmm + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasSdgmm
    :outline:
.. doxygenfunction:: hipblasDdgmm
    :outline:
.. doxygenfunction:: hipblasCdgmm
    :outline:
.. doxygenfunction:: hipblasZdgmm

.. doxygenfunction:: hipblasSdgmmBatched
    :outline:
.. doxygenfunction:: hipblasDdgmmBatched
    :outline:
.. doxygenfunction:: hipblasCdgmmBatched
    :outline:
.. doxygenfunction:: hipblasZdgmmBatched

.. doxygenfunction:: hipblasSdgmmStridedBatched
    :outline:
.. doxygenfunction:: hipblasDdgmmStridedBatched
    :outline:
.. doxygenfunction:: hipblasCdgmmStridedBatched
    :outline:
.. doxygenfunction:: hipblasZdgmmStridedBatched

SOLVER API
===========
.. contents:: List of SOLVER APIs
   :local:
   :backlinks: top


hipblasXgetrf + Batched, stridedBatched
----------------------------------------
.. doxygenfunction:: hipblasSgetrf
    :outline:
.. doxygenfunction:: hipblasDgetrf
    :outline:
.. doxygenfunction:: hipblasCgetrf
    :outline:
.. doxygenfunction:: hipblasZgetrf

.. doxygenfunction:: hipblasSgetrfBatched
    :outline:
.. doxygenfunction:: hipblasDgetrfBatched
    :outline:
.. doxygenfunction:: hipblasCgetrfBatched
    :outline:
.. doxygenfunction:: hipblasZgetrfBatched

.. doxygenfunction:: hipblasSgetrfStridedBatched
    :outline:
.. doxygenfunction:: hipblasDgetrfStridedBatched
    :outline:
.. doxygenfunction:: hipblasCgetrfStridedBatched
    :outline:
.. doxygenfunction:: hipblasZgetrfStridedBatched


hipblasXgetrs + Batched, stridedBatched
----------------------------------------
.. doxygenfunction:: hipblasSgetrs
    :outline:
.. doxygenfunction:: hipblasDgetrs
    :outline:
.. doxygenfunction:: hipblasCgetrs
    :outline:
.. doxygenfunction:: hipblasZgetrs

.. doxygenfunction:: hipblasSgetrsBatched
    :outline:
.. doxygenfunction:: hipblasDgetrsBatched
    :outline:
.. doxygenfunction:: hipblasCgetrsBatched
    :outline:
.. doxygenfunction:: hipblasZgetrsBatched

.. doxygenfunction:: hipblasSgetrsStridedBatched
    :outline:
.. doxygenfunction:: hipblasDgetrsStridedBatched
    :outline:
.. doxygenfunction:: hipblasCgetrsStridedBatched
    :outline:
.. doxygenfunction:: hipblasZgetrsStridedBatched

hipblasXgetri + Batched, stridedBatched
----------------------------------------

.. doxygenfunction:: hipblasSgetriBatched
    :outline:
.. doxygenfunction:: hipblasDgetriBatched
    :outline:
.. doxygenfunction:: hipblasCgetriBatched
    :outline:
.. doxygenfunction:: hipblasZgetriBatched

hipblasXgeqrf + Batched, stridedBatched
----------------------------------------
.. doxygenfunction:: hipblasSgeqrf
    :outline:
.. doxygenfunction:: hipblasDgeqrf
    :outline:
.. doxygenfunction:: hipblasCgeqrf
    :outline:
.. doxygenfunction:: hipblasZgeqrf

.. doxygenfunction:: hipblasSgeqrfBatched
    :outline:
.. doxygenfunction:: hipblasDgeqrfBatched
    :outline:
.. doxygenfunction:: hipblasCgeqrfBatched
    :outline:
.. doxygenfunction:: hipblasZgeqrfBatched

.. doxygenfunction:: hipblasSgeqrfStridedBatched
    :outline:
.. doxygenfunction:: hipblasDgeqrfStridedBatched
    :outline:
.. doxygenfunction:: hipblasCgeqrfStridedBatched
    :outline:
.. doxygenfunction:: hipblasZgeqrfStridedBatched

hipblasXgels + Batched, StridedBatched
----------------------------------------
.. doxygenfunction:: hipblasSgels
    :outline:
.. doxygenfunction:: hipblasDgels
    :outline:
.. doxygenfunction:: hipblasCgels
    :outline:
.. doxygenfunction:: hipblasZgels

.. doxygenfunction:: hipblasSgelsBatched
    :outline:
.. doxygenfunction:: hipblasDgelsBatched
    :outline:
.. doxygenfunction:: hipblasCgelsBatched
    :outline:
.. doxygenfunction:: hipblasZgelsBatched

.. doxygenfunction:: hipblasSgelsStridedBatched
    :outline:
.. doxygenfunction:: hipblasDgelsStridedBatched
    :outline:
.. doxygenfunction:: hipblasCgelsStridedBatched
    :outline:
.. doxygenfunction:: hipblasZgelsStridedBatched

BLAS Extensions
===============
.. contents:: List of BLAS Extension Functions
   :local:
   :backlinks: top


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
--------------
.. doxygenfunction:: hipblasCreate

hipblasDestroy
---------------
.. doxygenfunction:: hipblasDestroy

hipblasSetStream
-----------------
.. doxygenfunction:: hipblasSetStream

hipblasGetStream
------------------
.. doxygenfunction:: hipblasGetStream

hipblasSetPointerMode
----------------------
.. doxygenfunction:: hipblasSetPointerMode

hipblasGetPointerMode
----------------------
.. doxygenfunction:: hipblasGetPointerMode

hipblasSetVector
----------------
.. doxygenfunction:: hipblasSetVector

hipblasGetVector
-----------------
.. doxygenfunction:: hipblasGetVector

hipblasSetMatrix
-----------------
.. doxygenfunction:: hipblasSetMatrix

hipblasGetMatrix
------------------
.. doxygenfunction:: hipblasGetMatrix

hipblasSetVectorAsync
----------------------
.. doxygenfunction:: hipblasSetVectorAsync

hipblasGetVectorAsync
----------------------
.. doxygenfunction:: hipblasGetVectorAsync

hipblasSetMatrixAsync
-----------------------
.. doxygenfunction:: hipblasSetMatrixAsync

hipblasGetMatrixAsync
---------------------
.. doxygenfunction:: hipblasGetMatrixAsync

hipblasSetAtomicsMode
----------------------
.. doxygenfunction:: hipblasSetAtomicsMode

hipblasGetAtomicsMode
----------------------
.. doxygenfunction:: hipblasGetAtomicsMode

hipblasStatusToString
----------------------
.. doxygenfunction:: hipblasStatusToString
