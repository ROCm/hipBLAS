************
Introduction
************

hipBLAS is a BLAS marshaling library with multiple supported backends. It sits between the application and a 'worker' BLAS library, marshalling inputs into the backend library and marshalling results back to the application.
hipBLAS exports an interface that does not require the client to change, regardless of the chosen backend. Currently, it supports `rocBLAS <https://github.com/ROCmSoftwarePlatform/rocBLAS>`_ and `cuBLAS <https://developer.nvidia.com/cublas>`_ as backends.

======== =========
Acronym  Expansion
======== =========
**BLAS**    **B**\ asic **L**\ inear **A**\ lgebra **S**\ ubprograms
**ROCm**    **R**\ adeon **O**\ pen E\ **C**\ osyste\ **m**
**HIP**     **H**\ eterogeneous-Compute **I**\ nterface for **P**\ ortability
======== =========

Currently implemented functionality
===================================

The following tables summarizes teh BLAS, SOLVER functionalities implemented for different supported precisions in hipBLAS's latest release.

hipBLAS auxiliary functions
---------------------------

.. csv-table:: Auxiliary Functions
    :header: "Function"

    :ref:`hipblasCreate`
    :ref:`hipblasDestroy`
    :ref:`hipblasSetStream`
    :ref:`hipblasGetStream`
    :ref:`hipblasSetPointerMode`
    :ref:`hipblasGetPointerMode`
    :ref:`hipblasSetVector`
    :ref:`hipblasGetVector`
    :ref:`hipblasSetMatrix`
    :ref:`hipblasGetMatrix`
    :ref:`hipblasSetVectorAsync`
    :ref:`hipblasGetVectorAsync`
    :ref:`hipblasSetMatrixAsync`
    :ref:`hipblasGetMatrixAsync`
    :ref:`hipblasSetAtomicsMode`
    :ref:`hipblasGetAtomicsMode`

hipBLAS includes the following Level 1, 2, and 3 functions
----------------------------------------------------------

.. csv-table:: Level - 1
    :header: "Function", "single", "double", "single complex", "double complex", "half" , batched (rocBLAS)

    :ref:`hipblasXscal`,   	x,	x,	x,	x,	 ,	x
    :ref:`hipblasXcopy`,   	x,	x,	x,	x,	 ,	x
    :ref:`hipblasXdot` ,   	x,	x,	x,	x,	x,
    :ref:`hipblasXaxpy`,   	x,	x,	x,	x,	x,	x
    :ref:`hipblasXasum`,   	x,	x,	x,	x,	 ,	x
    :ref:`hipblasiXama`,   	x,	x,	x,	x,	 ,	x
    :ref:`hipblasiXami`,   	x,	x,	x,	x,	 ,	x
    :ref:`hipblasXnrm2`,   	x,	x,	x,	x,	 ,	x
    :ref:`hipblasXrot` ,   	x,	x,	x,	 ,	x,
    :ref:`hipblasXrotg`,   	x,	x,	x,	x,	 ,	x
    :ref:`hipblasXrotm`,   	x,	x,	 ,	 ,	 ,	x
    :ref:`hipblasXrotmg`,  	x,	x,	 ,	 ,	 ,	x
    :ref:`hipblasXswap`,   	x,	x,	x,	x,	 ,	x

.. csv-table:: Level - 2
    :header: "Function", "single", "double", "single complex", "double complex", "half" , batched (rocBLAS)

    :ref:`hipblasXgbmv`,   x,	x,	x,	x,	 ,	x,
    :ref:`hipblasXgemv`,   x,	x,	x,	x,	 ,	x,
    :ref:`hipblasXger`,    x,	x,	x,	x,	 ,	x,
    :ref:`hipblasXhbmv`,   	,	x,	x,	 ,	x,   ,
    :ref:`hipblasXhemv`,   	,	x,	x,	 ,	x,   ,
    :ref:`hipblasXher`,    	,	x,	x,	 ,	x,   ,
    :ref:`hipblasXher2`,	,	 ,	 ,	x,	x,	 ,	x
    :ref:`hipblasXhpmv`,	,	 ,	 ,	x,	x,	 ,	x
    :ref:`hipblasXhpr`,	   	,	x,	x,	 ,	x,   ,
    :ref:`hipblasXhpr2`,	,	 ,	 ,	x,	x,	 ,	x
    :ref:`hipblasXsbmv`,	,	x,	x,	 ,	 ,	 ,	x
    :ref:`hipblasXspmv`,	,	x,	x,	 ,	 ,	 ,	x
    :ref:`hipblasXspr`,    x,	x,	x,	x,	 ,	x,
    :ref:`hipblasXspr2`,   x,	x,	 ,	 ,	 ,	x,
    :ref:`hipblasXsymv`,   x,	x,	x,	x,	 ,	x,
    :ref:`hipblasXsyr`,    x,	x,	x,	x,	 ,	x,
    :ref:`hipblasXsyr2`,   x,	x,	x,	x,	 ,	x,
    :ref:`hipblasXtbmv`,   x,	x,	x,	x,	 ,	x,
    :ref:`hipblasXtbsv`,   x,	x,	x,	x,	 ,	x,
    :ref:`hipblasXtpmv`,   x,	x,	x,	x,	 ,	x,
    :ref:`hipblasXtpsv`,   x,	x,	x,	x,	 ,	x,
    :ref:`hipblasXtrmv`,   x,	x,	x,	x,	 ,	x,
    :ref:`hipblasXtrsv`,   x,	x,	x,	x,	 ,	x,

.. csv-table:: Level - 3
    :header: "Function", "single", "double", "single complex", "double complex", "half" , batched (rocBLAS)

    :ref:`hipblasXherk`     ,    ,	x,	x,	 ,	x,
    :ref:`hipblasXherkx`    ,    ,	x,	x,	 ,	x,
    :ref:`hipblasXher2k`    ,    ,	x,	x,	 ,	x,
    :ref:`hipblasXsymm`     ,   x,	x,	x,	x,	 ,	x
    :ref:`hipblasXsyrk`     ,   x,	x,	x,	x,	 ,	x
    :ref:`hipblasXsyr2k`    ,   x,	x,	x,	x,	 ,	x
    :ref:`hipblasXsyrkx`    ,   x,	x,	x,	x,	 ,	x
    :ref:`hipblasXhemm`     ,    ,	x,	x,	 ,	x,
    :ref:`hipblasXtrmm`     ,   x,	x,	x,	x,	 ,	x
    :ref:`hipblasXtrsm`     ,   x,	x,	x,	x,	 ,	x
    :ref:`hipblasXtrtri`    ,   x,	x,	x,	x,	 ,	x
    :ref:`hipblasXdgmm`     ,   x,	x,	x,	x,	 ,	x
    :ref:`hipblasXgeam`     ,   x,	x,	x,	x,	 ,	x
    :ref:`hipblasXgemm`     ,   x,	x,	x,	x,	x,	x

.. csv-table:: Solver Functions
    :header: "Function", "single", "double", "single complex", "double complex", "half" , batched (rocBLAS)

    :ref:`hipblasXgetrf`,	    x,	x,	x,	x,	,	x
    :ref:`hipblasXgetrs`,	    x,	x,	x,	x,	,	x
    :ref:`hipblasXgetriBatched`,x,	x,	x,	x,	,   x
    :ref:`hipblasXgeqrf`,	    x,	x,	x,	x,	,	x


.. csv-table:: BLASEx
    :header: "Function"

    :ref:`hipblasGemmEx`
    :ref:`hipblasTrsmEx`
    :ref:`hipblasAxpyEx`
    :ref:`hipblasDotEx`
    :ref:`hipblasNrm2Ex`
    :ref:`hipblasRotEx`
    :ref:`hipblasScalEx`









































