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









































