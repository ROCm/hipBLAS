************
Introduction
************
AMD **ROCm** has two classification of libraries,

- **roc**\* : AMD GPU Libraries, written in `HIP <https://docs.amd.com/bundle/AMD_HIP_Programming_Guide/page/Introduction.html>`_.
- **hip**\* : AMD CPU library that is a thin interface to either AMD **roc*** or Nvidia **cu*** libraries.

Users targetting both CUDA and AMD devices must use the **hip*** libraries.

hipBLAS is a BLAS marshaling library with multiple supported backends. It sits between the application and a 'worker' BLAS library, marshalling inputs into the backend library and marshalling results back to the application.
hipBLAS exports an interface that does not require the client to change, regardless of the chosen backend. Currently, it supports `rocBLAS <https://github.com/ROCmSoftwarePlatform/rocBLAS>`_ and `cuBLAS <https://developer.nvidia.com/cublas>`_ as backends.
