# **hipBLAS**

**hipBLAS** is a **BLAS** marshalling library, with multiple supported backends.  It sits between the application and a 'worker' BLAS library, marshalling inputs into the backend library and marshalling results back to the application.  **hipBLAS** exports an interface that does not require the client to change, regardless of the chosen backend.  Currently, **hipBLAS** supports **rocBLAS** and **cuBLAS** as backends.


| Acronym  | Expansion                                                   |
| :------- | :---------------------------------------------------------- |
| **BLAS** | **B**asic **L**inear **A**lgebra **S**ubprograms            |
| **ROCm** | **R**adeon **O**pen E**C**osyste**m**                       |
| **HIP**  | **H**eterogeneous-Compute **I**nterface for **P**ortability |

## Documentation

For a detailed description of the **hipBLAS** library, its implemented routines, the installation process and user guide, see the [**hipBLAS** Documentation](https://hipblas.readthedocs.io/en/latest/).

hipBLAS requires either **rocBLAS** + **rocSOLVER** or **cuBLAS** APIs for BLAS implementation. For more information dependent **roc*** libraries see [rocBLAS documentation](https://rocblas.readthedocs.io/en/latest/), and [rocSolver documentation](https://rocsolver.readthedocs.io/en/latest/).

## Quickstart build

To download the **hipBLAS** source code, use the below command to clone the repository

```bash
    git clone https://github.com/ROCmSoftwarePlatform/hipBLAS.git
```

**hipBLAS** requires specific version of **rocBLAS** & **rocSOLVER** to be installed on the system. The required **rocBLAS** and **rocSOLVER** versions to build **hipBLAS** is provided [here](https://github.com/ROCmSoftwarePlatform/hipBLAS/blob/develop/library/CMakeLists.txt).

Once the dependent libraries are installed, the following command will build hipBLAS and install to `/opt/rocm/hipblas`:

```bash
    cd hipblas
    ./install.sh -i
```

## hipBLAS interface examples

The hipBLAS interface is compatible with rocBLAS and cuBLAS-v2 APIs.  Porting a CUDA application which originally calls the cuBLAS API to an application calling hipBLAS API should be relatively straightforward. For example, the hipBLAS SGEMV interface is

### GEMV API

```c
hipblasStatus_t
hipblasSgemv( hipblasHandle_t handle,
              hipblasOperation_t trans,
              int m, int n, const float *alpha,
              const float *A, int lda,
              const float *x, int incx, const float *beta,
              float *y, int incy );
```

### Batched and strided GEMM API

hipBLAS GEMM can process matrices in batches with regular strides.  There are several permutations of these API's, the
following is an example that takes everything

```c
hipblasStatus_t
hipblasSgemmStridedBatched( hipblasHandle_t handle,
              hipblasOperation_t transa, hipblasOperation_t transb,
              int m, int n, int k, const float *alpha,
              const float *A, int lda, long long bsa,
              const float *B, int ldb, long long bsb, const float *beta,
              float *C, int ldc, long long bsc,
              int batchCount);
```

hipBLAS assumes matrices A and vectors x, y are allocated in GPU memory space filled with data.  Users are responsible for copying data from/to the host and device memory.

## Supported functionality

For a complete list of all supported functions, see the [hipBLAS user guide](https://hipblas.readthedocs.io/en/latest/usermanual.html) and [hipBLAS functions](https://rocblas.readthedocs.io/en/latest/functions.html#hipblas-functions).
