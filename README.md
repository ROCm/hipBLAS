# hipBLAS
hipBLAS is a BLAS marshalling library, with multiple supported backends.  It sits between the application and a 'worker' BLAS library, marshalling inputs into the backend library and marshalling results back to the application.  hipBLAS exports an interface that does not require the client to change, regardless of the chosen backend.  Currently, hipBLAS supports [rocBLAS](https://github.com/ROCmSoftwarePlatform/rocBLAS) and [cuBLAS](https://developer.nvidia.com/cublas) as backends.

## Installing pre-built packages
Download pre-built packages either from [ROCm's package servers](https://rocm.github.io/install.html#installing-from-amd-rocm-repositories) or by clicking the github releases tab and manually downloading, which could be newer.  Release notes are available for each release on the releases tab.
* `sudo apt update && sudo apt install hipblas`

## Quickstart hipBLAS build

#### Bash helper build script (Ubuntu only)
The root of this repository has a helper bash script `install.sh` to build and install hipBLAS on Ubuntu with a single command.  It does not take a lot of options and hard-codes configuration that can be specified through invoking cmake directly, but it's a great way to get started quickly and can serve as an example of how to build/install.  A few commands in the script need sudo access, so it may prompt you for a password.
*  `./install -h` -- shows help
*  `./install -id` -- build library, build dependencies and install (-d flag only needs to be passed once on a system)

## Manual build (all supported platforms)
If you use a distro other than Ubuntu, or would like more control over the build process, the [hipblas build wiki](https://github.com/ROCmSoftwarePlatform/hipBLAS/wiki/Build) has helpful information on how to configure cmake and manually build.

### Functions supported
A list of [exported functions](https://github.com/ROCmSoftwarePlatform/hipBLAS/wiki/Exported-functions) from hipblas can be found on the wiki

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

hipBLAS assumes matrices A and vectors x, y are allocated in GPU memory space filled with data.  Users are
responsible for copying data from/to the host and device memory.
