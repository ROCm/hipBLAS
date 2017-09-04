# hipBLAS
hipBLAS is a BLAS marshalling library, with multiple supported backends.  It sits between the application and a 'worker' BLAS library, marshalling inputs into the backend library and marshalling results back to the application.  hipBLAS exports an interface that does not require the client to change, regardless of the chosen backend.  Currently, hipBLAS supports [rocBLAS](https://github.com/ROCmSoftwarePlatform/rocBLAS) and [cuBLAS](https://developer.nvidia.com/cublas) as backends.

## Building hipBLAS
#### Bash helper build script (Ubuntu only)
The root of this repository has a helper bash script `install.sh` to build and install hipBLAS on Ubuntu with a single command.  It does not take a lot of options and hard-codes configuration that can be specified through invoking cmake directly, but it's a great way to get started quickly and can serve as an example of how to build/install.  A few commands in the script need sudo access, so it may prompt you for a password.
*  `./install -h` -- shows help
*  `./install -id` -- common invocation (installs dependencies, builds and installs library)

### Manual build (all supported platforms)
The build infrastructure for hipBLAS is based on [Cmake](https://cmake.org/) v3.5.  This is the version of cmake available on ROCm supported platforms.  Examples of installing cmake:
* Ubuntu: `sudo apt install cmake-qt-gui`
* Fedora: `sudo dnf install cmake-gui`

### Library
If building the library on a ROCm platform, hipBLAS depends on a rocBLAS installation to be found.  If building the library on a CUDA platform, hipBLAS depends on cuBLAS to be found.  If cmake cannot find these dependencies automatically, the user can specify additional search locations through the CMAKE\_PREFIX\_PATH cmake configuration variable

#### Configure and build steps
```bash
mkdir -p [HIPBLAS_BUILD_DIR]/release
cd [HIPBLAS_BUILD_DIR]/release
# Default install location is in /opt/rocm, define -DCMAKE_INSTALL_PREFIX=<path> to specify other
# Default build config is 'Release', define -DCMAKE_BUILD_TYPE=<config> to specify other
CXX=/opt/rocm/bin/hcc ccmake [HIPBLAS_SOURCE]
make -j$(nproc)
sudo make install # sudo required if installing into system directory such as /opt/rocm
```

### hipBLAS clients
The repository contains source for a unit testing framework, which can be found in the clients subdir.

### Dependencies (only necessary for hipBLAS clients)
The hipBLAS unit tester introduces the following dependencies:
1.  [boost](http://www.boost.org/)
2.  [lapack](https://github.com/Reference-LAPACK/lapack-release)
  * lapack itself brings a dependency on a fortran compiler
3.  [googletest](https://github.com/google/googletest)

Linux distros typically have an easy installation mechanism for boost through the native package manager.

* Ubuntu: `sudo apt install libboost-program-options-dev`
* Fedora: `sudo dnf install boost-program-options`

Unfortunately, googletest and lapack are not as easy to install.  Many distros do not provide a googletest package with pre-compiled libraries, and the lapack packages do not have the necessary cmake config files for cmake to configure linking the cblas library.  hipBLAS provide a cmake script that builds the above dependencies from source.  This is an optional step; users can provide their own builds of these dependencies and help cmake find them by setting the CMAKE\_PREFIX\_PATH definition.  The following is a sequence of steps to build dependencies and install them to the cmake default /usr/local.

#### (optional, one time only)
```bash
mkdir -p [ROCBLAS_BUILD_DIR]/release/deps
cd [ROCBLAS_BUILD_DIR]/release/deps
ccmake -DBUILD_BOOST=OFF [ROCBLAS_SOURCE]/deps   # assuming boost is installed through package manager as above
make -j$(nproc) install
```

Once dependencies are available on the system, it is possible to configure the clients to build.  This requires a few extra cmake flags to the library cmake configure script. If the dependencies are not installed into system defaults (like /usr/local ), you should pass the CMAKE\_PREFIX\_PATH to cmake to help find them.
* `-DCMAKE_PREFIX_PATH="<semicolon separated paths>"`
```bash
# Default install location is in /opt/rocm, use -DCMAKE_INSTALL_PREFIX=<path> to specify other
CXX=/opt/rocm/bin/hcc ccmake -DBUILD_CLIENTS_TESTS=ON -DBUILD_CLIENTS_BENCHMARKS=ON [ROCBLAS_SOURCE]
make -j$(nproc)
sudo make install   # sudo required if installing into system directory such as /opt/rocm
```
