#!/usr/bin/env bash

# ########################################################################
# Copyright (C) 2016-2022 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# ########################################################################

/bin/ln -fs ../../.githooks/pre-commit "$(dirname "$0")/.git/hooks/"


# #################################################
# helper functions
# #################################################
function display_help()
{
cat <<EOF

  hipBLAS library build & installation helper script.

  Usage:
    $0 (build hipblas and put library files at ./build/hipblas-install)
    $0 <options> (modify default behavior according to the following flags)

  Options:
    --address-sanitizer           Build with address sanitizer enabled. Uses hipcc as compiler.

    -b, --rocblas <version>       Specify rocblas version (e.g. 2.42.0).

    -c, --clients                 Build the library clients benchmark and gtest.
                                  (Generated binaries will be located at builddir/clients/staging)

    --cuda, --use-cuda            Build library for CUDA backend.

    --cudapath <cudadir>          Specify path of CUDA install (default /usr/local/cuda).

    --cmake-arg                   Forward the given argument to CMake when configuring the build.

    --compiler </compier/path>    Specify path to host compiler. (e.g. /opt/bin/hipcc)

    --custom-target <target>      Specify custom target to link the library against (eg. host, device).

    --codecoverage                Build with code coverage profiling enabled, excluding release mode.

    -d, --dependencies            Build and install external dependencies. Dependecies are to be installed in /usr/local.
                                  This should be done only once (this does not install rocBLAS, rocSolver, or cuda).

    --installcuda                 Install cuda package.

    --installcudaversion <version> Used with --installcuda, optionally specify cuda version to install.

    -g, --debug                   Build in Debug mode, equivalent to set CMAKE_BUILD_TYPE=Debug. (Default build type is Release)

    -h, --help                    Print this help message.

    --hip-clang                   Build library using the hip-clang compiler.

    -i, -install                  Generate and install library package after build.

    -k,  --relwithdebinfo         Build in release debug mode, equivalent to set CMAKE_BUILD_TYPE=RelWithDebInfo.(Default build type is Release)

    -n, --no-solver               Build hipLBAS library without rocSOLVER dependency

    --no-hip-clang                Build library without using hip-clang compiler.

    -p, --cmakepp                 To add CMAKE_PREFIX_PATH

    -r, --relocatable             Create a package to support relocatable ROCm

    --rocblas-path <blasdir>      Specify path to an existing rocBLAS install directory (e.g. /src/rocBLAS/build/release/rocblas-install).

    --rocsolver-path <solverdir>  Specify path to an existing rocSOLVER install directory (e.g. /src/rocSOLVER/build/release/rocsolver-install).

    -s, --static                  Build hipblas as a static library (hipblas must be built statically when the used companion rocblas is also static).

    -v, --rocm-dev <version>      Specify specific rocm-dev version. (e.g. 4.5.0)

    --rm-legacy-include-dir       Remove legacy include dir Packaging added for file/folder reorg backward compatibility.
EOF
}

# This function is helpful for dockerfiles that do not have sudo installed, but the default user is root
# true is a system command that completes successfully, function returns success
# prereq: ${ID} must be defined before calling
supported_distro( )
{
  if [ -z ${ID+foo} ]; then
    printf "supported_distro(): \$ID must be set\n"
    exit 2
  fi

  case "${ID}" in
    ubuntu|centos|rhel|fedora|sles|opensuse-leap)
        true
        ;;
    *)  printf "This script is currently supported on Ubuntu, SLES, CentOS, RHEL and Fedora\n"
        exit 2
        ;;
  esac
}

check_exit_code( )
{
  if (( $1 != 0 )); then
    exit $1
  fi
}

# This function is helpful for dockerfiles that do not have sudo installed, but the default user is root
elevate_if_not_root( )
{
  local uid=$(id -u)

  if (( ${uid} )); then
    sudo $@
    check_exit_code "$?"
  else
    $@
    check_exit_code "$?"
  fi
}

# Take an array of packages as input, and install those packages with 'apt' if they are not already installed
install_apt_packages( )
{
  package_dependencies="$@"
  printf "\033[32mInstalling following packages from distro package manager: \033[33m${package_dependencies}\033[32m \033[0m\n"
  elevate_if_not_root apt-get -y --no-install-recommends install ${package_dependencies}
}

install_apt_packages_version( )
{
  package_dependencies=("$1")
  package_versions=("$2")
  for index in ${package_dependencies[*]}; do
    printf "\033[32mInstalling \033[33m${package_dependencies[$index]} version ${package_versions[$index]} from distro package manager \033[0m\n"
    elevate_if_not_root apt install -y --no-install-recommends ${package_dependencies[$index]}=${package_versions[$index]}
  done
}

# Take an array of packages as input, and install those packages with 'yum' if they are not already installed
install_yum_packages( )
{
  package_dependencies="$@"
  printf "\033[32mInstalling following packages from distro package manager: \033[33m${package_dependencies}\033[32m \033[0m\n"
  elevate_if_not_root yum -y --nogpgcheck install ${package_dependencies}
}

install_yum_packages_version( )
{
  package_dependencies=("$1")
  package_versions=("$2")
  for index in ${package_dependencies[*]}; do
    printf "\033[32mInstalling \033[33m${package_dependencies[$index]} version ${package_versions[$index]} from distro package manage
r \033[0m\n"
    elevate_if_not_root yum -y --nogpgcheck install ${package_dependencies[$index]}-${package_versions[$index]}
  done
}

# Take an array of packages as input, and install those packages with 'dnf' if they are not already installed
install_dnf_packages( )
{
  package_dependencies="$@"
  printf "\033[32mInstalling following packages from distro package manager: \033[33m${package_dependencies}\033[32m \033[0m\n"
  elevate_if_not_root dnf install -y ${package_dependencies}
}

install_dnf_packages_version( )
{
  package_dependencies=("$1")
  package_versions=("$2")
  for index in ${package_dependencies[*]}; do
    printf "\033[32mInstalling \033[33m${package_dependencies[$index]} version ${package_versions[$index]} from distro package manage
r \033[0m\n"
    elevate_if_not_root dnf install -y ${package_dependencies[$index]}-${package_versions[$index]}
  done
}

# Take an array of packages as input, and install those packages with 'zypper' if they are not already installed
install_zypper_packages( )
{
  package_dependencies="$@"
  printf "\033[32mInstalling following packages from distro package manager: \033[33m${package_dependencies}\033[32m \033[0m\n"
  elevate_if_not_root zypper install -y ${package_dependencies}
}

install_zypper_packages_version( )
{
  package_dependencies=("$1")
  package_versions=("$2")
  for index in ${package_dependencies[*]}; do
    printf "\033[32mInstalling \033[33m${package_dependencies[$index]} version ${package_versions[$index]} from distro package manage
r \033[0m\n"
    elevate_if_not_root zypper -n --no-gpg-checks install ${package_dependencies[$index]}-${package_versions[$index]}
  done
}

# Take an array of packages as input, and delegate the work to the appropriate distro installer
# prereq: ${ID} must be defined before calling
# prereq: ${build_clients} must be defined before calling
install_packages( )
{
  if [ -z ${ID+foo} ]; then
    printf "install_packages(): \$ID must be set\n"
    exit 2
  fi

  if [ -z ${build_clients+foo} ]; then
    printf "install_packages(): \$build_clients must be set\n"
    exit 2
  fi

  # dependencies needed for library and clients to build
  local library_dependencies_ubuntu=( "make" "pkg-config" )
  local library_dependencies_centos_rhel=( "epel-release" "make" "gcc-c++" "rpm-build" )
  local library_dependencies_centos_rhel_8=( "epel-release" "make" "gcc-c++" "rpm-build" )
  local library_dependencies_fedora=( "make" "gcc-c++" "libcxx-devel" "rpm-build" )
  local library_dependencies_sles=( "make" "gcc-c++" "libcxxtools9" "rpm-build" )

  if [[ "${build_cuda}" == true ]]; then
    # Ideally, this could be cuda-cublas-dev, but the package name has a version number in it
    library_dependencies_ubuntu+=( "" ) # removed, use --installcuda option to install cuda
  elif [[ "${build_hip_clang}" == false ]]; then
    # Custom rocm-dev installation
    if [[ -z ${custom_rocm_dev+foo} ]]; then
      # Install base rocm-dev package unless -v/--rocm-dev flag is passed
      library_dependencies_ubuntu+=( "rocm-dev" )
      library_dependencies_centos_rhel+=( "rocm-dev" )
      library_dependencies_centos_rhel_8=( "rocm-dev" )
      library_dependencies_fedora+=( "rocm-dev" )
      library_dependencies_sles+=( "rocm-dev" )
    else
      # Install rocm-specific rocm-dev package
      library_dependencies_ubuntu+=( "${custom_rocm_dev}" )
      library_dependencies_centos_rhel+=( "${custom_rocm_dev}" )
      library_dependencies_centos_rhel_8+=( "${custom_rocm_dev}" )
      library_dependencies_fedora+=( "${custom_rocm_dev}" )
      library_dependencies_sles+=( "${custom_rocm_dev}" )
    fi

    # Custom rocblas installation
    # Do not install rocblas if --rocblas_path flag is set,
    # as we will be building against our own rocblas intead.
    if [[ -z ${rocblas_path+foo} ]]; then
      if [[ -z ${custom_rocblas+foo} ]]; then
        # Install base rocblas package unless -b/--rocblas flag is passed
        library_dependencies_ubuntu+=( "rocblas" )
        library_dependencies_centos_rhel+=( "rocblas" )
        library_dependencies_centos_rhel_8+=( "rocblas" )
        library_dependencies_fedora+=( "rocblas" )
        library_dependencies_sles+=( "rocblas" )
      else
        # Install rocm-specific rocblas package
        library_dependencies_ubuntu+=( "${custom_rocblas}" )
        library_dependencies_centos_rhel+=( "${custom_rocblas}" )
        library_dependencies_centos_rhel_8+=( "${custom_rocblas}" )
        library_dependencies_fedora+=( "${custom_rocblas}" )
        library_dependencies_sles+=( "${custom_rocblas}" )
      fi
    fi

    # Do not install rocsolver if --rocsolver_path flag is set,
    if [[ -z ${rocsolver_path+foo} ]]; then
      if [[ "${build_solver}" == true ]]; then
        library_dependencies_ubuntu+=( "rocsolver" )
        library_dependencies_centos_rhel+=( "rocsolver" )
        library_dependencies_centos_rhel_8+=( "rocsolver" )
        library_dependencies_fedora+=( "rocsolver" )
        library_dependencies_sles+=( "rocsolver" )
      fi
    fi
  fi

  # wget is needed for cmake
  if [ -z "$CMAKE_VERSION" ] || $(dpkg --compare-versions $CMAKE_VERSION lt 3.16.8); then
    if $update_cmake == true; then
      library_dependencies_ubuntu+=("wget")
      library_dependencies_centos_rhel+=("wget")
      library_dependencies_centos_rhel_8+=("wget")
      library_dependencies_fedora+=("wget")
      library_dependencies_sles+=("wget")
    fi
  fi

  if [[ "${build_clients}" == true ]]; then
    library_dependencies_ubuntu+=( "gfortran" )
    library_dependencies_centos_rhel+=( "devtoolset-7-gcc-gfortran" )
    library_dependencies_centos_rhel_8+=( "gcc-gfortran" )
    library_dependencies_fedora+=( "gcc-gfortran" )
    library_dependencies_sles+=( "gcc-fortran pkg-config" "dpkg" )
  fi

  case "${ID}" in
    ubuntu)
      elevate_if_not_root apt update
      install_apt_packages "${library_dependencies_ubuntu[@]}"
      ;;

    centos|rhel)
#     yum -y update brings *all* installed packages up to date
#     without seeking user approval
#     elevate_if_not_root yum -y update
      if (( "${VERSION_ID%%.*}" >= "8" )); then
        install_yum_packages "${library_dependencies_centos_rhel_8[@]}"
      else
        install_yum_packages "${library_dependencies_centos_rhel[@]}"
      fi
      ;;

    fedora)
#     elevate_if_not_root dnf -y update
      install_dnf_packages "${library_dependencies_fedora[@]}"
      ;;

    sles|opensuse-leap)
#     elevate_if_not_root zypper -y update
      install_zypper_packages "${library_dependencies_sles[@]}"
      ;;
    *)
      echo "This script is currently supported on Ubuntu, SLES, CentOS, RHEL and Fedora"
      exit 2
      ;;
  esac
}

install_cuda_package()
{
  if [ -z ${ID+foo} ]; then
    printf "install_packages(): \$ID must be set\n"
    exit 2
  fi

  local cuda_dependencies=("cuda")
  case "${ID}" in
    ubuntu)
      elevate_if_not_root apt update
      if [[ "${cuda_version_install}" == "default" ]]; then
        install_apt_packages "${cuda_dependencies[@]}"
      else
        install_apt_packages_version "${cuda_dependencies[@]}" "${cuda_version_install}"
      fi
      ;;

    centos|rhel)
      if [[ "${cuda_version_install}" == "default" ]]; then
        install_yum_packages "${cuda_dependencies[@]}"
      else
        install_yum_packages_version "${cuda_dependencies[@]}" "${cuda_version_install}"
      fi
      ;;

    fedora)
      if [[ "${cuda_version_install}" == "default" ]]; then
        install_dnf_packages "${cuda_dependencies[@]}"
      else
        install_dnf_packages_version "${cuda_dependencies[@]}" "${cuda_version_install}"
      fi
      ;;

    sles|opensuse-leap)
      if [[ "${cuda_version_install}" == "default" ]]; then
        install_zypper_packages "${cuda_dependencies[@]}"
      else
        install_zypper_packages_version "${cuda_dependencies[@]}" "${cuda_version_install}"
      fi
      ;;

    *)
      echo "This script is currently supported on Ubuntu, SLES, CentOS, RHEL and Fedora"
      exit 2
      ;;
  esac
}

# given a relative path, returns the absolute path
make_absolute_path( ) {
  (cd "$1" && pwd -P)
}

# #################################################
# Pre-requisites check
# #################################################
# Exit code 0: alls well
# Exit code 1: problems with getopt
# Exit code 2: problems with supported platforms

# check if getopt command is installed
type getopt > /dev/null
if [[ $? -ne 0 ]]; then
  echo "This script uses getopt to parse arguments; try installing the util-linux package";
  exit 1
fi

# os-release file describes the system
if [[ -e "/etc/os-release" ]]; then
  source /etc/os-release
else
  echo "This script depends on the /etc/os-release file"
  exit 2
fi

# The following function exits script if an unsupported distro is detected
supported_distro

# #################################################
# global variables
# #################################################
install_package=false
install_dependencies=false
install_prefix=hipblas-install
build_clients=false
build_solver=true
build_cuda=false
build_hip_clang=true
build_release=true
build_relocatable=false
build_address_sanitizer=false
install_cuda=false
cuda_version_install=default
cuda_path=/usr/local/cuda
cmake_prefix_path=/opt/rocm
rocm_path=/opt/rocm
compiler=g++
build_static=false
build_release_debug=false
build_codecoverage=false
update_cmake=false
build_freorg_bkwdcomp=true
declare -a cmake_common_options
declare -a cmake_client_options

# #################################################
# Parameter parsing
# #################################################

# check if we have a modern version of getopt that can handle whitespace and long parameters
getopt -T
if [[ $? -eq 4 ]]; then
  GETOPT_PARSE=$(getopt --name "${0}" --longoptions help,install,codecoverage,clients,no-solver,dependencies,debug,hip-clang,no-hip-clang,compiler:,cmake_install,cuda,use-cuda,cudapath:,installcuda,installcudaversion:,static,cmakepp,relocatable:,rocm-dev:,rocblas:,rocblas-path:,rocsolver-path:,custom-target:,address-sanitizer,rm-legacy-include-dir,cmake-arg: --options rhicndgp:v:b: -- "$@")
else
  echo "Need a new version of getopt"
  exit 1
fi

if [[ $? -ne 0 ]]; then
  echo "getopt invocation failed; could not parse the command line";
  exit 1
fi

eval set -- "${GETOPT_PARSE}"

while true; do
  case "${1}" in
    -h|--help)
        display_help
        exit 0
        ;;
    -i|--install)
        install_package=true
        shift ;;
    -d|--dependencies)
        install_dependencies=true
        shift ;;
    -r|--relocatable)
        build_relocatable=true
        shift ;;
    -c|--clients)
        build_clients=true
        shift ;;
    -n|--no-solver)
        build_solver=false
        shift ;;
    -g|--debug)
        build_release=false
        shift ;;
    -k|--relwithdebinfo)
        build_release=false
        build_release_debug=true
        shift ;;
    --codecoverage)
        build_codecoverage=true
        shift ;;
    --hip-clang)
        build_hip_clang=true
        shift ;;
    --no-hip-clang)
        build_hip_clang=false
        shift ;;
    --compiler)
        compiler=${2}
        shift 2 ;;
    --cuda|--use-cuda)
        build_cuda=true
        shift ;;
    --cudapath)
	cuda_path=${2}
	export CUDA_BIN_PATH=${cuda_path}
	shift 2 ;;
    --installcuda)
	install_cuda=true
	shift ;;
    --installcudaversion)
	cuda_version_install=${2}
	shift 2 ;;
    --static)
        build_static=true
        shift ;;
    --cmake_install)
        update_cmake=true
        shift ;;
    --address-sanitizer)
        build_address_sanitizer=true
        compiler=hipcc
        shift ;;
    --rm-legacy-include-dir)
        build_freorg_bkwdcomp=false
        shift ;;
    -p|--cmakepp)
        cmake_prefix_path=${2}
        shift 2 ;;
    --custom-target)
        custom_target=${2}
        shift 2 ;;
    -v|--rocm-dev)
         custom_rocm_dev=${2}
         shift 2;;
    -b|--rocblas)
         custom_rocblas=${2}
         shift 2;;
    --rocblas-path)
        rocblas_path=${2}
        shift 2 ;;
    --rocsolver-path)
        rocsolver_path=${2}
        shift 2 ;;
    --prefix)
        install_prefix=${2}
        shift 2 ;;
    --cmake-arg)
        cmake_common_options+=("${2}")
        shift 2 ;;
    --) shift ; break ;;
    *)  echo "Unexpected command line parameter received; aborting";
        exit 1
        ;;
  esac
done

if [[ "${build_relocatable}" == true ]]; then
    if ! [ -z ${ROCM_PATH+x} ]; then
        rocm_path=${ROCM_PATH}
    fi
fi

build_dir=./build
printf "\033[32mCreating project build directory in: \033[33m${build_dir}\033[0m\n"

# #################################################
# prep
# #################################################
# ensure a clean build environment
if [[ "${build_release}" == true ]]; then
  rm -rf ${build_dir}/release
elif [[ "${build_release_debug}" == true ]]; then
  rm -rf ${build_dir}/release-debug
else
  rm -rf ${build_dir}/debug
fi

# resolve relative paths
if [[ -n "${rocblas_path+x}" ]]; then
  rocblas_path="$(make_absolute_path "${rocblas_path}")"
fi
if [[ -n "${rocsolver_path+x}" ]]; then
  rocsolver_path="$(make_absolute_path "${rocsolver_path}")"
fi

# Default cmake executable is called cmake
cmake_executable=cmake

# #################################################
# dependencies
# #################################################
if [[ "${install_dependencies}" == true ]]; then

  CMAKE_VERSION=$(cmake --version | grep -oP '(?<=version )[^ ]*' )

  install_packages

  if [ -z "$CMAKE_VERSION" ] || $(dpkg --compare-versions $CMAKE_VERSION lt 3.16.8); then
      if $update_cmake == true; then
        CMAKE_REPO="https://github.com/Kitware/CMake/releases/download/v3.16.8/"
        wget -nv ${CMAKE_REPO}/cmake-3.16.8.tar.gz
        tar -xvf cmake-3.16.8.tar.gz
        cd cmake-3.16.8
        ./bootstrap --prefix=/usr --no-system-curl --parallel=16
        make -j16
        sudo make install
        cd ..
        rm -rf cmake-3.16.8.tar.gz cmake-3.16.8
      else
          echo "hipBLAS requires CMake version >= 3.16.8 and CMake version ${CMAKE_VERSION} is installed. Run install.sh again with --cmake_install flag and CMake version ${CMAKE_VERSION} will be uninstalled and CMake version 3.16.8 will be installed"
          exit 2
      fi
  fi

  # The following builds googletest & lapack from source, installs into cmake default /usr/local
  pushd .
    printf "\033[32mBuilding \033[33mgoogletest & lapack\033[32m from source; installing into \033[33m/usr/local\033[0m\n"
    mkdir -p ${build_dir}/deps && cd ${build_dir}/deps
    export FC="gfortran"
    ${cmake_executable} -DCMAKE_INSTALL_PREFIX=deps-install ../../deps
    make -j$(nproc)
    make install
  popd
fi

if [[ "${install_cuda}" == true ]]; then
  install_cuda_package
fi

# We append customary rocm path; if user provides custom rocm path in ${path}, our
# hard-coded path has lesser priority
# export PATH=${PATH}:/opt/rocm/bin
pushd .
  # #################################################
  # configure & build
  # #################################################

  if [[ "${build_static}" == true ]]; then
    if [[ "${build_cuda}" == true ]]; then
      printf "Static library not supported for CUDA backend.\n"
      exit 1
    fi
    cmake_common_options+=("-DBUILD_SHARED_LIBS=OFF")
    compiler="${rocm_path}/bin/hipcc" #force hipcc for static libs, g++ doesn't work
    printf "Forcing compiler to hipcc for static library.\n"
  fi

  # build type
  if [[ "${build_release}" == true ]]; then
    mkdir -p ${build_dir}/release/clients && cd ${build_dir}/release
    cmake_common_options+=("-DCMAKE_BUILD_TYPE=Release")
  elif [[ "${build_release_debug}" == true ]]; then
    mkdir -p ${build_dir}/release-debug/clients && cd ${build_dir}/release-debug
    cmake_common_options+=("-DCMAKE_BUILD_TYPE=RelWithDebInfo")
  else
    mkdir -p ${build_dir}/debug/clients && cd ${build_dir}/debug
    cmake_common_options+=("-DCMAKE_BUILD_TYPE=Debug")
  fi

  # cuda
  if [[ "${build_cuda}" == true ]]; then
    cmake_common_options+=("-DUSE_CUDA=ON")
  else
    cmake_common_options+=("-DUSE_CUDA=OFF")
  fi

  # clients
  if [[ "${build_clients}" == true ]]; then
    cmake_client_options+=("-DBUILD_CLIENTS_TESTS=ON" "-DBUILD_CLIENTS_BENCHMARKS=ON" "-DBUILD_CLIENTS_SAMPLES=ON")
  fi

  # solver
  if [[ "${build_solver}" == false ]]; then
    cmake_common_options+=("-DBUILD_WITH_SOLVER=OFF")
  fi

  # sanitizer
  if [[ "${build_address_sanitizer}" == true ]]; then
    cmake_common_options+=("-DBUILD_ADDRESS_SANITIZER=ON")
  fi

  if [[ ${custom_target+foo} ]]; then
    cmake_common_options+=("-DCUSTOM_TARGET=${custom_target}")
  fi

  # custom rocblas
  if [[ ${rocblas_path+foo} ]]; then
    cmake_common_options+=("-DCUSTOM_ROCBLAS=${rocblas_path}")
  fi

  # custom rocsolver
  if [[ ${rocsolver_path+foo} ]]; then
    cmake_common_options+=("-DCUSTOM_ROCSOLVER=${rocsolver_path}")
  fi

  # code coverage
  if [[ "${build_codecoverage}" == true ]]; then
      if [[ "${build_release}" == true ]]; then
          echo "Code coverage is disabled in Release mode, to enable code coverage select either Debug mode (-g | --debug) or RelWithDebInfo mode (-k | --relwithdebinfo); aborting";
          exit 1
      fi
      cmake_common_options+=("-DBUILD_CODE_COVERAGE=ON")
  fi

  if [[ "${build_freorg_bkwdcomp}" == true ]]; then
    cmake_common_options+=("-DBUILD_FILE_REORG_BACKWARD_COMPATIBILITY=ON")
  else
    cmake_common_options+=("-DBUILD_FILE_REORG_BACKWARD_COMPATIBILITY=OFF")
  fi

  # Build library
  if [[ "${build_relocatable}" == true ]]; then
    CXX=${compiler} ${cmake_executable} ${cmake_common_options[@]} ${cmake_client_options[@]} -DCPACK_SET_DESTDIR=OFF -DCMAKE_INSTALL_PREFIX="${rocm_path}" \
    -DCMAKE_PREFIX_PATH="${rocm_path};${rocm_path}/hip;$(pwd)/../deps/deps-install;${cuda_path};${cmake_prefix_path}" \
    -DCMAKE_SKIP_INSTALL_RPATH=TRUE \
    -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=FALSE \
    -DROCM_PATH="${rocm_path}" ../..
  else
    CXX=${compiler} ${cmake_executable} ${cmake_common_options[@]} ${cmake_client_options[@]} -DCPACK_SET_DESTDIR=OFF -DCMAKE_PREFIX_PATH="$(pwd)/../deps/deps-install;${cmake_prefix_path}" -DROCM_PATH=${rocm_path} ../..
  fi
  check_exit_code "$?"

  make -j$(nproc)
  check_exit_code "$?"

  # #################################################
  # install
  # #################################################
  # installing through package manager, which makes uninstalling easy
  if [[ "${install_package}" == true ]]; then
    make package
    check_exit_code "$?"

    case "${ID}" in
      ubuntu)
        elevate_if_not_root dpkg -i hipblas[-\_]*.deb
      ;;
      centos|rhel)
        elevate_if_not_root yum -y localinstall hipblas-*.rpm
      ;;
      fedora)
        elevate_if_not_root dnf install hipblas-*.rpm
      ;;
      sles|opensuse-leap)
        elevate_if_not_root zypper -n --no-gpg-checks install hipblas-*.rpm
      ;;
    esac

  fi
popd
