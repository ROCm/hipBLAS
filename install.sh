#!/usr/bin/env bash

/bin/ln -fs ../../.githooks/pre-commit "$(dirname "$0")/.git/hooks/"


# #################################################
# helper functions
# #################################################
function display_help()
{
  echo "hipBLAS build & installation helper script"
  echo "./install [-h|--help] "
  echo "    [-h|--help] prints this help message"
  echo "    [-i|--install] install after build"
  echo "    [-d|--dependencies] install build dependencies"
  echo "    [-c|--clients] build library clients too (combines with -i & -d)"
  echo "    [-n|--no-solver] build library without rocSOLVER dependency"
  echo "    [-g|--debug] -DCMAKE_BUILD_TYPE=Debug (default is =Release)"
  echo "    [-r]--relocatable] create a package to support relocatable ROCm"
  echo "    [--cuda|--use-cuda] build library for cuda backend"
  echo "    [--[no-]hip-clang] Whether to build library with hip-clang"
  echo "    [--compiler] specify host compiler"
  echo "    [-p|--cmakepp] addition to CMAKE_PREFIX_PATH"
  echo "    [--custom-target] link against custom target (e.g. host, device)"
  echo "    [-v|--rocm-dev] Set specific rocm-dev version"
  echo "    [-b|--rocblas] Set specific rocblas version"
  echo "    [--rocblas-path] Set specific path to custom built rocblas"
  echo "    [--static] Create static library instead of shared library"
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
  package_dependencies=("$@")
  for package in "${package_dependencies[@]}"; do
    if [[ $(dpkg-query --show --showformat='${db:Status-Abbrev}\n' ${package} 2> /dev/null | grep -q "ii"; echo $?) -ne 0 ]]; then
      printf "\033[32mInstalling \033[33m${package}\033[32m from distro package manager\033[0m\n"
      elevate_if_not_root apt install -y --no-install-recommends ${package}
    fi
  done
}

# Take an array of packages as input, and install those packages with 'yum' if they are not already installed
install_yum_packages( )
{
  package_dependencies=("$@")
  for package in "${package_dependencies[@]}"; do
    if [[ $(yum list installed ${package} &> /dev/null; echo $? ) -ne 0 ]]; then
      printf "\033[32mInstalling \033[33m${package}\033[32m from distro package manager\033[0m\n"
      elevate_if_not_root yum -y --nogpgcheck install ${package}
    fi
  done
}

# Take an array of packages as input, and install those packages with 'dnf' if they are not already installed
install_dnf_packages( )
{
  package_dependencies=("$@")
  for package in "${package_dependencies[@]}"; do
    if [[ $(dnf list installed ${package} &> /dev/null; echo $? ) -ne 0 ]]; then
      printf "\033[32mInstalling \033[33m${package}\033[32m from distro package manager\033[0m\n"
      elevate_if_not_root dnf install -y ${package}
    fi
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
  local library_dependencies_ubuntu=( "make" "cmake-curses-gui" "pkg-config" )
  local library_dependencies_centos=( "epel-release" "make" "cmake3" "gcc-c++" "rpm-build" )
  local library_dependencies_centos8=( "epel-release" "make" "cmake3" "gcc-c++" "rpm-build" )
  local library_dependencies_fedora=( "make" "cmake" "gcc-c++" "libcxx-devel" "rpm-build" )
  local library_dependencies_sles=( "make" "cmake" "gcc-c++" "libcxxtools9" "rpm-build" )

  if [[ "${build_cuda}" == true ]]; then
    # Ideally, this could be cuda-cublas-dev, but the package name has a version number in it
    library_dependencies_ubuntu+=( "cuda" )
    library_dependencies_centos+=( "" ) # how to install cuda on centos?
    library_dependencies_fedora+=( "" ) # how to install cuda on fedora?
  elif [[ "${build_hip_clang}" == false ]]; then
    # Custom rocm-dev installation
    if [[ -z ${custom_rocm_dev+foo} ]]; then
      # Install base rocm-dev package unless -v/--rocm-dev flag is passed
      library_dependencies_ubuntu+=( "rocm-dev" )
      library_dependencies_centos+=( "rocm-dev" )
      library_dependencies_fedora+=( "rocm-dev" )
      library_dependencies_sles+=( "rocm-dev" )
    else
      # Install rocm-specific rocm-dev package
      library_dependencies_ubuntu+=( "${custom_rocm_dev}" )
      library_dependencies_centos+=( "${custom_rocm_dev}" )
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
        library_dependencies_centos+=( "rocblas" )
        library_dependencies_fedora+=( "rocblas" )
        library_dependencies_sles+=( "rocblas" )
      else
        # Install rocm-specific rocblas package
        library_dependencies_ubuntu+=( "${custom_rocblas}" )
        library_dependencies_centos+=( "${custom_rocblas}" )
        library_dependencies_fedora+=( "${custom_rocblas}" )
        library_dependencies_sles+=( "${custom_rocblas}" )
      fi
    fi

    if [[ "${build_solver}" == true ]]; then
      library_dependencies_ubuntu+=( "rocsolver" )
      library_dependencies_centos+=( "rocsolver" )
      library_dependencies_fedora+=( "rocsolver" )
      library_dependencies_sles+=( "rocsolver" )
    fi
  fi

  local client_dependencies_ubuntu=( "gfortran" "libboost-program-options-dev" )
  local client_dependencies_centos=( "devtoolset-7-gcc-gfortran" "boost-devel" )
  local client_dependencies_centos8=( "gcc-gfortran" "boost-devel" )
  local client_dependencies_fedora=( "gcc-gfortran" "boost-devel" )
  local client_dependencies_sles=( "libboost_program_options1_66_0-devel" "pkg-config" "dpkg" )

  case "${ID}" in
    ubuntu)
      elevate_if_not_root apt update
      install_apt_packages "${library_dependencies_ubuntu[@]}"

      if [[ "${build_clients}" == true ]]; then
        install_apt_packages "${client_dependencies_ubuntu[@]}"
      fi
      ;;

    centos|rhel)
#     yum -y update brings *all* installed packages up to date
#     without seeking user approval
#     elevate_if_not_root yum -y update
      if [[ "${VERSION_ID}" == "8" ]]; then
        install_yum_packages "${library_dependencies_centos8[@]}"
        if [[ "${build_clients}" == true ]]; then
          install_yum_packages "${client_dependencies_centos8[@]}"
        fi
      else
        install_yum_packages "${library_dependencies_centos[@]}"
        if [[ "${build_clients}" == true ]]; then
          install_yum_packages "${client_dependencies_centos[@]}"
        fi
      fi
      ;;

    fedora)
#     elevate_if_not_root dnf -y update
      install_dnf_packages "${library_dependencies_fedora[@]}"

      if [[ "${build_clients}" == true ]]; then
        install_dnf_packages "${client_dependencies_fedora[@]}"
      fi
      ;;

    sles|opensuse-leap)
#     elevate_if_not_root zypper -y update
      install_zypper_packages "${library_dependencies_sles[@]}"

      if [[ "${build_clients}" == true ]]; then
        install_zypper_packages "${client_dependencies_sles[@]}"
      fi
      ;;
    *)
      echo "This script is currently supported on Ubuntu, SLES, CentOS, RHEL and Fedora"
      exit 2
      ;;
  esac
}

# Take an array of packages as input, and install those packages with 'zypper' if they are not already installed
install_zypper_packages( )
{
  package_dependencies=("$@")
  for package in "${package_dependencies[@]}"; do
    if [[ $(rpm -q ${package} &> /dev/null; echo $? ) -ne 0 ]]; then
      printf "\033[32mInstalling \033[33m${package}\033[32m from distro package manager\033[0m\n"
      elevate_if_not_root zypper -n --no-gpg-checks install ${package}
    fi
  done
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
cmake_prefix_path=/opt/rocm
rocm_path=/opt/rocm
compiler=g++
build_static=false

# #################################################
# Parameter parsing
# #################################################

# check if we have a modern version of getopt that can handle whitespace and long parameters
getopt -T
if [[ $? -eq 4 ]]; then
  GETOPT_PARSE=$(getopt --name "${0}" --longoptions help,install,clients,no-solver,dependencies,debug,hip-clang,no-hip-clang,compiler:,cuda,use-cuda,static,cmakepp,relocatable:,rocm-dev:,rocblas:,rocblas-path:,custom-target: --options rhicndgp:v:b: -- "$@")
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
    --static)
        build_static=true
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
    --prefix)
        install_prefix=${2}
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

    rocm_rpath=" -Wl,--enable-new-dtags -Wl,--rpath,/opt/rocm/lib:/opt/rocm/lib64"
    if ! [ -z ${ROCM_RPATH+x} ]; then
        rocm_rpath=" -Wl,--enable-new-dtags -Wl,--rpath,${ROCM_RPATH}"
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
else
  rm -rf ${build_dir}/debug
fi

# Default cmake executable is called cmake
cmake_executable=cmake

case "${ID}" in
  centos|rhel)
  cmake_executable=cmake3
  ;;
esac

# #################################################
# dependencies
# #################################################
if [[ "${install_dependencies}" == true ]]; then

  install_packages

  # The following builds googletest & lapack from source, installs into cmake default /usr/local
  pushd .
    printf "\033[32mBuilding \033[33mgoogletest & lapack\033[32m from source; installing into \033[33m/usr/local\033[0m\n"
    mkdir -p ${build_dir}/deps && cd ${build_dir}/deps
    ${cmake_executable} -DCMAKE_INSTALL_PREFIX=deps-install -DBUILD_BOOST=OFF ../../deps
    make -j$(nproc)
    make install
  popd
  fi

# We append customary rocm path; if user provides custom rocm path in ${path}, our
# hard-coded path has lesser priority
# export PATH=${PATH}:/opt/rocm/bin
pushd .
  # #################################################
  # configure & build
  # #################################################
  cmake_common_options=""
  cmake_client_options=""

  if [[ "${build_static}" == true ]]; then
    if [[ "${build_cuda}" == true ]]; then
      printf "Static library not supported for CUDA backend.\n"
      exit 1
    fi
    cmake_common_options="${cmake_common_options} -DBUILD_SHARED_LIBS=OFF"
    compiler="${rocm_path}/bin/hipcc" #force hipcc for static libs, g++ doesn't work
    printf "Forcing compiler to hipcc for static library.\n"
  fi

  # build type
  if [[ "${build_release}" == true ]]; then
    mkdir -p ${build_dir}/release/clients && cd ${build_dir}/release
    cmake_common_options="${cmake_common_options} -DCMAKE_BUILD_TYPE=Release"
  else
    mkdir -p ${build_dir}/debug/clients && cd ${build_dir}/debug
    cmake_common_options="${cmake_common_options} -DCMAKE_BUILD_TYPE=Debug"
  fi

  # cuda
  if [[ "${build_cuda}" == false ]]; then
    cmake_common_options="${cmake_common_options} -DUSE_CUDA=OFF"
  fi

  # clients
  if [[ "${build_clients}" == true ]]; then
    cmake_client_options="${cmake_client_options} -DBUILD_CLIENTS_SAMPLES=ON -DBUILD_CLIENTS_TESTS=ON -DBUILD_CLIENTS_BENCHMARKS=ON"
  fi

  # solver
  if [[ "${build_solver}" == false ]]; then
    cmake_client_options="${cmake_client_options} -DBUILD_WITH_SOLVER=OFF"
  fi

  if [[ ${custom_target+foo} ]]; then
    cmake_common_options="${cmake_common_options} -DCUSTOM_TARGET=${custom_target}"
  fi

  # custom rocblas
  if [[ ${rocblas_path+foo} ]]; then
    cmake_common_options="${cmake_common_options} -DCUSTOM_ROCBLAS=${rocblas_path}"
  fi

  # Build library
  if [[ "${build_relocatable}" == true ]]; then
    CXX=${compiler} ${cmake_executable} ${cmake_common_options} ${cmake_client_options} -DCPACK_SET_DESTDIR=OFF -DCMAKE_INSTALL_PREFIX="${rocm_path}" \
    -DCMAKE_PREFIX_PATH="${rocm_path};${rocm_path}/hcc;${rocm_path}/hip;$(pwd)/../deps/deps-install;/usr/local/cuda;${cmake_prefix_path}" \
    -DCMAKE_SHARED_LINKER_FLAGS="${rocm_rpath}" \
    -DCMAKE_EXE_LINKER_FLAGS=" -Wl,--enable-new-dtags -Wl,--rpath,${rocm_path}/lib:${rocm_path}/lib64" \
    -DROCM_DISABLE_LDCONFIG=ON \
    -DROCM_PATH="${rocm_path}" ../..
  else
    CXX=${compiler} ${cmake_executable} ${cmake_common_options} ${cmake_client_options} -DCPACK_SET_DESTDIR=OFF -DCMAKE_PREFIX_PATH="$(pwd)/../deps/deps-install;${cmake_prefix_path}" -DROCM_PATH=${rocm_path} ../..
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
