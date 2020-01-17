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
  echo "    [-g|--debug] -DCMAKE_BUILD_TYPE=Debug (default is =Release)"
  echo "    [-r]--relocatable] create a package to support relocatable ROCm"
  echo "    [--cuda] build library for cuda backend"
  echo "    [-p|--cmakepp] addition to CMAKE_PREFIX_PATH"
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

# This function is helpful for dockerfiles that do not have sudo installed, but the default user is root
elevate_if_not_root( )
{
  local uid=$(id -u)

  if (( ${uid} )); then
    sudo $@
  else
    $@
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
  local library_dependencies_ubuntu=( "make" "cmake-curses-gui" "pkg-config" "rocm-dev" )
  local library_dependencies_centos=( "epel-release" "make" "cmake3" "rocm-dev" "gcc-c++" "rpm-build" )
  local library_dependencies_fedora=( "make" "cmake" "rocm-dev" "gcc-c++" "libcxx-devel" "rpm-build" )
  local library_dependencies_sles=( "make" "cmake" "gcc-c++" "libcxxtools9" "rpm-build" )

  if [[ "${build_cuda}" == true ]]; then
    # Ideally, this could be cuda-cublas-dev, but the package name has a version number in it
    library_dependencies_ubuntu+=( "cuda" )
    library_dependencies_centos+=( "" ) # how to install cuda on centos?
    library_dependencies_fedora+=( "" ) # how to install cuda on fedora?
  else
    library_dependencies_ubuntu+=( "hcc" "rocblas" )
    library_dependencies_centos+=( "hcc" "rocblas" )
    library_dependencies_fedora+=( "hcc" "rocblas" )
    library_dependencies_sles+=( "hcc" "rocblas" )
  fi

  local client_dependencies_ubuntu=( "gfortran" "libboost-program-options-dev" )
  local client_dependencies_centos=( "gcc-gfortran" "boost-devel" )
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
      install_yum_packages "${library_dependencies_centos[@]}"

      if [[ "${build_clients}" == true ]]; then
        install_yum_packages "${client_dependencies_centos[@]}"
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
build_cuda=false
build_release=true
build_relocatable=false
cmake_prefix_path=/opt/rocm
rocm_path=/opt/rocm

# #################################################
# Parameter parsing
# #################################################

# check if we have a modern version of getopt that can handle whitespace and long parameters
getopt -T
if [[ $? -eq 4 ]]; then
  GETOPT_PARSE=$(getopt --name "${0}" --longoptions help,install,clients,dependencies,debug,cuda,cmakepp,relocatable: --options rhicdgp: -- "$@")
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
    -g|--debug)
        build_release=false
        shift ;;
    --cuda)
        build_cuda=true
        shift ;;
    -p|--cmakepp)
        cmake_prefix_path=${2}
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

  # build type
  if [[ "${build_release}" == true ]]; then
    mkdir -p ${build_dir}/release/clients && cd ${build_dir}/release
    cmake_common_options="${cmake_common_options} -DCMAKE_BUILD_TYPE=Release"
  else
    mkdir -p ${build_dir}/debug/clients && cd ${build_dir}/debug
    cmake_common_options="${cmake_common_options} -DCMAKE_BUILD_TYPE=Debug"
  fi

  # clients
  if [[ "${build_clients}" == true ]]; then
    cmake_client_options="${cmake_client_options} -DBUILD_CLIENTS_SAMPLES=ON -DBUILD_CLIENTS_TESTS=ON"
  fi

  # Build library
  if [[ "${build_relocatable}" == true ]]; then
    CXX=g++ ${cmake_executable} ${cmake_common_options} ${cmake_client_options} -DCPACK_SET_DESTDIR=OFF -DCMAKE_INSTALL_PREFIX=${rocm_path} \
    -DCMAKE_PREFIX_PATH="${rocm_path};${rocm_path}/hcc;${rocm_path}/hip;$(pwd)/../deps/deps-install;${cmake_prefix_path}" \
    -DCMAKE_SHARED_LINKER_FLAGS=${rocm_rpath} \
    -DROCM_DISABLE_LDCONFIG=ON \
    -DROCM_PATH=${rocm_path} ../..
  else
    CXX=g++ ${cmake_executable} ${cmake_common_options} ${cmake_client_options} -DCPACK_SET_DESTDIR=OFF -DCMAKE_PREFIX_PATH="$(pwd)/../deps/deps-install;${cmake_prefix_path}" -DROCM_PATH=${rocm_path} ../..
  fi
  make -j$(nproc)

  # #################################################
  # install
  # #################################################
  # installing through package manager, which makes uninstalling easy
  if [[ "${install_package}" == true ]]; then
    make package

    case "${ID}" in
      ubuntu)
        elevate_if_not_root dpkg -i hipblas-*.deb
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
