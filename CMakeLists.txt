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
# ########################################################################

# The ROCm platform requires Ubuntu 16.04 or Fedora 24, which has cmake 3.5
cmake_minimum_required( VERSION 3.5 )

# We use C++14 features, this will add compile option: -std=c++14
set( CMAKE_CXX_STANDARD 14 )

if (NOT python)
    set(python "python3") # default for linux
endif()

# Consider removing this in the future
# This should appear before the project command, because it does not use FORCE
if( WIN32 )
  set( CMAKE_INSTALL_PREFIX "${PROJECT_BINARY_DIR}/package" CACHE PATH "Install path prefix, prepended onto install directories" )
else( )
  set( CMAKE_INSTALL_PREFIX "/opt/rocm" CACHE PATH "Install path prefix, prepended onto install directories" )
endif( )

# This has to be initialized before the project() command appears
# Set the default of CMAKE_BUILD_TYPE to be release, unless user specifies with -D.  MSVC_IDE does not use CMAKE_BUILD_TYPE
if( NOT DEFINED CMAKE_CONFIGURATION_TYPES AND NOT DEFINED CMAKE_BUILD_TYPE )
  set( CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel." )
endif()

if (NOT WIN32)
    if ( NOT DEFINED CMAKE_Fortran_COMPILER AND NOT DEFINED ENV{FC} )
      set( CMAKE_Fortran_COMPILER  "gfortran" )
    endif()
    set( fortran_language "Fortran" )
endif( )

project( hipblas LANGUAGES CXX ${fortran_language} )

# This finds the rocm-cmake project, and installs it if not found
# rocm-cmake contains common cmake code for rocm projects to help setup and install
set( PROJECT_EXTERN_DIR ${CMAKE_CURRENT_BINARY_DIR}/extern )
find_package( ROCM 0.7.3 CONFIG QUIET PATHS /opt/rocm )
if( NOT ROCM_FOUND )
  set( rocm_cmake_tag "master" CACHE STRING "rocm-cmake tag to download" )
  file( DOWNLOAD https://github.com/RadeonOpenCompute/rocm-cmake/archive/${rocm_cmake_tag}.zip
    ${PROJECT_EXTERN_DIR}/rocm-cmake-${rocm_cmake_tag}.zip )

  execute_process( COMMAND ${CMAKE_COMMAND} -E tar xzf ${PROJECT_EXTERN_DIR}/rocm-cmake-${rocm_cmake_tag}.zip
    WORKING_DIRECTORY ${PROJECT_EXTERN_DIR} )
  execute_process( COMMAND ${CMAKE_COMMAND} -DCMAKE_INSTALL_PREFIX=${PROJECT_EXTERN_DIR}/rocm-cmake .
    WORKING_DIRECTORY ${PROJECT_EXTERN_DIR}/rocm-cmake-${rocm_cmake_tag} )
  execute_process( COMMAND ${CMAKE_COMMAND} --build rocm-cmake-${rocm_cmake_tag} --target install
    WORKING_DIRECTORY ${PROJECT_EXTERN_DIR})

  find_package( ROCM 0.7.3 REQUIRED CONFIG PATHS ${PROJECT_EXTERN_DIR}/rocm-cmake )
endif( )

include( ROCMSetupVersion )
include( ROCMCreatePackage )
include( ROCMInstallTargets )
include( ROCMPackageConfigHelpers )
include( ROCMInstallSymlinks )
include( ROCMClients )
include( ROCMHeaderWrapper )

set ( VERSION_STRING "0.53.0" )
rocm_setup_version( VERSION ${VERSION_STRING} )

if( NOT DEFINED ENV{HIP_PATH})
    set( HIP_PATH "/opt/rocm/hip" )
else( )
    set (HIP_PATH $ENV{HIP_PATH} )
endif( )

# Append our library helper cmake path and the cmake path for hip (for convenience)
# Users may override HIP path by specifying their own in CMAKE_MODULE_PATH
list( APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake  ${ROCM_PATH}/lib/cmake/hip /opt/rocm/lib/cmake/hip ${HIP_PATH}/cmake )

# NOTE:  workaround until hip cmake modules fixes symlink logic in their config files; remove when fixed
list( APPEND CMAKE_PREFIX_PATH /opt/rocm /opt/rocm/llvm /opt/rocm/hip )

option( BUILD_VERBOSE "Output additional build information" OFF )

option( BUILD_WITH_SOLVER "Add additional functions from rocSOLVER" ON )

if( BUILD_WITH_SOLVER )
    add_definitions( -D__HIP_PLATFORM_SOLVER__ )
endif( )

# BUILD_SHARED_LIBS is a cmake built-in; we make it an explicit option such that it shows in cmake-gui
option( BUILD_SHARED_LIBS "Build hipBLAS as a shared library" ON )

# Find CUDA if the user wants a CUDA version.
option(USE_CUDA "Look for CUDA and use that as a backend if found" OFF)
if (USE_CUDA)
	find_package( CUDA REQUIRED )
endif()

# Hip headers required of all clients; clients use hip to allocate device memory
if( USE_CUDA)
    find_package( HIP MODULE REQUIRED )
else( )
    find_package( hip REQUIRED CONFIG PATHS ${HIP_PATH} ${ROCM_PATH} /opt/rocm)
endif( )

if( USE_CUDA )
    list( APPEND HIP_INCLUDE_DIRS "${HIP_ROOT_DIR}/include" )
endif( )

option(BUILD_CODE_COVERAGE "Build with code coverage enabled" OFF)
if(BUILD_CODE_COVERAGE)
  add_compile_options(-fprofile-arcs -ftest-coverage)
  add_link_options(--coverage)
endif()

option(BUILD_ADDRESS_SANITIZER "Build with address sanitizer enabled" OFF)
if(BUILD_ADDRESS_SANITIZER)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -shared-libasan")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=address -shared-libasan")
endif()


# FOR HANDLING ENABLE/DISABLE OPTIONAL BACKWARD COMPATIBILITY for FILE/FOLDER REORG
option(BUILD_FILE_REORG_BACKWARD_COMPATIBILITY "Build with file/folder reorg with backward compatibility enabled" ON)
if(BUILD_FILE_REORG_BACKWARD_COMPATIBILITY AND NOT WIN32)
  rocm_wrap_header_dir(
    ${CMAKE_SOURCE_DIR}/library/include
    PATTERNS "*.h"
    GUARDS SYMLINK WRAPPER
    WRAPPER_LOCATIONS ${CMAKE_INSTALL_INCLUDEDIR}
  )
endif()

add_subdirectory( library )

include( clients/cmake/build-options.cmake )

# Build clients of the library
if( BUILD_CLIENTS_SAMPLES OR BUILD_CLIENTS_TESTS OR BUILD_CLIENTS_BENCHMARKS )
  if(NOT CLIENTS_OS)
    rocm_set_os_id(CLIENTS_OS)
    string(TOLOWER "${CLIENTS_OS}" CLIENTS_OS)
    rocm_read_os_release(CLIENTS_OS_VERSION VERSION_ID)
  endif()
  message(STATUS "OS: ${CLIENTS_OS} ${CLIENTS_OS_VERSION}")
  set(GFORTRAN_RPM "libgfortran4")
  set(GFORTRAN_DEB "libgfortran4")
  if(CLIENTS_OS STREQUAL "centos" OR CLIENTS_OS STREQUAL "rhel")
    if(CLIENTS_OS_VERSION VERSION_GREATER_EQUAL "8")
      set(GFORTRAN_RPM "libgfortran")
    endif()
  elseif(CLIENTS_OS STREQUAL "ubuntu" AND CLIENTS_OS_VERSION VERSION_GREATER_EQUAL "20.04")
    set(GFORTRAN_DEB "libgfortran5")
  endif()
  rocm_package_setup_component(clients)
  rocm_package_setup_client_component(clients-common)
  if(BUILD_CLIENTS_TESTS)
    rocm_package_setup_client_component(
      tests
      DEPENDS
        COMPONENT clients-common
        DEB "${GFORTRAN_DEB}"
        RPM "${GFORTRAN_RPM}")
  endif()
  if(BUILD_CLIENTS_BENCHMARKS)
    rocm_package_setup_client_component(
      benchmarks
      DEPENDS
        COMPONENT clients-common
        DEB "${GFORTRAN_DEB}"
        RPM "${GFORTRAN_RPM}")
  endif()
  add_subdirectory( clients )
endif( )

# The following code is setting variables to control the behavior of CPack to generate our
if( WIN32 )
    set( CPACK_SOURCE_GENERATOR "ZIP" )
    set( CPACK_GENERATOR "ZIP" )
endif( )

# Package specific CPACK vars
if( NOT USE_CUDA )
  rocm_package_add_dependencies(DEPENDS "rocblas >= 2.46.0" "rocsolver >= 3.20.0")
endif( )

set( CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.md" )
set( CPACK_RPM_PACKAGE_LICENSE "MIT")

if (WIN32)
  SET( CMAKE_INSTALL_PREFIX "C:/hipSDK" CACHE PATH "Install path" FORCE )
  SET( INSTALL_PREFIX "C:/hipSDK" )
  SET( CPACK_SET_DESTDIR FALSE )
  SET( CPACK_PACKAGE_INSTALL_DIRECTORY "C:/hipSDK" )
  SET( CPACK_PACKAGING_INSTALL_PREFIX "" )
  set( CPACK_INCLUDE_TOPLEVEL_DIRECTORY OFF )
else()
  if( NOT CPACK_PACKAGING_INSTALL_PREFIX )
    set( CPACK_PACKAGING_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}" )
  endif()
endif( )

set( CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION "\${CPACK_PACKAGING_INSTALL_PREFIX}" )

# Give hipblas compiled for CUDA backend a different name
if( NOT USE_CUDA )
    set( package_name hipblas )
else( )
    set( package_name hipblas-alt )
endif( )

set( HIPBLAS_CONFIG_DIR "\${CPACK_PACKAGING_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}" CACHE PATH "Path placed into ldconfig file" )

rocm_create_package(
    NAME ${package_name}
    DESCRIPTION "Radeon Open Compute BLAS marshalling library"
    MAINTAINER "hipBLAS Maintainer <hipblas-maintainer@amd.com>"
    LDCONFIG
    LDCONFIG_DIR ${HIPBLAS_CONFIG_DIR}
)


#
# ADDITIONAL TARGETS FOR CODE COVERAGE
#
if(BUILD_CODE_COVERAGE)
  #
  # > make coverage_cleanup (clean coverage related files.)
  # > make coverage GTEST_FILTER=<>
  # will run:
  #  > make coverage_analysis GTEST_FILTER=<> (analyze tests)
  #  > make coverage_output (generate html documentation)
  #

  #
  # Run coverage analysis
  #
  set(coverage_test ./clients/staging/hipblas-test)
  if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(coverage_test ./clients/staging/hipblas-test-d)
  endif()

  add_custom_target(coverage_analysis
    COMMAND echo Coverage GTEST_FILTER=\${GTEST_FILTER}
    COMMAND ${coverage_test} --gtest_filter=\"\${GTEST_FILTER}\"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )

  add_dependencies(coverage_analysis hipblas)

  #
  # Prepare coverage output
  # This little script is generated because the option '--gcov-tool <program name>' of lcov cannot take arguments.
  #
  add_custom_target(coverage_output
    DEPENDS coverage_analysis
    COMMAND mkdir -p lcoverage
    COMMAND echo "\\#!/bin/bash" > llvm-gcov.sh
    COMMAND echo "\\# THIS FILE HAS BEEN GENERATED" >> llvm-gcov.sh
    COMMAND printf "exec /opt/rocm/llvm/bin/llvm-cov gcov $$\\@" >> llvm-gcov.sh
    COMMAND chmod +x llvm-gcov.sh
    )

  #
  # Generate coverage output.
  #
  add_custom_command(TARGET coverage_output
    COMMAND lcov --directory . --base-directory . --gcov-tool ${CMAKE_BINARY_DIR}/llvm-gcov.sh --capture -o lcoverage/raw_main_coverage.info
    COMMAND lcov --remove lcoverage/raw_main_coverage.info "'/opt/*'" "'/usr/*'" -o lcoverage/main_coverage.info
    COMMAND genhtml lcoverage/main_coverage.info --output-directory lcoverage
    )

  add_custom_target(coverage DEPENDS coverage_output)

  #
  # Coverage cleanup
  #
  add_custom_target(coverage_cleanup
    COMMAND find ${CMAKE_BINARY_DIR} -name *.gcda -delete
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )
endif()

