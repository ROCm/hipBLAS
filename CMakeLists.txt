# ########################################################################
# Copyright 2016-2021 Advanced Micro Devices, Inc.
# ########################################################################

# The ROCm platform requires Ubuntu 16.04 or Fedora 24, which has cmake 3.5
cmake_minimum_required( VERSION 3.5 )

# We use C++14 features, this will add compile option: -std=c++14
set( CMAKE_CXX_STANDARD 14 )

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

if ( NOT DEFINED CMAKE_Fortran_COMPILER AND NOT DEFINED ENV{FC} )
  set( CMAKE_Fortran_COMPILER  "gfortran" )
endif()

project( hipblas LANGUAGES CXX Fortran )

# This finds the rocm-cmake project, and installs it if not found
# rocm-cmake contains common cmake code for rocm projects to help setup and install
set( PROJECT_EXTERN_DIR ${CMAKE_CURRENT_BINARY_DIR}/extern )
find_package( ROCM CONFIG QUIET PATHS /opt/rocm )
if( NOT ROCM_FOUND )
  set( rocm_cmake_tag "master" CACHE STRING "rocm-cmake tag to download" )
  file( DOWNLOAD https://github.com/RadeonOpenCompute/rocm-cmake/archive/${rocm_cmake_tag}.zip
    ${PROJECT_EXTERN_DIR}/rocm-cmake-${rocm_cmake_tag}.zip )

  execute_process( COMMAND ${CMAKE_COMMAND} -E tar xzf ${PROJECT_EXTERN_DIR}/rocm-cmake-${rocm_cmake_tag}.zip
    WORKING_DIRECTORY ${PROJECT_EXTERN_DIR} )

  find_package( ROCM REQUIRED CONFIG PATHS ${PROJECT_EXTERN_DIR}/rocm-cmake-${rocm_cmake_tag} )
endif( )

include( ROCMSetupVersion )
include( ROCMCreatePackage )
include( ROCMInstallTargets )
include( ROCMPackageConfigHelpers )
include( ROCMInstallSymlinks )

set ( VERSION_STRING "0.47.0" )
rocm_setup_version( VERSION ${VERSION_STRING} )

if( NOT DEFINED $ENV{HIP_PATH})
    set( HIP_PATH "/opt/rocm/hip" )
else( )
    set (HIP_PATH $ENV{HIP_PATH} )
endif( )

# Append our library helper cmake path and the cmake path for hip (for convenience)
# Users may override HIP path by specifying their own in CMAKE_MODULE_PATH
list( APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake  ${HIP_PATH}/cmake)

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
    find_package( hip REQUIRED CONFIG PATHS ${ROCM_PATH} /opt/rocm)
endif( )

if( USE_CUDA )
    list( APPEND HIP_INCLUDE_DIRS "${HIP_ROOT_DIR}/include" )
endif( )


option(BUILD_CODE_COVERAGE "Build with code coverage enabled" OFF)
if(BUILD_CODE_COVERAGE)
  add_compile_options(-fprofile-arcs -ftest-coverage)
  add_link_options(--coverage)
endif()


add_subdirectory( library )

include( clients/cmake/build-options.cmake )

# force library install path to lib (CentOS 7 defaults to lib64)
set(CMAKE_INSTALL_LIBDIR "lib" CACHE INTERNAL "Installation directory for libraries" FORCE)

# Build clients of the library
if( BUILD_CLIENTS_SAMPLES OR BUILD_CLIENTS_TESTS OR BUILD_CLIENTS_BENCHMARKS )
  add_subdirectory( clients )
endif( )


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

