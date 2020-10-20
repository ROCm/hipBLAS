# ########################################################################
# Copyright 2016-2020 Advanced Micro Devices, Inc.
# ########################################################################

# The ROCm platform requires Ubuntu 16.04 or Fedora 24, which has cmake 3.5
cmake_minimum_required( VERSION 3.5 )

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

# This project may compile dependencies for clients
project( hipblas-clients LANGUAGES CXX Fortran )

# We use C++14 features, this will add compile option: -std=c++14
set( CMAKE_CXX_STANDARD 14 )

list( APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake )

include( build-options )

set(hipblas_f90_source_clients
  include/hipblas_fortran.f90
)

set(hipblas_f90_source_clients_solver
  include/hipblas_fortran_solver.f90
)

if( BUILD_CLIENTS_TESTS OR BUILD_CLIENTS_BENCHMARKS OR BUILD_CLIENTS_SAMPLES )
  if( BUILD_WITH_SOLVER)
    add_library(hipblas_fortran_client ${hipblas_f90_source_clients} ${hipblas_f90_source_clients_solver})
  else()
    add_library(hipblas_fortran_client ${hipblas_f90_source_clients})
  endif()

  add_dependencies(hipblas_fortran_client hipblas_fortran)
  include_directories(${CMAKE_BINARY_DIR}/include)
endif( )

if( BUILD_CLIENTS_TESTS )
  add_subdirectory( gtest )
endif( )

if( BUILD_CLIENTS_BENCHMARKS )
  add_subdirectory( benchmarks )
endif( )

if( BUILD_CLIENTS_SAMPLES )
  add_subdirectory( samples )
endif( )
