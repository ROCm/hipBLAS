# ########################################################################
# Copyright (C) 2016-2023 Advanced Micro Devices, Inc. All rights reserved.
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
set( CMAKE_CXX_STANDARD 17 )

list( APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake )

include( build-options )

if( NOT WIN32 )
    set(hipblas_f90_source_clients_no_solver
      include/hipblas_fortran_no_solver_module.f90
    )

    set(hipblas_f90_source_clients_solver
      include/hipblas_fortran_module.f90
    )
endif()

if( BUILD_CLIENTS_TESTS OR BUILD_CLIENTS_BENCHMARKS OR BUILD_CLIENTS_SAMPLES )
  if( NOT WIN32 )
    if( BUILD_WITH_SOLVER )
      add_library(hipblas_fortran_client STATIC ${hipblas_f90_source_clients_solver})
    else()
      add_library(hipblas_fortran_client STATIC ${hipblas_f90_source_clients_no_solver})
    endif()
    add_dependencies(hipblas_fortran_client hipblas_fortran)
  endif()
  include_directories(${CMAKE_BINARY_DIR}/include/hipblas)
  include_directories(${CMAKE_BINARY_DIR}/include)
endif( )

if( BUILD_CLIENTS_SAMPLES )
  add_subdirectory( samples )
endif( )

if( BUILD_CLIENTS_BENCHMARKS OR BUILD_CLIENTS_TESTS)
  if( NOT WIN32 )
    # Linking lapack library requires fortran flags
    find_package( cblas REQUIRED CONFIG )
    if (LINK_BLIS)
      set( BLIS_INCLUDE_DIR ${BUILD_DIR}/deps/blis/include/blis )
      set( BLIS_CPP ../common/blis_interface.cpp )
      set( BLAS_LIBRARY ${BUILD_DIR}/deps/blis/lib/libblis.so )
    else()
      set( BLAS_LIBRARY "blas" )
    endif()
  else() # WIN32
    set( BLAS_INCLUDE_DIR ${OPENBLAS_DIR}/include CACHE PATH "OpenBLAS library include path" )
    find_library( BLAS_LIBRARY libopenblas
                  PATHS ${OPENBLAS_DIR}/lib
                  REQUIRED
                  NO_DEFAULT_PATH
                )
    if (NOT BLAS_LIBRARY)
      find_package( OPENBLAS CONFIG REQUIRED )
      set( BLAS_LIBRARY OpenBLAS::OpenBLAS )
      set( BLAS_INCLUDE_DIR "" )
    endif()
  endif()

  if( BUILD_CLIENTS_TESTS )
    add_subdirectory( gtest )
  endif( )

  if( BUILD_CLIENTS_BENCHMARKS )
    add_subdirectory( benchmarks )
  endif( )

endif()

set( HIPBLAS_COMMON "${PROJECT_BINARY_DIR}/staging/hipblas_common.yaml")
add_custom_command( OUTPUT "${HIPBLAS_COMMON}"
                    COMMAND ${CMAKE_COMMAND} -E copy include/hipblas_common.yaml "${HIPBLAS_COMMON}"
                    DEPENDS include/hipblas_common.yaml
                    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" )

set( HIPBLAS_TEMPLATE "${PROJECT_BINARY_DIR}/staging/hipblas_template.yaml")
add_custom_command( OUTPUT "${HIPBLAS_TEMPLATE}"
                    COMMAND ${CMAKE_COMMAND} -E copy include/hipblas_template.yaml "${HIPBLAS_TEMPLATE}"
                    DEPENDS include/hipblas_template.yaml
                    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" )

set( HIPBLAS_SMOKE "${PROJECT_BINARY_DIR}/staging/hipblas_smoke.yaml")
add_custom_command( OUTPUT "${HIPBLAS_SMOKE}"
                    COMMAND ${CMAKE_COMMAND} -E copy include/hipblas_smoke.yaml "${HIPBLAS_SMOKE}"
                    DEPENDS include/hipblas_smoke.yaml
                    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" )

set( HIPBLAS_GENTEST "${PROJECT_BINARY_DIR}/staging/hipblas_gentest.py")
add_custom_command( OUTPUT "${HIPBLAS_GENTEST}"
                    COMMAND ${CMAKE_COMMAND} -E copy common/hipblas_gentest.py "${HIPBLAS_GENTEST}"
                    DEPENDS common/hipblas_gentest.py
                    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" )


add_custom_target( hipblas-common DEPENDS "${HIPBLAS_COMMON}" "${HIPBLAS_TEMPLATE}" "${HIPBLAS_SMOKE}" "${HIPBLAS_GENTEST}" )

if(BUILD_CLIENTS_BENCHMARKS)
  rocm_install(
    FILES ${HIPBLAS_COMMON} ${HIPBLAS_TEMPLATE} ${HIPBLAS_SMOKE}
    DESTINATION "${CMAKE_INSTALL_BINDIR}"
    COMPONENT clients-common
  )
  rocm_install(
    PROGRAMS ${HIPBLAS_GENTEST}
    DESTINATION "${CMAKE_INSTALL_BINDIR}"
    COMPONENT clients-common
  )
endif()
