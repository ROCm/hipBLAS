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

set( THREADS_PREFER_PTHREAD_FLAG ON )
find_package( Threads REQUIRED )

# Linking lapack library requires fortran flags
enable_language( Fortran )

set(hipblas_bench_source client.cpp)

if( NOT TARGET hipblas )
  find_package( hipblas REQUIRED CONFIG PATHS /opt/rocm/hipblas )
endif( )

set( hipblas_benchmark_common
      ../common/utility.cpp
      ../common/cblas_interface.cpp
      ../common/hipblas_arguments.cpp
      ../common/hipblas_parse_data.cpp
      ../common/hipblas_datatype2string.cpp
      ../common/norm.cpp
      ../common/unit.cpp
      ../common/near.cpp
      ../common/arg_check.cpp
      ../common/argument_model.cpp
      ../common/hipblas_template_specialization.cpp
      ${BLIS_CPP}
    )

    add_executable( hipblas-bench ${hipblas_bench_source} ${hipblas_benchmark_common} )

target_compile_features( hipblas-bench PRIVATE cxx_static_assert cxx_nullptr cxx_auto_type )

if(LINK_BLIS)
  target_link_libraries( hipblas-bench PRIVATE ${BLIS_LIBRARY} )
else()
  target_link_libraries( hipblas-bench PRIVATE blas )
endif()

# Internal header includes
target_include_directories( hipblas-bench
  PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
)

# External header includes included as system files
target_include_directories( hipblas-bench
  SYSTEM PRIVATE
    $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>
    $<BUILD_INTERFACE:${CBLAS_INCLUDE_DIRS}>
    $<BUILD_INTERFACE:${BLAS_INCLUDE_DIR}>
    $<BUILD_INTERFACE:${BLIS_INCLUDE_DIR}>
)

if (NOT WIN32)
    target_link_libraries( hipblas-bench PRIVATE hipblas_fortran_client )
endif()

target_link_libraries( hipblas-bench PRIVATE ${BLAS_LIBRARY} roc::hipblas )

target_compile_definitions( hipblas-bench PRIVATE HIPBLAS_BENCH ROCM_USE_FLOAT16 )

if( NOT WIN32 )
    target_link_libraries( hipblas-bench PRIVATE lapack cblas )
    target_link_libraries( hipblas-bench PRIVATE stdc++fs)
endif()

# need mf16c flag for float->half convertion
target_compile_options( hipblas-bench PRIVATE -mf16c ) #-Wno-deprecated-declarations )

target_compile_definitions( hipblas-bench PRIVATE HIPBLAS_BENCH HIPBLAS_BFLOAT16_CLASS ROCM_USE_FLOAT16 )

if( NOT USE_CUDA )
  target_link_libraries( hipblas-bench PRIVATE hip::host )

  if( CUSTOM_TARGET )
    target_link_libraries( hipblas-bench PRIVATE hip::${CUSTOM_TARGET} )
  endif()

  if( CMAKE_CXX_COMPILER MATCHES ".*/hipcc$" )
    # hip-clang needs specific flag to turn on pthread and m
    target_link_libraries( hipblas-bench PRIVATE -lpthread -lm )

    if(BUILD_ADDRESS_SANITIZER)
      target_link_libraries( hipblas-bench PRIVATE -fuse-ld=lld -lgfortran )
    endif()
  endif()
else( )
  target_compile_definitions( hipblas-bench PRIVATE __HIP_PLATFORM_NVCC__ )

  target_include_directories( hipblas-bench
    PRIVATE
      $<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}>
  )

  target_link_libraries( hipblas-bench PRIVATE ${CUDA_LIBRARIES} Threads::Threads )
endif( )

set_target_properties( hipblas-bench PROPERTIES
  DEBUG_POSTFIX "-d"
  CXX_EXTENSIONS OFF
  RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging"
)

add_dependencies( hipblas-bench hipblas-common )

rocm_install(TARGETS hipblas-bench COMPONENT benchmarks)
