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
if(NOT WIN32)
    enable_language( Fortran )
endif()

if( NOT TARGET hipblas )
  find_package( hipblas REQUIRED CONFIG PATHS /opt/rocm/hipblas )
endif( )

find_package( GTest REQUIRED )

set(hipblas_test_source
  hipblas_gtest_main.cpp
  auxiliary_gtest.cpp
  set_get_pointer_mode_gtest.cpp
  set_get_vector_gtest.cpp
  set_get_matrix_gtest.cpp
  set_get_atomics_mode_gtest.cpp
  blas1_gtest.cpp
  axpy_ex_gtest.cpp
  dot_ex_gtest.cpp
  nrm2_ex_gtest.cpp
  rot_ex_gtest.cpp
  scal_ex_gtest.cpp
  gbmv_gtest.cpp
  gbmv_batched_gtest.cpp
  gbmv_strided_batched_gtest.cpp
  gemv_gtest.cpp
  gemv_batched_gtest.cpp
  gemv_strided_batched_gtest.cpp
  ger_gtest.cpp
  hbmv_gtest.cpp
  hemv_gtest.cpp
  hemv_batched_gtest.cpp
  hemv_strided_batched_gtest.cpp
  her_gtest.cpp
  her2_gtest.cpp
  hpmv_gtest.cpp
  hpr_gtest.cpp
  hpr2_gtest.cpp
  sbmv_gtest.cpp
  spmv_gtest.cpp
  spr_gtest.cpp
  spr2_gtest.cpp
  symv_gtest.cpp
  syr_gtest.cpp
  syr2_gtest.cpp
  tbmv_gtest.cpp
  tbsv_gtest.cpp
  tpmv_gtest.cpp
  tpsv_gtest.cpp
  trmv_gtest.cpp
  trsv_gtest.cpp
  dgmm_gtest.cpp
  gemm_gtest.cpp
  gemm_ex_gtest.cpp
  gemm_strided_batched_gtest.cpp
  gemm_batched_gtest.cpp
  hemm_gtest.cpp
  geam_gtest.cpp
  herk_gtest.cpp
  her2k_gtest.cpp
  herkx_gtest.cpp
  symm_gtest.cpp
  syrk_gtest.cpp
  syr2k_gtest.cpp
  syrkx_gtest.cpp
  trsm_gtest.cpp
  trsm_ex_gtest.cpp
  trmm_gtest.cpp
  trtri_gtest.cpp
)

if( BUILD_WITH_SOLVER )
  set( hipblas_solver_test_source
    getrf_gtest.cpp
    getrf_batched_gtest.cpp
    getrf_strided_batched_gtest.cpp
    getrs_gtest.cpp
    getrs_batched_gtest.cpp
    getrs_strided_batched_gtest.cpp
    getri_batched_gtest.cpp
    geqrf_gtest.cpp
    geqrf_batched_gtest.cpp
    geqrf_strided_batched_gtest.cpp
    gels_gtest.cpp
    gels_batched_gtest.cpp
    gels_strided_batched_gtest.cpp
  )
endif( )

if(LINK_BLIS)
  set( BLIS_CPP ../common/blis_interface.cpp )
endif()

set( hipblas_test_common
  ../common/utility.cpp
  ../common/cblas_interface.cpp
  ../common/clients_common.cpp
  ../common/norm.cpp
  ../common/unit.cpp
  ../common/near.cpp
  ../common/arg_check.cpp
  ../common/argument_model.cpp
  ../common/hipblas_arguments.cpp
  ../common/hipblas_parse_data.cpp
  ../common/hipblas_datatype2string.cpp
  ../common/hipblas_template_specialization.cpp
  ${BLIS_CPP}
)

add_executable( hipblas-test ${hipblas_f90_source} ${hipblas_test_source} ${hipblas_solver_test_source} ${hipblas_test_common} )

if(LINK_BLIS)
    if (NOT WIN32)
      set( BLIS_INCLUDE_DIR ${BUILD_DIR}/deps/blis/include/blis )
      set( BLIS_LIBRARY ${BUILD_DIR}/deps/blis/lib/libblis.so )
    else( )
      set( BLIS_INCLUDE_DIR ${BLIS_DIR}/include/blis CACHE PATH "Blis library include path" )
      find_library( BLIS_LIBRARY libblis
                      PATHS ${BLIS_DIR}/lib
                      REQUIRED
                      NO_DEFAULT_PATH
                  )
    endif()
endif()

target_include_directories( hipblas-test
  PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
)




target_compile_definitions( hipblas-test PRIVATE GOOGLE_TEST )

# External header includes included as SYSTEM files
target_include_directories( hipblas-test
  SYSTEM PRIVATE
    $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>
    $<BUILD_INTERFACE:${CBLAS_INCLUDE_DIRS}>
    $<BUILD_INTERFACE:${BLAS_INCLUDE_DIR}>
    $<BUILD_INTERFACE:${BLIS_INCLUDE_DIR}>
    ${ROCM_PATH}/hsa/include
)

if (NOT WIN32)
    target_link_libraries( hipblas-test PRIVATE hipblas_fortran_client lapack cblas stdc++fs )
endif()

target_link_libraries( hipblas-test PRIVATE ${BLAS_LIBRARY} roc::hipblas GTest::GTest Threads::Threads )

# need mf16c flag for float->half convertion
target_compile_options( hipblas-test PRIVATE -mf16c ) # -Wno-deprecated-declarations )

target_compile_definitions( hipblas-test PRIVATE HIPBLAS_BFLOAT16_CLASS ROCM_USE_FLOAT16 )

if( NOT USE_CUDA )
  target_link_libraries( hipblas-test PRIVATE hip::host )

  if( CUSTOM_TARGET )
    target_link_libraries( hipblas-test PRIVATE hip::${CUSTOM_TARGET} )
  endif( )

  if( CMAKE_CXX_COMPILER MATCHES ".*/hipcc$" )
    # hip-clang needs specific flag to turn on pthread and m
    target_link_libraries( hipblas-test PRIVATE -lpthread -lm )

    if(BUILD_ADDRESS_SANITIZER)
      target_link_libraries( hipblas-test PRIVATE -fuse-ld=lld -lgfortran )
    endif()
  endif()
else( )
  target_compile_definitions( hipblas-test PRIVATE __HIP_PLATFORM_NVCC__ )

  target_include_directories( hipblas-test
    PRIVATE
      $<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}>
  )

  target_link_libraries( hipblas-test PRIVATE ${CUDA_LIBRARIES} )
endif( )

if (WIN32)
  # for now adding in all .dll as dependency chain is not cmake based on win32
  file( GLOB third_party_dlls
    LIST_DIRECTORIES OFF
    CONFIGURE_DEPENDS
    ${LAPACK_DIR}/bin/*.dll
    ${BLIS_DIR}/lib/*.dll
    ${OPENBLAS_DIR}/bin/*.dll
    ${HIP_DIR}/bin/*.dll
    ${HIP_DIR}/bin/hipinfo.exe
    ${ROCBLAS_PATH}/bin/rocblas.dll
    ${ROCSOLVER_PATH}/bin/rocsolver.dll
    ${CMAKE_SOURCE_DIR}/rtest.*
    C:/Windows/System32/libomp140*.dll
  )
  foreach( file_i ${third_party_dlls})
    add_custom_command( TARGET hipblas-test POST_BUILD COMMAND ${CMAKE_COMMAND} ARGS -E copy ${file_i} ${PROJECT_BINARY_DIR}/staging/ )
  endforeach( file_i )

  add_custom_command( TARGET hipblas-test POST_BUILD COMMAND ${CMAKE_COMMAND} ARGS -E copy_directory ${ROCBLAS_PATH}/bin/rocblas/library/ ${PROJECT_BINARY_DIR}/staging/library/)
endif()

set_target_properties( hipblas-test PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARED_REQUIRED ON
    CXX_EXTENSIONS OFF
    IMPORT_PREFIX ""
    IMPORT_SUFFIX ".lib"
    LINKER_LANGUAGE CXX
    DEBUG_POSTFIX "-d"
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging" )

rocm_install(TARGETS hipblas-test COMPONENT tests)
