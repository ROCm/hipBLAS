# ########################################################################
# Copyright 2016-2021 Advanced Micro Devices, Inc.
# ########################################################################

# set( Boost_DEBUG ON )
set( Boost_USE_MULTITHREADED ON )
set( Boost_DETAILED_FAILURE_MSG ON )
set( Boost_ADDITIONAL_VERSIONS 1.64.0 1.64 )
set( Boost_USE_STATIC_LIBS OFF )
find_package( Boost COMPONENTS program_options )

if( NOT Boost_FOUND )
  message( STATUS "Dynamic boost libraries not found. Attempting to find static libraries " )
  set( Boost_USE_STATIC_LIBS ON )
  find_package( Boost COMPONENTS program_options )

  if( NOT Boost_FOUND )
    message( FATAL_ERROR "boost is a required dependency and is not found;  try adding boost path to CMAKE_PREFIX_PATH" )
  endif( )
endif( )

# Linking lapack library requires fortran flags
if(NOT WIN32)
    enable_language( Fortran )
endif()
find_package( cblas CONFIG REQUIRED PATHS ${LAPACK_DIR} )
if( NOT cblas_FOUND )
  message( FATAL_ERROR "cblas is a required dependency and is not found;  try adding cblas path to CMAKE_PREFIX_PATH" )
endif( )

if( NOT TARGET hipblas )
  find_package( hipblas CONFIG PATHS /opt/rocm/hipblas )

  if( NOT hipblas_FOUND )
    message( FATAL_ERROR "hipBLAS is a required dependency and is not found; try adding hipblas path to CMAKE_PREFIX_PATH")
  endif( )
endif( )

find_package( GTest REQUIRED )
include_directories(${GTEST_INCLUDE_DIRS})

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
  )
endif( )

if(LINK_BLIS)
  set( BLIS_CPP ../common/blis_interface.cpp )
endif()

set( hipblas_test_common
  ../common/utility.cpp
  ../common/cblas_interface.cpp
  ../common/norm.cpp
  ../common/unit.cpp
  ../common/near.cpp
  ../common/arg_check.cpp
  ../common/hipblas_datatype2string.cpp
  ../common/hipblas_template_specialization.cpp
  ${BLIS_CPP}
)



add_executable( hipblas-test ${hipblas_f90_source} ${hipblas_test_source} ${hipblas_solver_test_source} ${hipblas_test_common} )

if (NOT WIN32)
  set( BLIS_INCLUDE_DIR ${BUILD_DIR}/deps/blis/include/blis )
  set( BLIS_LIBRARY ${BUILD_DIR}/deps/blis/lib/libblis.so )
else()
  set( BLIS_INCLUDE_DIR ${BLIS_DIR}/include/blis CACHE PATH "Blis library include path" )
  find_library( BLIS_LIBRARY libblis
                  PATHS ${BLIS_DIR}/lib
                  REQUIRED
                  NO_DEFAULT_PATH
              )
endif()

target_include_directories( hipblas-test
  PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
)

set( THREADS_PREFER_PTHREAD_FLAG ON )
find_package( Threads REQUIRED )
target_link_libraries( hipblas-test PRIVATE Threads::Threads )

target_compile_definitions( hipblas-test PRIVATE GOOGLE_TEST )

# External header includes included as SYSTEM files
target_include_directories( hipblas-test
  SYSTEM PRIVATE
    $<BUILD_INTERFACE:${Boost_INCLUDE_DIRS}>
    $<BUILD_INTERFACE:${GTEST_INCLUDE_DIRS}>
    $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>
    $<BUILD_INTERFACE:${BLIS_INCLUDE_DIR}>
    ${ROCM_PATH}/hsa/include
)

if (NOT WIN32)
    target_link_libraries( hipblas-test PRIVATE hipblas_fortran_client roc::hipblas cblas lapack)
endif()

target_link_libraries( hipblas-test PRIVATE roc::hipblas cblas lapack ${GTEST_LIBRARIES} ${Boost_LIBRARIES} )

if(LINK_BLIS)
  target_link_libraries( hipblas-test PRIVATE ${BLIS_LIBRARY} )
endif()

# need mf16c flag for float->half convertion
target_compile_options( hipblas-test PRIVATE -mf16c )

if( NOT USE_CUDA )
  target_link_libraries( hipblas-test PRIVATE hip::host )

  if( CUSTOM_TARGET )
    target_link_libraries( hipblas-test PRIVATE hip::${CUSTOM_TARGET} )
  endif( )

  if( CMAKE_CXX_COMPILER MATCHES ".*/hcc$|.*/hipcc$" )
    # hip-clang needs specific flag to turn on pthread and m
    target_link_libraries( hipblas-test PRIVATE -lpthread -lm )
  endif()
else( )
  target_compile_definitions( hipblas-test PRIVATE __HIP_PLATFORM_NVCC__ )

  target_include_directories( hipblas-test
    PRIVATE
      $<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}>
  )

  target_link_libraries( hipblas-test PRIVATE ${CUDA_LIBRARIES} Threads::Threads )
endif( )

if (WIN32)
  # for now adding in all .dll as dependency chain is not cmake based on win32
  file( GLOB third_party_dlls
    LIST_DIRECTORIES OFF
    CONFIGURE_DEPENDS
    ${LAPACK_DIR}/bin/*.dll 
    ${BLIS_DIR}/lib/*.dll 
    ${HIP_DIR}/bin/*.dll
    ${HIP_DIR}/bin/hipinfo.exe
    ${HIPSDK_PATH}/rocblas/bin/rocblas.dll
    C:/Users/dmcniven/source/repos/hipblas/build/release/staging/hipblas.dll
    ${CMAKE_SOURCE_DIR}/rtest.*
    C:/Windows/System32/libomp140*.dll
  ) 
  foreach( file_i ${third_party_dlls})
    add_custom_command( TARGET hipblas-test POST_BUILD COMMAND ${CMAKE_COMMAND} ARGS -E copy ${file_i} ${PROJECT_BINARY_DIR}/staging/ )
  endforeach( file_i )
endif()

set_target_properties( hipblas-test PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARED_REQUIRED ON
    CXX_EXTENSIONS OFF
    IMPORT_PREFIX ""
    IMPORT_SUFFIX ".lib"
    LINKER_LANGUAGE CXX
    DEBUG_POSTFIX "-d"
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging" )
