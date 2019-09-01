# ########################################################################
# Copyright 2016 Advanced Micro Devices, Inc.
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
enable_language( Fortran )
find_package( cblas CONFIG REQUIRED )
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

set(hipblas_test_source
  hipblas_gtest_main.cpp
  set_get_pointer_mode_gtest.cpp
  set_get_vector_gtest.cpp
  set_get_matrix_gtest.cpp
  blas1_gtest.cpp
  gemv_gtest.cpp
  ger_gtest.cpp
  gemm_gtest.cpp
  gemm_ex_gtest.cpp
  gemm_strided_batched_gtest.cpp
  gemm_batched_gtest.cpp
  geam_gtest.cpp
  trsm_gtest.cpp
)

set( hipblas_benchmark_common
  ../common/utility.cpp
  ../common/cblas_interface.cpp
  ../common/flops.cpp
  ../common/norm.cpp
  ../common/unit.cpp
  ../common/arg_check.cpp
  ../common/hipblas_template_specialization.cpp
)

add_executable( hipblas-test ${hipblas_test_source} ${hipblas_benchmark_common} )

target_include_directories( hipblas-test
  PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
)

set( THREADS_PREFER_PTHREAD_FLAG ON )
find_package( Threads REQUIRED )
target_link_libraries( hipblas-test PRIVATE Threads::Threads )
target_compile_features( hipblas-test PRIVATE cxx_static_assert cxx_nullptr cxx_auto_type)

target_compile_definitions( hipblas-test PRIVATE GOOGLE_TEST )

# External header includes included as SYSTEM files
target_include_directories( hipblas-test
  SYSTEM PRIVATE
    $<BUILD_INTERFACE:${Boost_INCLUDE_DIRS}>
    $<BUILD_INTERFACE:${GTEST_INCLUDE_DIRS}>
    $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>
    /opt/rocm/hsa/include
)

target_link_libraries( hipblas-test PRIVATE roc::hipblas cblas lapack ${GTEST_LIBRARIES} ${Boost_LIBRARIES} )

if( NOT CUDA_FOUND )
  target_compile_definitions( hipblas-test PRIVATE __HIP_PLATFORM_HCC__ )

  get_target_property( HIP_HCC_LOCATION hip::hip_hcc IMPORTED_LOCATION_RELEASE )
  target_link_libraries( hipblas-test PRIVATE ${HIP_HCC_LOCATION} )


  if( CMAKE_CXX_COMPILER MATCHES ".*/hcc$|.*/hipcc$" )
    # Remove following when hcc is fixed; hcc emits following spurious warning
    # "clang-5.0: warning: argument unused during compilation: '-isystem /opt/rocm/include'"
    target_compile_options( hipblas-test PRIVATE -Wno-unused-command-line-argument -mf16c)

  elseif( CMAKE_COMPILER_IS_GNUCXX )
    # GCC needs specific flag to turn on f16c intrinsics
    target_compile_options( hipblas-test PRIVATE -mf16c )

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

  target_link_libraries( hipblas-test PRIVATE ${CUDA_LIBRARIES} )
endif( )

set_target_properties( hipblas-test PROPERTIES DEBUG_POSTFIX "-d" CXX_EXTENSIONS NO )
set_target_properties( hipblas-test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging" )
