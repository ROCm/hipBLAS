# ########################################################################
# Copyright 2016-2020 Advanced Micro Devices, Inc.
# ########################################################################

set( THREADS_PREFER_PTHREAD_FLAG ON )
find_package( Threads REQUIRED )

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
      ../common/hipblas_template_specialization.cpp
    )

add_executable( hipblas-bench client.cpp ${hipblas_benchmark_common} )


target_compile_features( hipblas-bench PRIVATE cxx_static_assert cxx_nullptr cxx_auto_type )

# Internal header includes
target_include_directories( hipblas-bench
  PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
)

# External header includes included as system files
target_include_directories( hipblas-bench
  SYSTEM PRIVATE
    $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>
)

target_link_libraries( hipblas-bench PRIVATE hipblas_fortran_client roc::hipblas cblas lapack)

# need mf16c flag for float->half convertion
target_compile_options( hipblas-bench PRIVATE -mf16c)

if( NOT CUDA_FOUND )
  target_compile_definitions( hipblas-bench PRIVATE __HIP_PLATFORM_HCC__ )

  if( CUSTOM_TARGET )
    target_link_libraries( hipblas-bench PRIVATE hip::${CUSTOM_TARGET} hip::host )
  else( )
    target_link_libraries( hipblas-bench PRIVATE hip::host )
  endif( )

  if( CMAKE_CXX_COMPILER MATCHES ".*/hcc$|.*/hipcc$" )
    # hip-clang needs specific flag to turn on pthread and m
    target_link_libraries( hipblas-bench PRIVATE -lpthread -lm )
  endif()
else( )
  target_compile_definitions( hipblas-bench PRIVATE __HIP_PLATFORM_NVCC__ )

  target_include_directories( hipblas-bench
    PRIVATE
      $<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}>
  )

  target_link_libraries( hipblas-bench PRIVATE ${CUDA_LIBRARIES} Threads::Threads )
endif( )

set_target_properties( hipblas-bench PROPERTIES DEBUG_POSTFIX "-d" CXX_EXTENSIONS NO )
set_target_properties( hipblas-bench PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging" )
#add_dependencies( hipblas-bench hipblas-bench-common )

target_compile_definitions( hipblas-bench PRIVATE HIPBLAS_BENCH ROCM_USE_FLOAT16 )
