# ########################################################################
# Copyright 2016 Advanced Micro Devices, Inc.
# ########################################################################

set( hipblas_samples_common ../common/utility.cpp )

add_executable( example-sscal example_sscal.cpp ${hipblas_samples_common} )
add_executable( example-sgemm example_sgemm.cpp ${hipblas_samples_common} )
add_executable( example-sgemm-strided-batched example_sgemm_strided_batched.cpp ${hipblas_samples_common} )

if( NOT TARGET hipblas )
  find_package( hipblas CONFIG PATHS /opt/rocm/hipblas )

  if( NOT hipblas_FOUND )
    message( FATAL_ERROR "hipBLAS is a required dependency and is not found; try adding rocblas path to CMAKE_PREFIX_PATH")
  endif( )
endif( )

foreach( exe example-sscal;example-sgemm;example-sgemm-strided-batched )

  target_link_libraries( ${exe} PRIVATE roc::hipblas )

  # External header includes included as SYSTEM files
  target_include_directories( ${exe}
    SYSTEM PRIVATE
      $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>
      /opt/rocm/hsa/include
  )

  # Try to test for specific compiler features if cmake version is recent enough
  target_compile_features( ${exe} PRIVATE cxx_static_assert cxx_nullptr cxx_auto_type )

  set_target_properties( ${exe} PROPERTIES DEBUG_POSTFIX "-d" CXX_EXTENSIONS NO )
  set_target_properties( ${exe} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging" )
  target_include_directories( ${exe} PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include> )

if( NOT CUDA_FOUND )
  target_compile_definitions( ${exe} PRIVATE __HIP_PLATFORM_HCC__ )

  get_target_property( HIP_HCC_LOCATION hip::hip_hcc IMPORTED_LOCATION_RELEASE )
  target_link_libraries( ${exe} PRIVATE ${HIP_HCC_LOCATION} )

  if( CMAKE_CXX_COMPILER MATCHES ".*/hcc$|.*/hipcc$" )
    # Remove following when hcc is fixed; hcc emits following spurious warning
    # "clang-5.0: warning: argument unused during compilation: '-isystem /opt/rocm/include'"
    target_compile_options( ${exe} PRIVATE -Wno-unused-command-line-argument )
  endif( )
else( )
  target_compile_definitions( ${exe} PRIVATE __HIP_PLATFORM_NVCC__ )

  target_include_directories( ${exe}
    PRIVATE
      $<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}>
  )

  target_link_libraries( ${exe} PRIVATE ${CUDA_LIBRARIES} )
endif( )

endforeach( )

