# ########################################################################
# Copyright 2016 Advanced Micro Devices, Inc.
# ########################################################################

# ########################################################################
# A helper function to prefix a source list of files with a common path into a new list (non-destructive)
# ########################################################################
function( prepend_path prefix source_list_of_files return_list_of_files )
  foreach( file ${${source_list_of_files}} )
    if(IS_ABSOLUTE ${file} )
      list( APPEND new_list ${file} )
    else( )
      list( APPEND new_list ${prefix}/${file} )
    endif( )
  endforeach( )
  set( ${return_list_of_files} ${new_list} PARENT_SCOPE )
endfunction( )

# ########################################################################
# Main
# ########################################################################

# Build hipblas from source on AMD platform
if( CMAKE_CXX_COMPILER MATCHES ".*/hcc$" )
  prepend_path( ".." hipblas_headers_public relative_hipblas_headers_public )

  add_library( hipblas
    ${CMAKE_CURRENT_SOURCE_DIR}/hcc_detail/hipblas.cpp
    ${relative_hipblas_headers_public}
  )

  find_package( rocblas REQUIRED CONFIG )

  # NOTE: hipblas header 'hipblas-types.h' exposes the hip header hip/hip_vector_types.h, requiring exposing hip::hip_hcc as PUBLIC
  target_link_libraries( hipblas PUBLIC hip::hip_hcc PRIVATE hip::hip_device hcc::hccshared roc::rocblas )

  # Test for specific compiler features if cmake version is recent enough
  if( CMAKE_VERSION VERSION_GREATER "3.0" )
    target_compile_features( hipblas PRIVATE cxx_static_assert cxx_nullptr cxx_auto_type )
  else( )
    # Otherwise, just try to compile the library with a standards flag
    if( CMAKE_COMPILER_IS_GNUCXX OR ( CMAKE_CXX_COMPILER_ID MATCHES "Clang" ) )
      target_compile_options( hipblas PRIVATE -std=c++11 )
    endif( )
  endif( )

  # Remove following when hcc is fixed; hcc emits following spurious warning ROCm v1.6.3
  # "clang-5.0: warning: argument unused during compilation: '-isystem /opt/rocm/include'"
  target_compile_options( hipblas PRIVATE -Wno-unused-command-line-argument )

  foreach( target ${AMDGPU_TARGETS} )
    target_link_libraries( hipblas PRIVATE --amdgpu-target=${target} )
  endforeach( )

  set_target_properties( hipblas PROPERTIES DEBUG_POSTFIX "-d" OUTPUT_NAME hipblas-hcc )
else( )
  # If HiP is not installed in default path, user can append to CMAKE_MODULE_PATH
  list( APPEND CMAKE_MODULE_PATH /opt/rocm/hip/cmake )
  find_package( HIP REQUIRED ) # Using module mode;
  find_package( CUDA REQUIRED ) # Using module mode; packaged in cmake

  hip_add_library( hipblas
    ${CMAKE_CURRENT_SOURCE_DIR}/nvcc_detail/hipblas.cpp
    ${relative_hipblas_headers_public}
  )

  target_link_libraries( hipblas PRIVATE ${CUDA_CUBLAS_LIBRARIES} )

  target_include_directories( hipblas PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
  $<BUILD_INTERFACE:${HIP_PATH}/include>
  $<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}> )

  target_compile_definitions( hipblas PRIVATE __HIP_PLATFORM_NVCC__ )
  set_target_properties( hipblas PROPERTIES DEBUG_POSTFIX "-d" OUTPUT_NAME hipblas-nvcc )
endif( )

target_include_directories( hipblas
PUBLIC  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/library/include>
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

set_target_properties( hipblas PROPERTIES VERSION ${hipblas_VERSION} SOVERSION ${hipblas_SOVERSION} CXX_EXTENSIONS NO )
set_target_properties( hipblas PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging" )

# Package that helps me set visibility for function names exported from shared library
include( GenerateExportHeader )
set_target_properties( hipblas PROPERTIES CXX_VISIBILITY_PRESET "hidden" VISIBILITY_INLINES_HIDDEN ON )
generate_export_header( hipblas EXPORT_FILE_NAME ${PROJECT_BINARY_DIR}/include/hipblas-export.h )

# Following Boost conventions of prefixing 'lib' on static built libraries, across all platforms
if( NOT BUILD_SHARED_LIBS )
  set_target_properties( hipblas PROPERTIES PREFIX "lib" )
endif( )

############################################################
# Installation

rocm_install_targets(
  TARGETS hipblas
  INCLUDE
    ${CMAKE_SOURCE_DIR}/library/include
    ${CMAKE_BINARY_DIR}/include
  PREFIX hipblas
)
#         PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ GROUP_EXECUTE GROUP_READ WORLD_EXECUTE WORLD_READ

rocm_export_targets(
  TARGETS hipblas-targets
  PREFIX hipblas
  DEPENDS PACKAGE hip
  NAMESPACE roc::
 )

rocm_install_symlink_subdir( hipblas )