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
prepend_path( ".." hipblas_headers_public relative_hipblas_headers_public )

if( NOT CUDA_FOUND )
  set( hipblas_source "${CMAKE_CURRENT_SOURCE_DIR}/hcc_detail/hipblas.cpp" )
else( )
  set( hipblas_source "${CMAKE_CURRENT_SOURCE_DIR}/nvcc_detail/hipblas.cpp" )
endif( )

add_library( hipblas
  ${hipblas_source}
  ${relative_hipblas_headers_public}
)
add_library( roc::hipblas ALIAS hipblas )

# External header includes included as system files
target_include_directories( hipblas
  SYSTEM PRIVATE
    $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>
)

# Test for specific compiler features if cmake version is recent enough
target_compile_features( hipblas PRIVATE cxx_static_assert cxx_nullptr cxx_auto_type )

# Build hipblas from source on AMD platform
if( NOT CUDA_FOUND )
  if( NOT TARGET rocblas )
    find_package( rocblas REQUIRED CONFIG PATHS /opt/rocm /opt/rocm/rocblas )
  endif( )

  target_compile_definitions( hipblas PRIVATE __HIP_PLATFORM_HCC__ )

  get_target_property( HIPHCC_LOCATION hip::hip_hcc IMPORTED_LOCATION_RELEASE )
  target_link_libraries( hipblas PRIVATE roc::rocblas ${HIPHCC_LOCATION} )

  # Remove following when hcc is fixed; hcc emits following spurious warning ROCm v1.6.3
  # "clang-5.0: warning: argument unused during compilation: '-isystem /opt/rocm/include'"
  if( CMAKE_CXX_COMPILER MATCHES ".*/hcc$|.*/hipcc$" )
    target_compile_options( hipblas PRIVATE -Wno-unused-command-line-argument )
  endif( )
else( )
  target_compile_definitions( hipblas PRIVATE __HIP_PLATFORM_NVCC__ )

  target_link_libraries( hipblas PRIVATE ${CUDA_CUBLAS_LIBRARIES} )

  # External header includes included as system files
  target_include_directories( hipblas
    SYSTEM PRIVATE
      $<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}>
  )
endif( )

# Internal header includes
target_include_directories( hipblas
  PUBLIC  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/library/include>
          $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
          $<INSTALL_INTERFACE:include>
)

set_target_properties( hipblas PROPERTIES VERSION ${hipblas_VERSION} SOVERSION ${hipblas_SOVERSION} CXX_EXTENSIONS NO )
set_target_properties( hipblas PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging" )
set_target_properties( hipblas PROPERTIES DEBUG_POSTFIX "-d" )

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
  TARGETS roc::hipblas
  PREFIX hipblas
  DEPENDS PACKAGE hip
  NAMESPACE roc::
 )

rocm_install_symlink_subdir( hipblas )
