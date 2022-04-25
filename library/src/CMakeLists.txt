# ########################################################################
# Copyright 2016-2022 Advanced Micro Devices, Inc. All rights reserved.
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

if( NOT USE_CUDA )
  set( hipblas_source "${CMAKE_CURRENT_SOURCE_DIR}/hcc_detail/hipblas.cpp" )
else( )
  set( hipblas_source "${CMAKE_CURRENT_SOURCE_DIR}/nvcc_detail/hipblas.cpp" )
endif( )

set (hipblas_f90_source
  hipblas_module.f90
)

# Create hipBLAS Fortran module
if(NOT WIN32)
    add_library(hipblas_fortran ${hipblas_f90_source})
endif()

if(BUILD_ADDRESS_SANITIZER)
    add_link_options(-fuse-ld=lld)
endif()

add_library( hipblas
  ${hipblas_source}
  ${CMAKE_CURRENT_SOURCE_DIR}/hipblas_auxiliary.cpp
  ${relative_hipblas_headers_public}
)
add_library( roc::hipblas ALIAS hipblas )

# External header includes included as system files
target_include_directories( hipblas
  SYSTEM PRIVATE
    $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>
)

# Build hipblas from source on AMD platform
if( NOT USE_CUDA )
  if( NOT TARGET rocblas )
    if( CUSTOM_ROCBLAS )
      set ( ENV{rocblas_DIR} ${CUSTOM_ROCBLAS})
      find_package( rocblas REQUIRED CONFIG NO_CMAKE_PATH )
    elseif( WIN32 )
        find_package( rocblas REQUIRED CONFIG PATHS ${ROCBLAS_PATH})
    else()
      find_package( rocblas REQUIRED CONFIG PATHS /opt/rocm /opt/rocm/rocblas )
    endif( )
  endif( )

  target_link_libraries( hipblas PRIVATE roc::rocblas hip::host )

  # Add rocSOLVER as a dependency if BUILD_WITH_SOLVER is on
  if( BUILD_WITH_SOLVER )
    if( NOT TARGET rocsolver )
      if( CUSTOM_ROCSOLVER)
        set ( ENV{rocsolver_DIR} ${CUSTOM_ROCSOLVER})
        find_package( rocsolver REQUIRED CONFIG NO_CMAKE_PATH )
      elseif(WIN32)
        find_package( rocsolver REQUIRED CONFIG PATHS ${ROCSOLVER_PATH} )
      else()
        find_package( rocsolver REQUIRED CONFIG PATHS /opt/rocm /opt/rocm/rocsolver /usr/local/rocsolver )
      endif()
    endif( )
    target_link_libraries( hipblas PRIVATE roc::rocsolver )
  endif( )

  if( CUSTOM_TARGET )
    target_link_libraries( hipblas PRIVATE hip::${CUSTOM_TARGET} )
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
          $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include/hipblas>
          $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
          $<INSTALL_INTERFACE:include>
  PRIVATE
          ${CMAKE_CURRENT_SOURCE_DIR}/include
          ${CMAKE_CURRENT_SOURCE_DIR}
)

rocm_set_soversion( hipblas ${hipblas_SOVERSION} )
set_target_properties( hipblas PROPERTIES CXX_EXTENSIONS NO )
set_target_properties( hipblas PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging" )
set_target_properties( hipblas PROPERTIES DEBUG_POSTFIX "-d" )

if (WIN32)
  add_custom_command( TARGET hipblas POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_BINARY_DIR}/staging/$<TARGET_FILE_NAME:hipblas> ${PROJECT_BINARY_DIR}/clients/staging/$<TARGET_FILE_NAME:hipblas> )
  if( ${CMAKE_BUILD_TYPE} MATCHES "Debug")
    add_custom_command( TARGET hipblas POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_BINARY_DIR}/staging/hipblas-d.pdb ${PROJECT_BINARY_DIR}/clients/staging/hipblas-d.pdb )
  endif()
endif()

# Package that helps me set visibility for function names exported from shared library
include( GenerateExportHeader )
set_target_properties( hipblas PROPERTIES CXX_VISIBILITY_PRESET "hidden" VISIBILITY_INLINES_HIDDEN ON )
generate_export_header( hipblas EXPORT_FILE_NAME ${PROJECT_BINARY_DIR}/include/hipblas/hipblas-export.h )

if (BUILD_FILE_REORG_BACKWARD_COMPATIBILITY)
  rocm_wrap_header_file(
    hipblas-version.h hipblas-export.h
    GUARDS SYMLINK WRAPPER
    WRAPPER_LOCATIONS include hipblas/include
  )
endif( )


# Following Boost conventions of prefixing 'lib' on static built libraries, across all platforms
if( NOT BUILD_SHARED_LIBS )
  set_target_properties( hipblas PROPERTIES PREFIX "lib" )
endif( )

############################################################
# Installation

rocm_install_targets(
  TARGETS hipblas
  INCLUDE
    ${CMAKE_BINARY_DIR}/include
)
#         PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ GROUP_EXECUTE GROUP_READ WORLD_EXECUTE WORLD_READ

if ( NOT USE_CUDA )
    rocm_export_targets(
        TARGETS roc::hipblas
	DEPENDS PACKAGE hip
	NAMESPACE roc::
    )
else( )
    rocm_export_targets(
        TARGETS roc::hipblas
	DEPENDS PACKAGE HIP
	NAMESPACE roc::
    )
endif( )

# Force installation of .f90 module files
rocm_install(FILES "hipblas_module.f90"
        DESTINATION "include/hipblas")

if(BUILD_FILE_REORG_BACKWARD_COMPATIBILITY)
  rocm_install(
    DIRECTORY
       "${PROJECT_BINARY_DIR}/hipblas"
        DESTINATION "." )

  if ( NOT WIN32 )

    #Create SymLink for Fortran Object Module for backward compatibility
    rocm_install(
      CODE "
        set(PREFIX \$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX})
	set(INPUT_FILE \${PREFIX}/include/hipblas/hipblas_module.f90)
        set(SYMLINK_LOCATIONS \${PREFIX}/hipblas/include \${PREFIX}/include)
        foreach(LOCATION IN LISTS SYMLINK_LOCATIONS)
          file(MAKE_DIRECTORY \${LOCATION})
          execute_process(COMMAND ln -sfr \${INPUT_FILE} \${LOCATION})
          message(STATUS \"Created symlink in \${LOCATION} to \${INPUT_FILE}.\")
        endforeach()
        "
    )
  endif() #NOT WIN32
  message( STATUS "Backward Compatible Sym Link Created for include directories" )
endif()

