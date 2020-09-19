# ########################################################################
# Copyright 2016-2020 Advanced Micro Devices, Inc.
# ########################################################################

# This is incremented when the ABI to the library changes
set( hipblas_SOVERSION 0.1 )

list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake )

# This option only works for make/nmake and the ninja generators, but no reason it shouldn't be on all the time
# This tells cmake to create a compile_commands.json file that can be used with clang tooling or vim
set( CMAKE_EXPORT_COMPILE_COMMANDS ON )

# Print out compiler flags for viewing/debug
if( BUILD_VERBOSE )
  message( STATUS "hipblas_VERSION: ${hipblas_VERSION}" )
  message( STATUS "\t==>CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}" )
  message( STATUS "\t==>BUILD_SHARED_LIBS: ${BUILD_SHARED_LIBS}" )
  message( STATUS "\t==>CMAKE_INSTALL_PREFIX link: " ${CMAKE_INSTALL_PREFIX} )
  message( STATUS "\t==>CMAKE_MODULE_PATH link: " ${CMAKE_MODULE_PATH} )
  message( STATUS "\t==>CMAKE_PREFIX_PATH link: " ${CMAKE_PREFIX_PATH} )

  message( STATUS "\t==>CMAKE_CXX_COMPILER flags: " ${CMAKE_CXX_FLAGS} )
  message( STATUS "\t==>CMAKE_CXX_COMPILER debug flags: " ${CMAKE_CXX_FLAGS_DEBUG} )
  message( STATUS "\t==>CMAKE_CXX_COMPILER release flags: " ${CMAKE_CXX_FLAGS_RELEASE} )
  message( STATUS "\t==>CMAKE_CXX_COMPILER relwithdebinfo flags: " ${CMAKE_CXX_FLAGS_RELWITHDEBINFO} )
  message( STATUS "\t==>CMAKE_EXE_LINKER link flags: " ${CMAKE_EXE_LINKER_FLAGS} )
endif( )

find_package( HIP REQUIRED )

# configure a header file to pass the CMake version settings to the source, and package the header files in the output archive
configure_file( "${CMAKE_CURRENT_SOURCE_DIR}/include/hipblas-version.h.in" "${PROJECT_BINARY_DIR}/include/hipblas-version.h" )

set( hipblas_headers_public
  include/hipblas.h
  ${PROJECT_BINARY_DIR}/include/hipblas-version.h
)

source_group( "Header Files\\Public" FILES ${hipblas_headers_public} )

include( GNUInstallDirs )

set( BIN_INSTALL_DIR ${CMAKE_INSTALL_BINDIR} )
set( LIB_INSTALL_DIR ${CMAKE_INSTALL_LIBDIR} )
set( INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_INCLUDEDIR} )

set(CMAKE_Fortran_MODULE_DIRECTORY ${PROJECT_BINARY_DIR}/include)

# Build into subdirectories
add_subdirectory( src )

# The following code is setting variables to control the behavior of CPack to generate our
# if( WIN32 )
#     set( CPACK_SOURCE_GENERATOR "ZIP" )
#     set( CPACK_GENERATOR "ZIP" )
# else( )
#     set( CPACK_SOURCE_GENERATOR "TGZ" )
#     set( CPACK_GENERATOR "DEB;RPM" CACHE STRING "cpack list: 7Z, DEB, IFW, NSIS, NSIS64, RPM, STGZ, TBZ2, TGZ, TXZ, TZ, ZIP" )
# endif( )

# Package specific CPACK vars
set( CPACK_DEBIAN_PACKAGE_DEPENDS "rocblas (>= 0.10.3)" )
set( CPACK_RPM_PACKAGE_REQUIRES "rocblas >= 0.10.3" )

set( CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/../LICENSE.md" )

if( NOT CPACK_PACKAGING_INSTALL_PREFIX )
  set( CPACK_PACKAGING_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}" )
endif( )

set( CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION "\${CPACK_PACKAGING_INSTALL_PREFIX}" "\${CPACK_PACKAGING_INSTALL_PREFIX}/include" "\${CPACK_PACKAGING_INSTALL_PREFIX}/lib" )

# Give hipblas compiled for CUDA backend a different name
if( NOT CUDA_FOUND )
    set( package_name hipblas )
else( )
    set( package_name hipblas-alt )
endif( )

set( HIPBLAS_CONFIG_DIR "\${CPACK_PACKAGING_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}" CACHE PATH "Path placed into ldconfig file" )

rocm_create_package(
    NAME ${package_name}
    DESCRIPTION "Radeon Open Compute BLAS marshalling library"
    MAINTAINER "hipBLAS Maintainer <hipblas-maintainer@amd.com>"
    LDCONFIG
    LDCONFIG_DIR ${HIPBLAS_CONFIG_DIR}
)
