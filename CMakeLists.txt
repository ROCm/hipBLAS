# ########################################################################
# Copyright 2016 Advanced Micro Devices, Inc.
# ########################################################################

# Natively available on including Ubuntu 14.04, OpenSUSE 13.2, CentOS 6.6
cmake_minimum_required( VERSION 2.8.12 )

# On windows, it's convenient to change the default install prefix such that it does NOT point to 'program files'
if( WIN32 AND CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT )
  set( CMAKE_INSTALL_PREFIX "${PROJECT_BINARY_DIR}/library-package" CACHE PATH "Install path prefix, prepended onto install directories" FORCE )
endif( )

# This has to be initialized before the project() command appears
# Set the default of CMAKE_BUILD_TYPE to be release, unless user specifies with -D.  MSVC_IDE does not use CMAKE_BUILD_TYPE
if( NOT DEFINED CMAKE_CONFIGURATION_TYPES AND NOT DEFINED CMAKE_BUILD_TYPE )
  set( CMAKE_BUILD_TYPE "Release" CACHE STRINGS "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel.")
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS None Debug Release RelWithDebInfo MinSizeRel)
endif()

# Append our library helper cmake path and the cmake path for hip (for convenience)
# Users may override HIP path by specifying their own in CMAKE_MODULE_PATH
list( APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake )

# # Temp fix, there is a problem building our test program using googletest with gcc
# # Change compiler default to clang
# if( NOT DEFINED CMAKE_CXX_COMPILER AND NOT DEFINED ENV{CXX} )
#   set( CMAKE_CXX_COMPILER clang++ )
# endif( )
# if( NOT DEFINED CMAKE_C_COMPILER AND NOT DEFINED ENV{CC} )
#   set( CMAKE_C_COMPILER clang )
# endif( )

# The superbuild does not build anything itself, all compiling is done in external projects
project( hipblas-superbuild NONE )

# Everything is initially off, so that cache is not initialized until user elects to build
option( BUILD_LIBRARY "Build hipBLAS library" OFF )


# BUILD_SHARED_LIBS is a cmake built-in; we make it an explicit option such that it shows in cmake-gui
option( BUILD_SHARED_LIBS "Build hipBLAS as a shared library" OFF )

# set( hipblas_INSTALL_DIR ${CMAKE_INSTALL_PREFIX} )
set( hipblas_INSTALL_DIR "${PROJECT_BINARY_DIR}/library-package" )

# Default behavior is to NOT install library, but clients may overload
set( hipblas_INSTALL_COMMAND INSTALL_COMMAND ${CMAKE_COMMAND} -E echo_append )

# standard cmake module to manage dependencies
include( ExternalProject )

# This captures all of the dependencies cmake builds itself
set( hipblas_dependencies )

set( BASE_CMAKE_ARGS )

message( STATUS "CMAKE_BUILD_TYPE         : ${CMAKE_BUILD_TYPE} " )
message( STATUS "BUILD_LIBRARY            : ${BUILD_LIBRARY} " )
message( STATUS "BUILD_SHARED_LIBS        : ${BUILD_SHARED_LIBS} " )
message( STATUS "hipblas_INSTALL_DIR      : ${hipblas_INSTALL_DIR} " )
message( STATUS "PROJECT_BINARY_DIR       : ${PROJECT_BINARY_DIR} " )
message( STATUS "CMAKE_CURRENT_SOURCE_DIR : ${CMAKE_CURRENT_SOURCE_DIR} " )

# # Noramlize the different ways of specifying a c++ compiler through -DCMAKE_CXX_COMPILER
# if( DEFINED CMAKE_CXX_COMPILER )
#   message( STATUS "CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER} " )
#   list( APPEND BASE_CMAKE_ARGS -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER} )
# elseif( DEFINED ENV{CXX} )
#   message( STATUS "ENV{CXX}: $ENV{CXX} " )
#   list( APPEND BASE_CMAKE_ARGS -DCMAKE_CXX_COMPILER=$ENV{CXX} )
# endif( )

# # Noramlize the different ways of specifying a c compiler through -DCMAKE_C_COMPILER
# if( DEFINED CMAKE_C_COMPILER )
#   message( STATUS "CMAKE_C_COMPILER: ${CMAKE_C_COMPILER}" )
#   list( APPEND BASE_CMAKE_ARGS -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER} )
# elseif( DEFINED ENV{CC} )
#   message( STATUS "ENV{CC}: $ENV{CC} " )
#   list( APPEND BASE_CMAKE_ARGS -DCMAKE_C_COMPILER=$ENV{CC} )
# endif( )

# if( DEFINED CMAKE_CXX_FLAGS )
#   message( STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS} " )
#   list( APPEND BASE_CMAKE_ARGS -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS} )
# endif( )

# if( DEFINED CMAKE_C_FLAGS )
#   message( STATUS "CMAKE_C_FLAGS: ${CMAKE_C_FLAGS}" )
#   list( APPEND BASE_CMAKE_ARGS -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS} )
# endif( )

# CMAKE_BUILD_TYPE only applies to single configuration build systems
# if( DEFINED CMAKE_BUILD_TYPE )
#   list( APPEND BASE_CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} )
# endif( )

# if( NOT DEFINED DEVICE_CXX_COMPILER )
#   find_package( HIP REQUIRED )
#   set( DEVICE_CXX_COMPILER ${HIP_ROOT_DIR}/bin/hipcc )
# endif()

if( BUILD_LIBRARY )
  # WARNING: do not surround CMAKE_PREFIX_PATH with quotes, it breaks
  # Replace all occurances of ; with ^^, which we elect to use a path seperator
  # string(REGEX REPLACE ";" "^^" LIBRARY_PREFIX_PATH "${CMAKE_PREFIX_PATH}" )
  # string(REGEX REPLACE ";" "^^" LIBRARY_MODULE_PATH "${CMAKE_MODULE_PATH}" )

  # execute_process(COMMAND ${HIP_ROOT_DIR}/bin/hipconfig --platform OUTPUT_VARIABLE PLATFORM)
  # message (STATUS "PLATFORM                 : ${PLATFORM}")

  # IF (${PLATFORM} MATCHES "hcc")
  #   set( LIBRARY_CXX_COMPILER ${DEVICE_CXX_COMPILER})
  # ELSEIF (${PLATFORM} MATCHES "nvcc")
  #   set( LIBRARY_CXX_COMPILER ${CMAKE_CXX_COMPILER})
  # ENDIF()

    # -DCMAKE_PREFIX_PATH=${LIBRARY_PREFIX_PATH}
    # -DCMAKE_MODULE_PATH=${LIBRARY_MODULE_PATH}
  set( LIBRARY_CMAKE_ARGS
    -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
    -DCMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}
    -DCMAKE_MODULE_PATH=${CMAKE_MODULE_PATH}
    -DBUILD_SHARED_LIBS=${BUILD_SHARED_LIBS}
    # -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
    # -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
    # -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
  )

  # Build the library as an external project
  ExternalProject_Add( hipblas
    DEPENDS ${hipblas_dependencies}
    SOURCE_DIR ${PROJECT_SOURCE_DIR}/library
    BINARY_DIR library-build
    INSTALL_DIR library-package
    LIST_SEPARATOR ^^
    CMAKE_ARGS ${LIBRARY_CMAKE_ARGS}
    ${hipblas_INSTALL_COMMAND}
  )
endif( )

