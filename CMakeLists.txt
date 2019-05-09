# ########################################################################
# Copyright 2016 Advanced Micro Devices, Inc.
# ########################################################################

# The ROCm platform requires Ubuntu 16.04 or Fedora 24, which has cmake 3.5
cmake_minimum_required( VERSION 3.5 )

# Consider removing this in the future
# This should appear before the project command, because it does not use FORCE
if( WIN32 )
  set( CMAKE_INSTALL_PREFIX "${PROJECT_BINARY_DIR}/package" CACHE PATH "Install path prefix, prepended onto install directories" )
else( )
  set( CMAKE_INSTALL_PREFIX "/opt/rocm" CACHE PATH "Install path prefix, prepended onto install directories" )
endif( )

# This has to be initialized before the project() command appears
# Set the default of CMAKE_BUILD_TYPE to be release, unless user specifies with -D.  MSVC_IDE does not use CMAKE_BUILD_TYPE
if( NOT DEFINED CMAKE_CONFIGURATION_TYPES AND NOT DEFINED CMAKE_BUILD_TYPE )
  set( CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel." )
endif()

project( hipblas LANGUAGES CXX )

# This finds the rocm-cmake project, and installs it if not found
# rocm-cmake contains common cmake code for rocm projects to help setup and install
set( PROJECT_EXTERN_DIR ${CMAKE_CURRENT_BINARY_DIR}/extern )
find_package( ROCM CONFIG QUIET PATHS /opt/rocm )
if( NOT ROCM_FOUND )
  set( rocm_cmake_tag "master" CACHE STRING "rocm-cmake tag to download" )
  file( DOWNLOAD https://github.com/RadeonOpenCompute/rocm-cmake/archive/${rocm_cmake_tag}.zip
    ${PROJECT_EXTERN_DIR}/rocm-cmake-${rocm_cmake_tag}.zip )

  execute_process( COMMAND ${CMAKE_COMMAND} -E tar xzf ${PROJECT_EXTERN_DIR}/rocm-cmake-${rocm_cmake_tag}.zip
    WORKING_DIRECTORY ${PROJECT_EXTERN_DIR} )

  find_package( ROCM REQUIRED CONFIG PATHS ${PROJECT_EXTERN_DIR}/rocm-cmake-${rocm_cmake_tag} )
endif( )

include( ROCMSetupVersion )
include( ROCMCreatePackage )
include( ROCMInstallTargets )
include( ROCMPackageConfigHelpers )
include( ROCMInstallSymlinks )

rocm_setup_version( VERSION 0.12.5.0 NO_GIT_TAG_VERSION )

# Append our library helper cmake path and the cmake path for hip (for convenience)
# Users may override HIP path by specifying their own in CMAKE_MODULE_PATH
list( APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake )

# NOTE:  workaround until hcc & hip cmake modules fixes symlink logic in their config files; remove when fixed
list( APPEND CMAKE_PREFIX_PATH /opt/rocm/hcc /opt/rocm/hip )

# Building tensile can add significant compile time; this option allows to build
# library without tensile to allow for rapid iteration without GEMM functionality
option( BUILD_VERBOSE "Output additional build information" OFF )

# BUILD_SHARED_LIBS is a cmake built-in; we make it an explicit option such that it shows in cmake-gui
option( BUILD_SHARED_LIBS "Build hipBLAS as a shared library" ON )

# Find HCC/HIP dependencies
if( CMAKE_CXX_COMPILER MATCHES ".*/hcc$|.*/hipcc$" )
  message( STATUS "Building with ROCm tools" )
  find_package( hcc REQUIRED CONFIG PATHS /opt/rocm )
endif( )

# Hip headers required of all clients; clients use hip to allocate device memory
find_package( hip REQUIRED CONFIG PATHS /opt/rocm )

# Quietly look for CUDA, but if not found it's not an error
# The presense of hip is not sufficient to determine if we want a rocm or cuda backend
option(TRY_CUDA "Look for CUDA and use that as a backend if found" ON)
if(TRY_CUDA)
    find_package( CUDA QUIET )
endif()

# CMake list of machine targets
set( AMDGPU_TARGETS gfx803;gfx900 CACHE STRING "List of specific machine types for library to target" )

add_subdirectory( library )

include( clients/cmake/build-options.cmake )

# force library install path to lib (CentOS 7 defaults to lib64)
set(CMAKE_INSTALL_LIBDIR "lib" CACHE INTERNAL "Installation directory for libraries" FORCE)

# Build clients of the library
if( BUILD_CLIENTS_SAMPLES OR BUILD_CLIENTS_TESTS )
  add_subdirectory( clients )
endif( )
