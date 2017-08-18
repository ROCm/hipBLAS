# ************************************************************************
# Copyright 2016 Advanced Micro Devices, Inc.
# ************************************************************************

CMAKE_MINIMUM_REQUIRED (VERSION 2.8.8)
message(STATUS "CMAKE VERSION ${CMAKE_VERSION}")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/../../cmake ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(ROCM_PATH "/opt/rocm/")
message (STATUS "PROJECT NAME       : ${PROJECT_NAME}")

set(HIP_SUPPORT $ENV{HIP_SUPPORT})

# Find HIP
set (HIP_PATH $ENV{HIP_PATH})
if (NOT DEFINED HIP_PATH)
    set (HIP_PATH ${ROCM_PATH}/hip)
endif()
message (STATUS "HIP_PATH           : ${HIP_PATH}")

execute_process(COMMAND ${HIP_PATH}/bin/hipconfig --platform OUTPUT_VARIABLE PLATFORM)
message (STATUS "PLATFORM           : ${PLATFORM}")

# Build hipblas from source on AMD platform
if (${PLATFORM} MATCHES "hcc")
  message (STATUS "building for hcc")

# Find HCC compiler
  execute_process(COMMAND ${HCC_CONFIG} --install --cxxflags
                            OUTPUT_VARIABLE HCC_CXXFLAGS)
  execute_process(COMMAND ${HCC_CONFIG} --install --ldflags --shared
                            OUTPUT_VARIABLE HCC_LDFLAGS)

  string(STRIP "${HCC_CXXFLAGS}" HCC_CXXFLAGS)
  string(STRIP "${HCC_LDFLAGS}" HCC_LDFLAGS)

  set (HCC_CXXFLAGS "${HCC_CXXFLAGS} -I${CMAKE_CURRENT_SOURCE_DIR}/../include -I${ROCM_PATH}/include -I${HIP_PATH}/include -D__HIP_PLATFORM_HCC__= -fPIC")
  set (HCC_LDFLAGS "${HCC_LDFLAGS} -L${HIP_PATH}/lib -lhip_hcc -Wl,-rpath-link,${HIP_PATH}/lib -fPIC")
  message (STATUS "CMAKE_CXX_COMPILER : ${CMAKE_CXX_COMPILER}")
  message (STATUS "HCC_CXXFLAGS       : ${HCC_CXXFLAGS}")
  message (STATUS "HCC_LDFLAGS        : ${HCC_LDFLAGS}")

  set (HIPBLASSRCS ${CMAKE_CURRENT_SOURCE_DIR}/hcc_detail/hipblas.cpp)

    # Setting header dependencies for hipblas library generation
  foreach(src_file ${HIPBLASSRCS})
    set_property(SOURCE ${src_file} APPEND_STRING PROPERTY COMPILE_FLAGS " ${HCC_CXXFLAGS} ")
  endforeach()

  find_package( rocblas REQUIRED CONFIG )

  #Generating hipblas shared object
  add_library( hipblas ${HIPBLASSRCS} )
  set_property(TARGET hipblas APPEND_STRING PROPERTY LINK_FLAGS " ${HCC_LDFLAGS} ")
  target_link_libraries( hipblas hc_am roc::rocblas )
  set_target_properties( hipblas PROPERTIES DEBUG_POSTFIX "-d" OUTPUT_NAME hipblas-hcc )

elseif (${PLATFORM} MATCHES "nvcc")
  message(STATUS "Building for nvcc")
  find_package( HIP REQUIRED ) # Using module mode; make sure to set CMAKE_MODULE_PATH to /opt/rocm/hip/cmake
  find_package( CUDA REQUIRED ) # Using module mode; packaged in cmake

  set (HIPBLASSRCS ${CMAKE_CURRENT_SOURCE_DIR}/nvcc_detail/hipblas.cpp)

  hip_add_library( hipblas ${HIPBLASSRCS} )
  target_link_libraries( hipblas PRIVATE ${CUDA_CUBLAS_LIBRARIES} )

  target_include_directories( hipblas PRIVATE
     $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
	 $<BUILD_INTERFACE:${HIP_PATH}/include>
	 $<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}> )

  target_compile_definitions( hipblas PRIVATE __HIP_PLATFORM_NVCC__ )
  set_target_properties( hipblas PROPERTIES DEBUG_POSTFIX "-d" OUTPUT_NAME hipblas-nvcc )
endif() # (${PLATFORM} MATCHES "hcc" or "nvcc")

target_include_directories( hipblas
  PUBLIC  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
          $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
          $<INSTALL_INTERFACE:include>
)

set_target_properties( hipblas PROPERTIES VERSION ${hipblas_VERSION} SOVERSION ${hipblas_SOVERSION} )

# Package that helps me set visibility for function names exported from shared library
include( GenerateExportHeader )
if( CMAKE_VERSION VERSION_LESS "3.0" )
  add_compiler_export_flags( )
else()
  set_target_properties( hipblas PROPERTIES CXX_VISIBILITY_PRESET "hidden" VISIBILITY_INLINES_HIDDEN ON )
endif( )
GENERATE_EXPORT_HEADER( hipblas EXPORT_FILE_NAME hipblas-export.h )

install(TARGETS hipblas
  EXPORT hipblas-targets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ GROUP_EXECUTE GROUP_READ WORLD_EXECUTE WORLD_READ
)

# The following is cmake code to generate a config file package for hipblas
# Documentation for how to use the following package:
# https://cmake.org/cmake/help/v3.0/module/CMakePackageConfigHelpers.html
include( CMakePackageConfigHelpers )

set( LIB_INSTALL_DIR ${CMAKE_INSTALL_LIBDIR} )
set( INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_INCLUDEDIR} )
set( config_package_location ${LIB_INSTALL_DIR}/cmake/hipblas )

configure_package_config_file(
 hipblas-config.cmake.in
 ${CMAKE_CURRENT_BINARY_DIR}/hipblas-config.cmake
 INSTALL_DESTINATION ${config_package_location}
 PATH_VARS INCLUDE_INSTALL_DIR LIB_INSTALL_DIR
)

write_basic_package_version_file(
 ${CMAKE_CURRENT_BINARY_DIR}/hipblas-config-version.cmake
 VERSION ${hipblas_VERSION}
 COMPATIBILITY SameMajorVersion
)

install( EXPORT hipblas-targets
  DESTINATION ${config_package_location}
)

install( FILES
  ${CMAKE_CURRENT_BINARY_DIR}/hipblas-config.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/hipblas-config-version.cmake
  DESTINATION ${config_package_location}
  PERMISSIONS OWNER_WRITE OWNER_READ GROUP_READ WORLD_READ
)

install( FILES ${hipblas_headers_public}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    PERMISSIONS OWNER_WRITE OWNER_READ GROUP_READ WORLD_READ
)

install( FILES ${CMAKE_CURRENT_BINARY_DIR}/hipblas-export.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  PERMISSIONS OWNER_WRITE OWNER_READ GROUP_READ WORLD_READ
)
