# ************************************************************************
# Copyright 2016 Advanced Micro Devices, Inc.
# ************************************************************************

CMAKE_MINIMUM_REQUIRED (VERSION 2.8.8) 
message(STATUS "CMAKE VERSION ${CMAKE_VERSION}")

# The following is cmake code to generate a config file package for hipblas
# Documentation for how to use the following package:
# https://cmake.org/cmake/help/v3.0/module/CMakePackageConfigHelpers.html
#include( CMakePackageConfigHelpers )

#set( LIB_INSTALL_DIR ${CMAKE_INSTALL_LIBDIR} )
#set( INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_INCLUDEDIR} )
#set( config_package_location ${LIB_INSTALL_DIR}/cmake/hipblas )

#configure_package_config_file(
#  hipblas-config.cmake.in
#  ${CMAKE_CURRENT_BINARY_DIR}/hipblas-config.cmake
#  INSTALL_DESTINATION ${config_package_location}
#  PATH_VARS INCLUDE_INSTALL_DIR LIB_INSTALL_DIR
#)
#
#write_basic_package_version_file(
#  ${CMAKE_CURRENT_BINARY_DIR}/hipblas-config-version.cmake
#  VERSION ${hipblas_VERSION}
#  COMPATIBILITY SameMajorVersion
#)


set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/../../cmake ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

#Setting a Project name
set(PROJECT_NAME "hipblas")
set(ROCM_PATH "/opt/rocm/")

project (${PROJECT_NAME})
message (STATUS "PROJECT NAME       : ${PROJECT_NAME}")

find_package( rocblas REQUIRED CONFIG )

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
# FIND_PACKAGE(HC++ 1.0 REQUIRED)

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
          set_PROPERTY(SOURCE ${src_file} APPEND_STRING PROPERTY COMPILE_FLAGS " ${HCC_CXXFLAGS} ")
  endforeach()
    
  #Generating hipblas shared object
  add_library("${PROJECT_NAME}-hcc" ${HIPBLASSRCS})
  set_PROPERTY(TARGET "${PROJECT_NAME}-hcc" APPEND_STRING PROPERTY LINK_FLAGS " ${HCC_LDFLAGS} ")
  target_link_libraries("${PROJECT_NAME}-hcc" hc_am rocblas )

  install(TARGETS "${PROJECT_NAME}-hcc" 
        RUNTIME DESTINATION lib
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        PERMISSIONS WORLD_READ WORLD_WRITE WORLD_EXECUTE
  )
  
elseif (${PLATFORM} MATCHES "nvcc")

  message(STATUS "Building hipblas.cpp")
  
  set (CXXFLAGS "-I${CMAKE_CURRENT_SOURCE_DIR}/../include -I${HIP_PATH}/include -I/usr/local/cuda/include -D__HIP_PLATFORM_NVCC__=")
  set (LDFLAGS "-L/usr/local/cuda/lib64 -L/usr/local/cuda/lib")
  set (HIPBLASSRCS ${CMAKE_CURRENT_SOURCE_DIR}/nvcc_detail/hipblas.cpp)
  
  foreach(src_file ${HIPBLASSRCS})
	  set_PROPERTY(SOURCE ${src_file} APPEND_STRING PROPERTY COMPILE_FLAGS " ${CXXFLAGS} ")
  endforeach()

  ADD_LIBRARY("${PROJECT_NAME}_nvcc" ${HIPBLASSRCS})
  set_PROPERTY(TARGET "${PROJECT_NAME}_nvcc" APPEND_STRING PROPERTY LINK_FLAGS " ${LDFLAGS} ")
  TARGET_LINK_LIBRARIES("${PROJECT_NAME}_nvcc" cudart cublas)
  
  install(TARGETS "${PROJECT_NAME}_nvcc" 
    RUNTIME DESTINATION lib
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    PERMISSIONS WORLD_READ WORLD_WRITE WORLD_EXECUTE
  )

endif()

install(DIRECTORY 
        "${CMAKE_CURRENT_SOURCE_DIR}/../include/" 
    DESTINATION 
        ${CMAKE_INSTALL_INCLUDEDIR} 
    PATTERN 
            "*.h.in" EXCLUDE
)

install(FILES 
        "${PROJECT_BINARY_DIR}/../include/hipblas-version.h" 
    DESTINATION 
        ${CMAKE_INSTALL_INCLUDEDIR}
)

