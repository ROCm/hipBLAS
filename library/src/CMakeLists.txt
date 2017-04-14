CMAKE_MINIMUM_REQUIRED (VERSION 2.8.8) 
MESSAGE(STATUS "CMAKE VERSION ${CMAKE_VERSION}")

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


SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/../../cmake ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

#Setting a Project name
SET(PROJECT_NAME "hipblas")
SET(ROCM_PATH "/opt/rocm/")

PROJECT (${PROJECT_NAME})
MESSAGE(STATUS "PROJECT NAME : ${PROJECT_NAME}")

SET(HIP_SUPPORT $ENV{HIP_SUPPORT})

# Find HIP
set (HIP_PATH $ENV{HIP_PATH})
if (NOT DEFINED HIP_PATH)
       set (HIP_PATH ${ROCM_PATH}/hip)
endif()
MESSAGE (STATUS "HIP_PATH : ${HIP_PATH}")

execute_process(COMMAND ${HIP_PATH}/bin/hipconfig --platform OUTPUT_VARIABLE PLATFORM)
MESSAGE (STATUS "PLATFORM : ${PLATFORM}")

# Build hcfft and hipfft from source on AMD platform
IF (${PLATFORM} MATCHES "hcc")
  MESSAGE (STATUS "building for hcc")
  # Find HCC compiler
# FIND_PACKAGE(HC++ 1.0 REQUIRED)

  #Setting a variable for source files
  SET (HCBLASSRCS ${BLASSRC} hcblas.cpp)

  execute_process(COMMAND ${HCC_CONFIG} --install --cxxflags
                            OUTPUT_VARIABLE HCC_CXXFLAGS)
  execute_process(COMMAND ${HCC_CONFIG} --install --ldflags --shared
                            OUTPUT_VARIABLE HCC_LDFLAGS)
 
  string(STRIP "${HCC_CXXFLAGS}" HCC_CXXFLAGS)
  string(STRIP "${HCC_LDFLAGS}" HCC_LDFLAGS)
  

  set (HCC_CXXFLAGS "${HCC_CXXFLAGS} -I${ROCM_PATH}/rocblas/include -I${CMAKE_CURRENT_SOURCE_DIR}/../include -I${ROCM_PATH}/include -I${HIP_PATH}/include -D__HIP_PLATFORM_HCC__= -fPIC")
  set (HCC_LDFLAGS "${HCC_LDFLAGS} -L${HIP_PATH}/lib -lhip_hcc -Wl,-rpath-link,${HIP_PATH}/lib -fPIC")
 
  SET (HIPBLASSRCS ${CMAKE_CURRENT_SOURCE_DIR}/hcc_detail/hipblas.cpp)
    
    # Setting header dependencies for hipblas library generation
  FOREACH(src_file ${HIPBLASSRCS})
          SET_PROPERTY(SOURCE ${src_file} APPEND_STRING PROPERTY COMPILE_FLAGS " ${HCC_CXXFLAGS} ")
  ENDFOREACH()
    
  #Generating hipblas shared object
  ADD_LIBRARY("${PROJECT_NAME}-hcc" ${HIPBLASSRCS})
  SET_PROPERTY(TARGET "${PROJECT_NAME}-hcc" APPEND_STRING PROPERTY LINK_FLAGS " ${HCC_LDFLAGS} ")
  TARGET_LINK_LIBRARIES("${PROJECT_NAME}-hcc" hc_am)

  INSTALL(TARGETS "${PROJECT_NAME}-hcc" 
        RUNTIME DESTINATION lib
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        PERMISSIONS WORLD_READ WORLD_WRITE WORLD_EXECUTE
  )
  
  MESSAGE(STATUS "CMAKE_CURRENT_SOURCE_DIR : ${CMAKE_CURRENT_SOURCE_DIR}")

# INSTALL(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../include/" DESTINATION include )  
# INSTALL(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../include/" DESTINATION include PATTERN "*.h.in" EXCLUDE)  

ELSEIF (${PLATFORM} MATCHES "nvcc")
  
  MESSAGE(STATUS "Building hipblas.cpp")
  
  SET (CXXFLAGS "-I${CMAKE_CURRENT_SOURCE_DIR}/../include -I${HIP_PATH}/include -I/usr/local/cuda/include -D__HIP_PLATFORM_NVCC__=")
  SET (LDFLAGS "-L/usr/local/cuda/lib64 -L/usr/local/cuda/lib")
  SET (HIPBLASSRCS ${CMAKE_CURRENT_SOURCE_DIR}/nvcc_detail/hipblas.cpp)
  
  FOREACH(src_file ${HIPBLASSRCS})
	  SET_PROPERTY(SOURCE ${src_file} APPEND_STRING PROPERTY COMPILE_FLAGS " ${CXXFLAGS} ")
  ENDFOREACH()

  ADD_LIBRARY("${PROJECT_NAME}_nvcc" ${HIPBLASSRCS})
  SET_PROPERTY(TARGET "${PROJECT_NAME}_nvcc" APPEND_STRING PROPERTY LINK_FLAGS " ${LDFLAGS} ")
  TARGET_LINK_LIBRARIES("${PROJECT_NAME}_nvcc" cudart cublas)
  
  INSTALL(TARGETS "${PROJECT_NAME}_nvcc" 
    RUNTIME DESTINATION lib
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    PERMISSIONS WORLD_READ WORLD_WRITE WORLD_EXECUTE
  )

  INSTALL(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../include/" DESTINATION include PATTERN "*.h.in" EXCLUDE) 
 
ENDIF()

