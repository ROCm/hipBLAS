# ########################################################################
# Copyright 2016-2020 Advanced Micro Devices, Inc.
# ########################################################################

if(EXISTS /etc/redhat-release)
    if(CXX_VERSION_STRING MATCHES "clang")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp=libgomp -pthread -stdlib=libstdc++ -std=c++14")
    else()
        set(CMAKE_CXX_FLAGS "-isystem /opt/rocm/include ${CMAKE_CXX_FLAGS} -fopenmp=libgomp -pthread")
    endif()
else()
    set(CMAKE_CXX_FLAGS "-isystem /opt/rocm/include ${CMAKE_CXX_FLAGS} -fopenmp -pthread")
endif()

set( THREADS_PREFER_PTHREAD_FLAG ON )
find_package( Threads REQUIRED )

# Linking lapack library requires fortran flags
enable_language( Fortran )
find_package( cblas CONFIG REQUIRED )
if( NOT cblas_FOUND )
  message( FATAL_ERROR "cblas is a required dependency and is not found;  try adding cblas path to CMAKE_PREFIX_PATH" )
endif( )


set( hipblas_benchmark_common
      ../common/utility.cpp
      ../common/cblas_interface.cpp
      ../common/hipblas_arguments.cpp
      ../common/hipblas_parse_data.cpp
      ../common/hipblas_datatype2string.cpp
      ../common/norm.cpp
      ../common/unit.cpp
      ../common/near.cpp
      ../common/arg_check.cpp
      ../common/hipblas_template_specialization.cpp
    )

add_executable( hipblas-bench client.cpp ${hipblas_benchmark_common} )


target_compile_features( hipblas-bench PRIVATE cxx_static_assert cxx_nullptr cxx_auto_type )

# Internal header includes
target_include_directories( hipblas-bench
  PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include/blas1>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include/blas2>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include/blas3>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include/blas_ex>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../library/include>
)

set( BLIS_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/${BUILD_DIR}/deps/blis/include/blis )
set( BLIS_LIBRARY ${CMAKE_SOURCE_DIR}/${BUILD_DIR}/deps/blis/lib/libblis.so )

if( OS_ID_rhel OR OS_ID_sles OR OS_ID_centos)
    if( OS_ID_rhel OR OS_ID_centos)
        if( EXISTS "/usr/lib/gcc/x86_64-redhat-linux/8/" )
            set( ROCM_OPENMP_PATH /usr/lib/gcc/x86_64-redhat-linux/8 )
        else()
            set( ROCM_OPENMP_PATH /opt/rh/devtoolset-7/root/usr/lib/gcc/x86_64-redhat-linux/7 )
        endif()

        # defer OpenMP include as search order must come after clang
        set( XXX_OPENMP_INCLUDE_DIR ${ROCM_OPENMP_PATH}/include )
        set( OPENMP_LIBRARY ${ROCM_OPENMP_PATH}/libgomp.so )
        if( CXX_VERSION_STRING MATCHES "clang")
            set( XXX_OPENMP_INCLUDE_DIR ${ROCM_OPENMP_PATH}/include )
        endif()
    else()
    #SLES
        set( OPENMP_INCLUDE_DIR /usr/lib64/gcc/x86_64-suse-linux/7/include/ )
        set( OPENMP_LIBRARY /usr/lib64/gcc/x86_64-suse-linux/7/libgomp.so )
    endif()

    message(STATUS "RocmPath: ${ROCM_PATH}")
    if(EXISTS ${ROCM_PATH}/llvm/lib/clang/12.0.0/include/immintrin.h AND CMAKE_CXX_COMPILER MATCHES ".*/hcc$|.*/hipcc$" )
        set( CLANG_INCLUDE_DIR ${ROCM_PATH}/llvm/lib/clang/12.0.0/include )
    elseif(EXISTS ${ROCM_PATH}/llvm/lib/clang/11.0.0/include/immintrin.h AND CMAKE_CXX_COMPILER MATCHES ".*/hcc$|.*/hipcc$" )
        set( CLANG_INCLUDE_DIR ${ROCM_PATH}/llvm/lib/clang/11.0.0/include )
    elseif (EXISTS ${ROCM_PATH}/hcc/lib/clang/10.0.0/include/immintrin.h AND CMAKE_CXX_COMPILER MATCHES ".*/hcc$|.*/hipcc$" )
        set( CLANG_INCLUDE_DIR ${ROCM_PATH}/hcc/lib/clang/10.0.0/include )
    elseif (EXISTS ${ROCM_PATH}/hcc/lib/clang/9.0.0/include/immintrin.h AND CMAKE_CXX_COMPILER MATCHES ".*/hcc$|.*/hipcc$")
        set( CLANG_INCLUDE_DIR ${ROCM_PATH}/hcc/lib/clang/9.0.0/include )
    else()
        set( CLANG_INCLUDE_DIR )
    endif()

    # External header includes included as system files
    target_include_directories( hipblas-bench
      SYSTEM PRIVATE
        $<BUILD_INTERFACE:${CLANG_INCLUDE_DIR}>
        $<BUILD_INTERFACE:${OPENMP_INCLUDE_DIR}>
        $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>
        $<BUILD_INTERFACE:${HCC_INCLUDE_DIRS}>
        $<BUILD_INTERFACE:${CBLAS_INCLUDE_DIRS}>
    )

    target_link_libraries( hipblas-bench PRIVATE hipblas_fortran_client roc::hipblas cblas lapack ${OPENMP_LIBRARY} )
else()
    # External header includes included as system files
    target_include_directories( hipblas-bench
      SYSTEM PRIVATE
        $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>
        $<BUILD_INTERFACE:${HCC_INCLUDE_DIRS}>
        $<BUILD_INTERFACE:${CBLAS_INCLUDE_DIRS}>
    )

    target_link_libraries( hipblas-bench PRIVATE hipblas_fortran_client roc::hipblas cblas lapack )
endif()

if( CUDA_FOUND )
  target_include_directories( hipblas-bench
    PRIVATE
      $<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}>
      $<BUILD_INTERFACE:${hip_INCLUDE_DIRS}>
    )
  target_compile_definitions( hipblas-bench PRIVATE __HIP_PLATFORM_NVCC__ )
  target_link_libraries( hipblas-bench PRIVATE ${CUDA_LIBRARIES} )
else( )
  # auto set in hip_common.h
  #target_compile_definitions( hipblas-bench PRIVATE __HIP_PLATFORM_HCC__ )
  target_link_libraries( hipblas-bench PRIVATE hip::host )
endif( )

if( CMAKE_COMPILER_IS_GNUCXX OR CXX_VERSION_STRING MATCHES "clang")
  # GCC or hip-clang needs specific flags to turn on f16c intrinsics
  target_compile_options( hipblas-bench PRIVATE -mf16c )
endif( )

if( CXX_VERSION_STRING MATCHES "clang" )
  target_link_libraries( hipblas-bench PRIVATE -lpthread -lstdc++ -lm )
  if(OS_ID_rhel OR OS_ID_centos)
    set(CMAKE_CXX_FLAGS "-isystem ${CLANG_INCLUDE_DIR} -isystem ${XXX_OPENMP_INCLUDE_DIR} ${CMAKE_CXX_FLAGS}")
  endif( )
else( )
  if(OS_ID_rhel OR OS_ID_centos)
    set(CMAKE_CXX_FLAGS "-isystem ${CLANG_INCLUDE_DIR} -isystem ${XXX_OPENMP_INCLUDE_DIR} ${CMAKE_CXX_FLAGS}")
  endif( )
  set(CMAKE_CXX_FLAGS "-isystem ${ROCM_PATH}/include ${CMAKE_CXX_FLAGS}")
endif( )

set_target_properties( hipblas-bench PROPERTIES CXX_EXTENSIONS NO )
set_target_properties( hipblas-bench PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging" )
#add_dependencies( hipblas-bench hipblas-bench-common )

target_compile_definitions( hipblas-bench PRIVATE HIPBLAS_BENCH ROCM_USE_FLOAT16 )
