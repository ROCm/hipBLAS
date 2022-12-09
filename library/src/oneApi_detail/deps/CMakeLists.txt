set(CMAKE_CXX_COMPILER icpx)
set(CMAKE_CXX_FLAGS "-Wno-deprecated-declarations")
set(POSITION_INDEPENDENT_CODE ON)

add_library(sycl_wrapper SHARED ./sycl_wrapper.cpp ./onemkl.cpp)

#message(STATUS "DPCPP_SYCL_INCUDE_DIR: ${DPCPP_INCLUDE_DIR}")
target_link_libraries(sycl_wrapper PUBLIC sycl MKL::MKL mkl_sycl MKL::mkl_intel_ilp64 MKL::mkl_sequential MKL::mkl_core)
target_include_directories(sycl_wrapper
                            PUBLIC  $<BUILD_INTERFACE:${DPCPP_INCLUDE_DIR}/sycl>
                            PUBLIC  $<BUILD_INTERFACE:${DPCPP_INCLUDE_DIR}>
                            PUBLIC  $<BUILD_INTERFACE:~/packages>
                            PRIVATE ${PROJECT_BINARY_DIR}/include/hipblas
                            PRIVATE $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>
                            PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/.. "$<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>")
target_compile_options(sycl_wrapper PRIVATE "$<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>")
