add_subdirectory(deps)

set( hipblas_source "${CMAKE_CURRENT_SOURCE_DIR}/hipblas.cpp" )

#target_link_libraries(hipblas sycl_wrapper)
#target_link_options(hipblas PRIVATE -fsycl -L${CMAKE_BINARY_DIR} -Wl,-rpath=${CMAKE_BINARY_DIR}:${DPCPP_CORE_LIBDIR}:${DPCPP_SYCL_LIBDIR})

#target_link_libraries(hip_sycl_interop onemkl_gemm_wrapper)
#target_include_directories(hip_sycl_interop PUBLIC ${CHIP_SRC_DIR}/HIP/include ${CHIP_SRC_DIR}/include)
