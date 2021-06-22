/* ************************************************************************
 * Copyright 2016-2021 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#ifdef WIN32
#include <windows.h>
#endif

#include "hipblas.h"
#include "utility.h"
#include <chrono>

//
// https://en.cppreference.com/w/User:D41D8CD98F/feature_testing_macros
//
#ifdef __cpp_lib_filesystem
#include <filesystem>
#else
#include <experimental/filesystem>

namespace std
{
    namespace filesystem = experimental::filesystem;
}
#endif

hipblas_rng_t hipblas_rng(69069);
hipblas_rng_t hipblas_seed(hipblas_rng);

template <>
char type2char<float>()
{
    return 's';
}

template <>
char type2char<double>()
{
    return 'd';
}

//  template<>
//  char type2char<hipblasComplex>(){
//      return 'c';
//  }

//  template<>
//  char type2char<hipblasDoubleComplex>(){
//      return 'z';
//  }

template <>
int type2int<float>(float val)
{
    return (int)val;
}

template <>
int type2int<double>(double val)
{
    return (int)val;
}

template <>
int type2int<hipblasComplex>(hipblasComplex val)
{
    return (int)val.real();
}

template <>
int type2int<hipblasDoubleComplex>(hipblasDoubleComplex val)
{
    return (int)val.real();
}

/* ============================================================================================ */
// Return path of this executable
std::string hipblas_exepath()
{
#ifdef WIN32
    std::vector<TCHAR> result(MAX_PATH + 1);
    // Ensure result is large enough to accomodate the path
    for(;;)
    {
        auto length = GetModuleFileNameA(nullptr, result.data(), result.size());
        if(length < result.size() - 1)
        {
            result.resize(length + 1);
            // result.shrink_to_fit();
            break;
        }
        result.resize(result.size() * 2);
    }

    std::filesystem::path exepath(result.begin(), result.end());

    exepath = exepath.remove_filename();
    // Add trailing "/" to exepath if required
    exepath += exepath.empty() ? "" : "/";
    return exepath.string();
#else
    std::string pathstr;
    char*       path = realpath("/proc/self/exe", 0);
    if(path)
    {
        char* p = strrchr(path, '/');
        if(p)
        {
            p[1]    = 0;
            pathstr = path;
        }
        free(path);
    }
    return pathstr;
#endif
}

/*****************
 * local handles *
 *****************/

hipblasLocalHandle::hipblasLocalHandle()
{
    auto status = hipblasCreate(&m_handle);
    if(status != HIPBLAS_STATUS_SUCCESS)
        throw std::runtime_error(hipblasStatusToString(status));
}

hipblasLocalHandle::hipblasLocalHandle(const Arguments& arg)
    : hipblasLocalHandle()
{
    // for future customization of handle based on arguments, example from rocblas below

    /*
    auto status = rocblas_set_atomics_mode(m_handle, arg.atomics_mode);

    if(status == rocblas_status_success)
    {
        // If the test specifies user allocated workspace, allocate and use it
        if(arg.user_allocated_workspace)
        {
            if((hipMalloc)(&m_memory, arg.user_allocated_workspace) != hipSuccess)
                throw std::bad_alloc();
            status = rocblas_set_workspace(m_handle, m_memory, arg.user_allocated_workspace);
        }
    }

    if(status != rocblas_status_success)
        throw std::runtime_error(rocblas_status_to_string(status));
    */
}

hipblasLocalHandle::~hipblasLocalHandle()
{
    if(m_memory)
        (hipFree)(m_memory);
    hipblasDestroy(m_handle);
}

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================================ */
/*  timing:*/

/*! \brief  CPU Timer(in microsecond): synchronize with the default device and return wall time */
double get_time_us(void)
{
    hipDeviceSynchronize();

    auto now = std::chrono::steady_clock::now();
    // now.time_since_epoch() is the dureation since epogh
    // which is converted to microseconds
    auto duration
        = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();
    return (static_cast<double>(duration));
};

/*! \brief  CPU Timer(in microsecond): synchronize with given queue/stream and return wall time */
double get_time_us_sync(hipStream_t stream)
{
    hipStreamSynchronize(stream);

    auto now = std::chrono::steady_clock::now();
    // now.time_since_epoch() is the dureation since epogh
    // which is converted to microseconds
    auto duration
        = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();
    return (static_cast<double>(duration));
};

/* ============================================================================================ */
/*  device query and print out their ID and name; return number of compute-capable devices. */
int query_device_property()
{
    int             device_count;
    hipblasStatus_t status = (hipblasStatus_t)hipGetDeviceCount(&device_count);
    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        printf("Query device error: cannot get device count \n");
        return -1;
    }
    else
    {
        printf("Query device success: there are %d devices \n", device_count);
    }

    for(int i = 0; i < device_count; i++)
    {
        hipDeviceProp_t props;
        hipblasStatus_t status = (hipblasStatus_t)hipGetDeviceProperties(&props, i);
        if(status != HIPBLAS_STATUS_SUCCESS)
        {
            printf("Query device error: cannot get device ID %d's property\n", i);
        }
        else
        {
            printf("Device ID %d : %s ------------------------------------------------------\n",
                   i,
                   props.name);
            printf("with %3.1f GB memory, clock rate %dMHz @ computing capability %d.%d \n",
                   props.totalGlobalMem / 1e9,
                   (int)(props.clockRate / 1000),
                   props.major,
                   props.minor);
            printf(
                "maxGridDimX %d, sharedMemPerBlock %3.1f KB, maxThreadsPerBlock %d, warpSize %d\n",
                props.maxGridSize[0],
                props.sharedMemPerBlock / 1e3,
                props.maxThreadsPerBlock,
                props.warpSize);

            printf("-------------------------------------------------------------------------\n");
        }
    }

    return device_count;
}

/*  set current device to device_id */
void set_device(int device_id)
{
    hipblasStatus_t status = (hipblasStatus_t)hipSetDevice(device_id);
    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        printf("Set device error: cannot set device ID %d, there may not be such device ID\n",
               (int)device_id);
    }
}

/*******************************************************************************
 * GPU architecture-related functions
 ******************************************************************************/

int getArch()
{
    int device;
    hipGetDevice(&device);
    hipDeviceProp_t deviceProperties;
    hipGetDeviceProperties(&deviceProperties, device);
    return deviceProperties.gcnArch;
}

/*******************************************************************************
 * gemm_ex int8 layout
 ******************************************************************************/
bool layout_pack_int8()
{
    int arch = getArch();
    return arch != 908;
}

#ifdef __cplusplus
}
#endif
