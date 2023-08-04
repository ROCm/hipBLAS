/* ************************************************************************
 * Copyright (C) 2016-2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *
 * ************************************************************************ */

#ifdef WIN32
#include <windows.h>
//
#include <random>
#endif

#include "hipblas.h"
#include "utility.h"
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <new>
#include <stdexcept>
#include <stdlib.h>

#ifdef WIN32
#define strcasecmp(A, B) _stricmp(A, B)

#ifdef __cpp_lib_filesystem
#include <filesystem>
namespace fs = std::filesystem;
#else
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#endif

// Not WIN32
#else
#include <fcntl.h>
#include <unistd.h>
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

    fs::path exepath(result.begin(), result.end());

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

/* ============================================================================================ */
// Temp directory rooted random path
std::string hipblas_tempname()
{
#ifdef WIN32
    // Generate "/tmp/hipblas-XXXXXX" like file name
    const std::string alphanum     = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuv";
    int               stringlength = alphanum.length() - 1;
    std::string       uniquestr    = "hipblas-";

    for(auto n : {0, 1, 2, 3, 4, 5})
        uniquestr += alphanum.at(rand() % stringlength);

    fs::path tmpname = fs::temp_directory_path() / uniquestr;

    return tmpname.string();
#else
    char tmp[] = "/tmp/hipblas-XXXXXX";
    int  fd    = mkostemp(tmp, O_CLOEXEC);
    if(fd == -1)
    {
        dprintf(STDERR_FILENO, "Cannot open temporary file: %m\n");
        exit(EXIT_FAILURE);
    }

    return std::string(tmp);
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
    hipblasAtomicsMode_t mode;
    auto                 status = hipblasGetAtomicsMode(m_handle, &mode);
    if(status != HIPBLAS_STATUS_SUCCESS)
        throw std::runtime_error(hipblasStatusToString(status));

    if(mode != hipblasAtomicsMode_t(arg.atomics_mode))
        status = hipblasSetAtomicsMode(m_handle, hipblasAtomicsMode_t(arg.atomics_mode));
    if(status == HIPBLAS_STATUS_SUCCESS)
    {
        /*
        // If the test specifies user allocated workspace, allocate and use it
        if(arg.user_allocated_workspace)
        {
            if((hipMalloc)(&m_memory, arg.user_allocated_workspace) != hipSuccess)
                throw std::bad_alloc();
            status = rocblas_set_workspace(m_handle, m_memory, arg.user_allocated_workspace);
        }
    */
    }
    else
    {
        throw std::runtime_error(hipblasStatusToString(status));
    }
}

hipblasLocalHandle::~hipblasLocalHandle()
{
    if(m_memory)
    {
        CHECK_HIP_ERROR(hipFree(m_memory));
    }
    hipblasStatus_t status = hipblasDestroy(m_handle);
    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        printf("hipblasDestroy error!\n");
    }
}

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================================ */
/*  timing:*/

/*! \brief  CPU Timer(in microsecond): synchronize with the default device and return wall time */
double get_time_us(void)
{
    (void)hipDeviceSynchronize();

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
    (void)hipStreamSynchronize(stream);

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
hipblasClientProcessor getArch()
{
    int device;
    CHECK_HIP_ERROR(hipGetDevice(&device));
    hipDeviceProp_t deviceProperties;
    CHECK_HIP_ERROR(hipGetDeviceProperties(&deviceProperties, device));

    // strip out xnack/ecc from name
    std::string deviceFullString(deviceProperties.gcnArchName);
    std::string deviceString = deviceFullString.substr(0, deviceFullString.find(":"));

    if(deviceString.find("gfx803") != std::string::npos)
    {
        return hipblasClientProcessor::gfx803;
    }
    else if(deviceString.find("gfx900") != std::string::npos)
    {
        return hipblasClientProcessor::gfx900;
    }
    else if(deviceString.find("gfx906") != std::string::npos)
    {
        return hipblasClientProcessor::gfx906;
    }
    else if(deviceString.find("gfx908") != std::string::npos)
    {
        return hipblasClientProcessor::gfx908;
    }
    else if(deviceString.find("gfx90a") != std::string::npos)
    {
        return hipblasClientProcessor::gfx90a;
    }
    else if(deviceString.find("gfx940") != std::string::npos)
    {
        return hipblasClientProcessor::gfx940;
    }
    else if(deviceString.find("gfx941") != std::string::npos)
    {
        return hipblasClientProcessor::gfx941;
    }
    else if(deviceString.find("gfx942") != std::string::npos)
    {
        return hipblasClientProcessor::gfx942;
    }
    else if(deviceString.find("gfx1010") != std::string::npos)
    {
        return hipblasClientProcessor::gfx1010;
    }
    else if(deviceString.find("gfx1011") != std::string::npos)
    {
        return hipblasClientProcessor::gfx1011;
    }
    else if(deviceString.find("gfx1012") != std::string::npos)
    {
        return hipblasClientProcessor::gfx1012;
    }
    else if(deviceString.find("gfx1030") != std::string::npos)
    {
        return hipblasClientProcessor::gfx1030;
    }
    else if(deviceString.find("gfx1100") != std::string::npos)
    {
        return hipblasClientProcessor::gfx1100;
    }
    else if(deviceString.find("gfx1101") != std::string::npos)
    {
        return hipblasClientProcessor::gfx1101;
    }
    else if(deviceString.find("gfx1102") != std::string::npos)
    {
        return hipblasClientProcessor::gfx1102;
    }
    return static_cast<hipblasClientProcessor>(0);
}

int getArchMajor()
{
    return static_cast<int>(getArch()) / 100;
}

#ifdef __cplusplus
}
#endif
