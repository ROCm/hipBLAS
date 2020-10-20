/* ************************************************************************
 * Copyright 2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#ifndef _ARGUMENT_MODEL_HPP_
#define _ARGUMENT_MODEL_HPP_

#include "hipblas_arguments.hpp"
#include <iostream>
#include <sstream>

// ArgumentModel template has a variadic list of argument enums
template <hipblas_argument... Args>
class ArgumentModel
{
    // Whether model has a particular parameter
    // TODO: Replace with C++17 fold expression ((Args == param) || ...)
    static constexpr bool has(hipblas_argument param)
    {
        for(auto x : {Args...})
            if(x == param)
                return true;
        return false;
    }

public:
    void log_perf(std::stringstream& name_line,
                  std::stringstream& val_line,
                  const Arguments&   arg,
                  double             gpu_us,
                  double             gflops,
                  double             gbytes,
                  double             norm1,
                  double             norm2)
    {
        constexpr bool has_batch_count = has(e_batch_count);
        int            batch_count     = has_batch_count ? arg.batch_count : 1;
        int            hot_calls       = arg.iters < 1 ? 1 : arg.iters;

        // per/us to per/sec *10^6
        double hipblas_gflops = gflops * batch_count * hot_calls / gpu_us * 1e6;
        double hipblas_GBps   = gbytes * batch_count * hot_calls / gpu_us * 1e6;

        // append performance fields
        name_line << ",hipblas-Gflops,hipblas-GB/s,hipblas-us,";
        val_line << ", " << hipblas_gflops << ", " << hipblas_GBps << ", " << gpu_us << ", ";

        if(arg.unit_check || arg.norm_check)
        {
            if(arg.norm_check)
            {
                name_line << "norm_error_host_ptr,norm_error_device_ptr,";
                val_line << norm1 << ", " << norm2 << ", ";
            }
        }
    }

    template <typename T>
    void log_args(std::ostream&    str,
                  const Arguments& arg,
                  double           gpu_us,
                  double           gflops,
                  double           gpu_bytes = 0,
                  double           norm1     = 0,
                  double           norm2     = 0)
    {
        std::stringstream name_list;
        std::stringstream value_list;

        // Output (name, value) pairs to name_list and value_list
        auto print = [&, delim = ""](const char* name, auto&& value) mutable {
            name_list << delim << name;
            value_list << delim << value;
            delim = ",";
        };

        // Args is a parameter pack of type:   hipblas_argument...
        // The hipblas_argument enum values in Args correspond to the function arguments that
        // will be printed by hipblas_test or hipblas_bench. For example, the function:
        //
        //  hipblas_ddot(hipblas_handle handle,
        //                                 hipblas_int    n,
        //                                 const double*  x,
        //                                 hipblas_int    incx,
        //                                 const double*  y,
        //                                 hipblas_int    incy,
        //                                 double*        result);
        // will have <Args> = <e_N, e_incx, e_incy>
        //
        // print is a lambda defined above this comment block
        //
        // arg is an instance of the Arguments struct
        //
        // apply is a templated lambda for C++17 and a templated fuctor for C++14
        //
        // For hipblas_ddot, the following template specialization of apply will be called:
        // apply<e_N>(print, arg, T{}), apply<e_incx>(print, arg, T{}),, apply<e_incy>(print, arg, T{})
        //
        // apply in turn calls print with a string corresponding to the enum, for example "N" and the value of N
        //

#if __cplusplus >= 201703L
        // C++17
        (ArgumentsHelper::apply<Args>(print, arg, T{}), ...);
#else
        // C++14. TODO: Remove when C++17 is used
        (void)(int[]){(ArgumentsHelper::apply<Args>{}()(print, arg, T{}), 0)...};
#endif

        if(arg.timing)
            log_perf(name_list, value_list, arg, gpu_us, gflops, gpu_bytes, norm1, norm2);

        str << name_list.str() << "\n" << value_list.str() << std::endl;
    }
};

#endif
