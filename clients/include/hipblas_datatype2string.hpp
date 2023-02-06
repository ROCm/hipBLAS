/* ************************************************************************
 * Copyright (C) 2018-2023 Advanced Micro Devices, Inc. All rights reserved.
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
 * ************************************************************************ */

#ifndef hipblas_DATATYPE2STRING_H_
#define hipblas_DATATYPE2STRING_H_

#include "hipblas.h"
#include "hipblas.hpp"
#include <ostream>
#include <string>

enum hipblas_initialization
{
    rand_int   = 111,
    trig_float = 222,
    hpl        = 333,
};

inline constexpr auto hipblas_initialization2string(hipblas_initialization init)
{
    switch(init)
    {
    case hipblas_initialization::rand_int:
        return "rand_int";
    case hipblas_initialization::trig_float:
        return "trig_float";
    case hipblas_initialization::hpl:
        return "hpl";
    }
    return "invalid";
}

hipblas_initialization string2hipblas_initialization(const std::string& value);

inline std::ostream& operator<<(std::ostream& os, hipblas_initialization init)
{
    return os << hipblas_initialization2string(init);
}

// Complex output
inline std::ostream& operator<<(std::ostream& os, const hipblasComplex& x)
{
    os << "'(" << x.real() << ":" << x.imag() << ")'";
    return os;
}

inline std::ostream& operator<<(std::ostream& os, const hipblasDoubleComplex& x)
{
    os << "'(" << x.real() << ":" << x.imag() << ")'";
    return os;
}

/* ============================================================================================ */
/*  Convert hipblas constants to lapack char. */

char hipblas2char_operation(hipblasOperation_t value);

char hipblas2char_fill(hipblasFillMode_t value);

char hipblas2char_diagonal(hipblasDiagType_t value);

char hipblas2char_side(hipblasSideMode_t value);

/* ============================================================================================ */
/*  Convert lapack char constants to hipblas type. */

hipblasOperation_t char2hipblas_operation(char value);

hipblasFillMode_t char2hipblas_fill(char value);

hipblasDiagType_t char2hipblas_diagonal(char value);

hipblasSideMode_t char2hipblas_side(char value);

hipDataType string2hipblas_datatype(const std::string& value);

// return precision string for hipblas_datatype
inline constexpr auto hipblas_datatype2string(hipDataType type)
{
    switch(type)
    {
    case HIP_R_16F:
        return "f16_r";
    case HIP_R_32F:
        return "f32_r";
    case HIP_R_64F:
        return "f64_r";
    case HIP_C_16F:
        return "f16_k";
    case HIP_C_32F:
        return "f32_c";
    case HIP_C_64F:
        return "f64_c";
    case HIP_R_8I:
        return "i8_r";
    case HIP_R_8U:
        return "u8_r";
    case HIP_R_32I:
        return "i32_r";
    case HIP_R_32U:
        return "u32_r";
    case HIP_C_8I:
        return "i8_c";
    case HIP_C_8U:
        return "u8_c";
    case HIP_C_32I:
        return "i32_c";
    case HIP_C_32U:
        return "u32_c";
    case HIP_R_16BF:
        return "bf16_r";
    case HIP_C_16BF:
        return "bf16_c";
        // case HIPBLAS_DATATYPE_INVALID:
        //     return "invalid";
    }
    return "invalid";
}

#endif
