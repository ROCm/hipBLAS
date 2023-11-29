/* ************************************************************************
 * Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ************************************************************************ */

#include "blas2/testing_hbmv.hpp"
#include "blas2/testing_hbmv_batched.hpp"
#include "blas2/testing_hbmv_strided_batched.hpp"
#include "hipblas_data.hpp"
#include "hipblas_test.hpp"
#include "type_dispatch.hpp"

namespace
{
    // possible hbmv test cases
    enum hbmv_test_type
    {
        HBMV,
        HBMV_BATCHED,
        HBMV_STRIDED_BATCHED,
    };

    //hbmv test template
    template <template <typename...> class FILTER, hbmv_test_type HBMV_TYPE>
    struct hbmv_template : HipBLAS_Test<hbmv_template<FILTER, HBMV_TYPE>, FILTER>
    {
        template <typename... T>
        struct type_filter_functor
        {
            bool operator()(const Arguments& args)
            {
                // additional global filters applied first
                if(!hipblas_client_global_filters(args))
                    return false;

                // type filters
                return static_cast<bool>(FILTER<T...>{});
            }
        };

        // Filter for which types apply to this suite
        static bool type_filter(const Arguments& arg)
        {
            return hipblas_simple_dispatch<hbmv_template::template type_filter_functor>(arg);
        }

        // Filter for which functions apply to this suite
        static bool function_filter(const Arguments& arg)
        {
            switch(HBMV_TYPE)
            {
            case HBMV:
                return !strcmp(arg.function, "hbmv") || !strcmp(arg.function, "hbmv_bad_arg");
            case HBMV_BATCHED:
                return !strcmp(arg.function, "hbmv_batched")
                       || !strcmp(arg.function, "hbmv_batched_bad_arg");
            case HBMV_STRIDED_BATCHED:
                return !strcmp(arg.function, "hbmv_strided_batched")
                       || !strcmp(arg.function, "hbmv_strided_batched_bad_arg");
            }
            return false;
        }

        // Google Test name suffix based on parameters
        static std::string name_suffix(const Arguments& arg)
        {
            std::string name;
            if constexpr(HBMV_TYPE == HBMV)
                testname_hbmv(arg, name);
            else if constexpr(HBMV_TYPE == HBMV_BATCHED)
                testname_hbmv_batched(arg, name);
            else if constexpr(HBMV_TYPE == HBMV_STRIDED_BATCHED)
                testname_hbmv_strided_batched(arg, name);
            return std::move(name);
        }
    };

    // By default, arbitrary type combinations are invalid.
    // The unnamed second parameter is used for enable_if_t below.
    template <typename, typename = void>
    struct hbmv_testing : hipblas_test_invalid
    {
    };

    // When the condition in the second argument is satisfied, the type combination
    // is valid. When the condition is false, this specialization does not apply.
    template <typename T>
    struct hbmv_testing<
        T,
        std::enable_if_t<
            std::is_same_v<T, hipblasComplex> || std::is_same_v<T, hipblasDoubleComplex>>>
        : hipblas_test_valid
    {
        void operator()(const Arguments& arg)
        {
            if(!strcmp(arg.function, "hbmv"))
                testing_hbmv<T>(arg);
            else if(!strcmp(arg.function, "hbmv_bad_arg"))
                testing_hbmv_bad_arg<T>(arg);
            else if(!strcmp(arg.function, "hbmv_batched"))
                testing_hbmv_batched<T>(arg);
            else if(!strcmp(arg.function, "hbmv_batched_bad_arg"))
                testing_hbmv_batched_bad_arg<T>(arg);
            else if(!strcmp(arg.function, "hbmv_strided_batched"))
                testing_hbmv_strided_batched<T>(arg);
            else if(!strcmp(arg.function, "hbmv_strided_batched_bad_arg"))
                testing_hbmv_strided_batched_bad_arg<T>(arg);
            else
                FAIL() << "Internal error: Test called with unknown function: " << arg.function;
        }
    };

    using hbmv = hbmv_template<hbmv_testing, HBMV>;
    TEST_P(hbmv, blas2)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(hipblas_simple_dispatch<hbmv_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(hbmv);

    using hbmv_batched = hbmv_template<hbmv_testing, HBMV_BATCHED>;
    TEST_P(hbmv_batched, blas2)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(hipblas_simple_dispatch<hbmv_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(hbmv_batched);

    using hbmv_strided_batched = hbmv_template<hbmv_testing, HBMV_STRIDED_BATCHED>;
    TEST_P(hbmv_strided_batched, blas2)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(hipblas_simple_dispatch<hbmv_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(hbmv_strided_batched);

} // namespace
