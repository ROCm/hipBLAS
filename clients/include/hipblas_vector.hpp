/* ************************************************************************
 * Copyright 2018-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#ifndef HIPBLAS_VECTOR_H_
#define HIPBLAS_VECTOR_H_

#include "d_vector.hpp"
#include "device_batch_vector.hpp"
#include "hipblas/hipblas.h"
#include "host_batch_vector.hpp"
#include "utility.h"
#include <cinttypes>
#include <cstdio>
#include <gtest/gtest.h>
#include <locale.h>
#include <vector>

/* ============================================================================================ */
/*! \brief  pseudo-vector subclass which uses device memory */
template <typename T, size_t PAD = 4096, typename U = T>
class device_vector : private d_vector<T, PAD, U>
{
public:
    // Must wrap constructor and destructor in functions to allow Google Test macros to work
    explicit device_vector(size_t s)
        : d_vector<T, PAD, U>(s)
    {
        data = this->device_vector_setup();
    }

    ~device_vector()
    {
        this->device_vector_teardown(data);
    }

    // Decay into pointer wherever pointer is expected
    operator T*()
    {
        return data;
    }

    operator const T*() const
    {
        return data;
    }

    // Tell whether malloc failed
    explicit operator bool() const
    {
        return data != nullptr;
    }

    // Disallow copying or assigning
    device_vector(const device_vector&) = delete;
    device_vector& operator=(const device_vector&) = delete;

private:
    T* data;
};

/* ============================================================================================ */
/*! \brief  pseudo-vector subclass which uses host memory */
template <typename T>
struct host_vector : std::vector<T>
{
    // Inherit constructors
    using std::vector<T>::vector;

    // Decay into pointer wherever pointer is expected
    operator T*()
    {
        return this->data();
    }
    operator const T*() const
    {
        return this->data();
    }
};

//!
//! @brief Template for initializing a host (non_batched|batched|strided_batched)vector.
//! @param that That vector.
//! @param rand_gen The random number generator
//! @param seedReset Reset the seed if true, do not reset the seed otherwise.
//!
template <typename U, typename T>
void hipblas_init_template(U& that, T rand_gen(), bool seedReset)
{
    if(seedReset)
        hipblas_seedrand();

    for(int batch_index = 0; batch_index < that.batch_count(); ++batch_index)
    {
        auto*     batched_data = that[batch_index];
        ptrdiff_t inc          = that.inc();
        auto      n            = that.n();
        if(inc < 0)
            batched_data -= (n - 1) * inc;

        for(int i = 0; i < n; ++i)
            batched_data[i * inc] = rand_gen();
    }
}

//!
//! @brief Initialize a host_batch_vector.
//! @param that The host batch vector.
//! @param seedReset reset the seed if true, do not reset the seed otherwise.
//!
template <typename T>
inline void hipblas_init(host_batch_vector<T>& that, bool seedReset = false)
{
    hipblas_init_template(that, random_generator<T>, seedReset);
}

//!
//! @brief Template for initializing a host (non_batched|batched|strided_batched)vector.
//! @param that That vector.
//! @param rand_gen The random number generator for odd elements
//! @param rand_gen_alt The random number generator for even elements
//! @param seedReset Reset the seed if true, do not reset the seed otherwise.
//!
template <typename U, typename T>
void hipblas_init_alternating_template(U& that, T rand_gen(), T rand_gen_alt(), bool seedReset)
{
    if(seedReset)
        hipblas_seedrand();

    for(int b = 0; b < that.batch_count(); ++b)
    {
        auto*     batched_data = that[b];
        ptrdiff_t inc          = that.inc();
        auto      n            = that.n();
        if(inc < 0)
            batched_data -= (n - 1) * inc;

        for(int i = 0; i < n; ++i)
        {
            if(i % 2)
                batched_data[i * inc] = rand_gen();
            else
                batched_data[i * inc] = rand_gen_alt();
        }
    }
}

template <typename T>
void hipblas_init_alternating_sign(host_batch_vector<T>& that, bool seedReset = false)
{
    hipblas_init_alternating_template(
        that, random_generator<T>, random_generator_negative<T>, seedReset);
}

#endif
