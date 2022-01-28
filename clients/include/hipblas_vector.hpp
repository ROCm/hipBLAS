/* ************************************************************************
 * Copyright 2018-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#ifndef HIPBLAS_VECTOR_H_
#define HIPBLAS_VECTOR_H_

#include "d_vector.hpp"
#include "device_batch_vector.hpp"
#include "hipblas.h"
#include "host_batch_vector.hpp"
#include "utility.h"
#include <cinttypes>
#include <cstdio>
#include <locale.h>
#include <vector>

//!
//! @brief enum to check for NaN initialization of the Input vector/matrix
//!
typedef enum hipblas_check_nan_init_
{
    // Alpha sets NaN
    hipblas_client_alpha_sets_nan,

    // Beta sets NaN
    hipblas_client_beta_sets_nan,

    //  Never set NaN
    hipblas_client_never_set_nan

} hipblas_check_nan_init;

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

//!
//! @brief  Pseudo-vector subclass which uses host memory.
//!
template <typename T>
struct host_vector : std::vector<T>
{
    // Inherit constructors
    using std::vector<T>::vector;

    //!
    //! @brief Constructor.
    //!
    host_vector(size_t n, ptrdiff_t inc)
        : std::vector<T>(n * std::abs(inc))
        , m_n(n)
        , m_inc(inc)
    {
    }

    //!
    //! @brief Copy constructor from host_vector of other types convertible to T
    //!
    template <typename U, std::enable_if_t<std::is_convertible<U, T>{}, int> = 0>
    host_vector(const host_vector<U>& x)
        : std::vector<T>(x.size())
        , m_n(x.size())
        , m_inc(1)
    {
        for(size_t i = 0; i < m_n; ++i)
            (*this)[i] = x[i];
    }

    //!
    //! @brief Decay into pointer wherever pointer is expected
    //!
    operator T*()
    {
        return this->data();
    }

    //!
    //! @brief Decay into constant pointer wherever constant pointer is expected
    //!
    operator const T*() const
    {
        return this->data();
    }

    //!
    //! @brief Transfer from a device vector.
    //! @param  that That device vector.
    //! @return the hip error.
    //!
    hipError_t transfer_from(const device_vector<T>& that)
    {
        hipError_t hip_err;

        if(that.use_HMM && hipSuccess != (hip_err = hipDeviceSynchronize()))
            return hip_err;

        return hipMemcpy(*this,
                         that,
                         sizeof(T) * this->size(),
                         that.use_HMM ? hipMemcpyHostToHost : hipMemcpyDeviceToHost);
    }

    //!
    //! @brief Returns the length of the vector.
    //!
    size_t n() const
    {
        return m_n;
    }

    //!
    //! @brief Returns the increment of the vector.
    //!
    ptrdiff_t inc() const
    {
        return m_inc;
    }

    //!
    //! @brief Returns the batch count (always 1).
    //!
    static constexpr int batch_count()
    {
        return 1;
    }

    //!
    //! @brief Returns the stride (out of context, always 0)
    //!
    static constexpr hipblasStride stride()
    {
        return 0;
    }

    //!
    //! @brief Check if memory exists (out of context, always hipSuccess)
    //!
    static constexpr hipError_t memcheck()
    {
        return hipSuccess;
    }

private:
    size_t    m_n   = 0;
    ptrdiff_t m_inc = 0;
};

//!
//! @brief Template for initializing a host (non_batched|batched|strided_batched)vector.
//! @param that That vector.
//! @param rand_gen The random number generator
//! @param seedReset Reset the seed if true, do not reset the seed otherwise.
//!
template <typename U, typename T>
void hipblas_init_template(U& that, T rand_gen(), bool seedReset, bool alternating_sign = false)
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

        if(alternating_sign)
        {
            for(int i = 0; i < n; i++)
            {
                auto value            = rand_gen();
                batched_data[i * inc] = (i ^ 0) & 1 ? value : hipblas_negate(value);
            }
        }
        else
        {
            for(int i = 0; i < n; ++i)
                batched_data[i * inc] = rand_gen();
        }
    }
}

//!
//! @brief Initialize a host_batch_vector with NaNs.
//! @param that The host_batch_vector to be initialized.
//! @param seedReset reset the seed if true, do not reset the seed otherwise.
//!
template <typename T>
inline void hipblas_init_nan(host_batch_vector<T>& that, bool seedReset = false)
{
    hipblas_init_template(that, random_nan_generator<T>, seedReset);
}

// //!
// //! @brief Initialize a host_vector with NaNs.
// //! @param that The host_vector to be initialized.
// //! @param seedReset reset he seed if true, do not reset the seed otherwise.
// //!
// template <typename T>
// inline void hipblas_init_nan(host_vector<T>& that, bool seedReset = false)
// {
//     hipblas_init_template(that, random_nan_generator<T>, seedReset);
// }

template <typename T>
inline void hipblas_init_nan(
    host_vector<T>& A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
        for(size_t i = 0; i < M; ++i)
            for(size_t j = 0; j < N; ++j)
                A[i + j * lda + i_batch * stride] = T(hipblas_nan_rng());
}

//!
//! @brief Initialize a host_batch_vector.
//! @param that The host_batch_vector.
//! @param seedReset reset the seed if true, do not reset the seed otherwise.
//!
template <typename T>
inline void hipblas_init_hpl(host_batch_vector<T>& that,
                             bool                  seedReset        = false,
                             bool                  alternating_sign = false)
{
    hipblas_init_template(that, random_hpl_generator<T>, seedReset, alternating_sign);
}

//!
//! @brief Initialize a host_batch_vector.
//! @param that The host_batch_vector.
//! @param seedReset reset the seed if true, do not reset the seed otherwise.
//!
template <typename T>
inline void
    hipblas_init(host_batch_vector<T>& that, bool seedReset = false, bool alternating_sign = false)
{
    hipblas_init_template(that, random_generator<T>, seedReset, alternating_sign);
}

//!
//! @brief Initialize a host_vector.
//! @param that The host_vector.
//! @param seedReset reset the seed if true, do not reset the seed otherwise.
//!
template <typename T>
inline void hipblas_init(host_vector<T>& that, bool seedReset = false)
{
    if(seedReset)
        hipblas_seedrand();
    hipblas_init(that, that.size(), 1, 1);
}

//!
//! @brief trig Initialize of a host_batch_vector.
//! @param that The host_batch_vector.
//! @param init_cos cos initialize if true, else sin initialize.
//!
template <typename T>
inline void hipblas_init_trig(host_batch_vector<T>& that, bool init_cos = false)
{
    if(init_cos)
    {
        for(int batch_index = 0; batch_index < that.batch_count(); ++batch_index)
        {
            auto*     batched_data = that[batch_index];
            ptrdiff_t inc          = that.inc();
            auto      n            = that.n();

            if(inc < 0)
                batched_data -= (n - 1) * inc;

            hipblas_init_cos(batched_data, 1, n, inc, 0, 1);
        }
    }
    else
    {
        for(int batch_index = 0; batch_index < that.batch_count(); ++batch_index)
        {
            auto*     batched_data = that[batch_index];
            ptrdiff_t inc          = that.inc();
            auto      n            = that.n();

            if(inc < 0)
                batched_data -= (n - 1) * inc;

            hipblas_init_sin(batched_data, 1, n, inc, 0, 1);
        }
    }
}

//!
//! @brief Initialize a host_vector.
//! @param hx The host_vector.
//! @param arg Specifies the argument class.
//! @param N Length of the host vector.
//! @param incx Increment for the host vector.
//! @param stride_x Incement between the host vector.
//! @param batch_count number of instances in the batch.
//! @param nan_init Initialize vector with Nan's depending upon the hipblas_check_nan_init enum value.
//! @param seedReset reset the seed if true, do not reset the seed otherwise. Use init_cos if seedReset is true else use init_sin.
//! @param alternating_sign Initialize vector so adjacent entries have alternating sign.
//!
template <typename T>
inline void hipblas_init_vector(host_vector<T>&        hx,
                                const Arguments&       arg,
                                size_t                 N,
                                size_t                 incx,
                                hipblasStride          stride_x,
                                int                    batch_count,
                                hipblas_check_nan_init nan_init,
                                bool                   seedReset        = false,
                                bool                   alternating_sign = false)
{
    if(seedReset)
        hipblas_seedrand();

    if(nan_init == hipblas_client_alpha_sets_nan && hipblas_isnan(arg.alpha))
    {
        hipblas_init_nan(hx, 1, N, incx, stride_x, batch_count);
    }
    else if(nan_init == hipblas_client_beta_sets_nan && hipblas_isnan(arg.beta))
    {
        hipblas_init_nan(hx, 1, N, incx, stride_x, batch_count);
    }
    else if(arg.initialization == hipblas_initialization::hpl)
    {
        if(alternating_sign)
            hipblas_init_hpl_alternating_sign(hx, 1, N, incx, stride_x, batch_count);
        else
            hipblas_init_hpl(hx, 1, N, incx, stride_x, batch_count);
    }
    else if(arg.initialization == hipblas_initialization::rand_int)
    {
        if(alternating_sign)
            hipblas_init_alternating_sign(hx, 1, N, incx, stride_x, batch_count);
        else
            hipblas_init(hx, 1, N, incx, stride_x, batch_count);
    }
    else if(arg.initialization == hipblas_initialization::trig_float)
    {
        if(seedReset)
            hipblas_init_cos(hx, 1, N, incx, stride_x, batch_count);
        else
            hipblas_init_sin(hx, 1, N, incx, stride_x, batch_count);
    }
}

//!
//! @brief Initialize a host_batch_vector.
//! @param hx The host_batch_vector.
//! @param arg Specifies the argument class.
//! @param nan_init Initialize vector with Nan's depending upon the hipblas_check_nan_init enum value.
//! @param seedReset reset the seed if true, do not reset the seed otherwise. Use init_cos if seedReset is true else use init_sin.
//! @param alternating_sign Initialize vector so adjacent entries have alternating sign.
//!
template <typename T>
inline void hipblas_init_vector(host_batch_vector<T>&  hx,
                                const Arguments&       arg,
                                hipblas_check_nan_init nan_init,
                                bool                   seedReset        = false,
                                bool                   alternating_sign = false)
{
    if(nan_init == hipblas_client_alpha_sets_nan && hipblas_isnan(arg.alpha))
    {
        hipblas_init_nan(hx, seedReset);
    }
    else if(nan_init == hipblas_client_beta_sets_nan && hipblas_isnan(arg.beta))
    {
        hipblas_init_nan(hx, seedReset);
    }
    else if(arg.initialization == hipblas_initialization::hpl)
    {
        hipblas_init_hpl(hx, seedReset, alternating_sign);
    }
    else if(arg.initialization == hipblas_initialization::rand_int)
    {
        hipblas_init(hx, seedReset, alternating_sign);
    }
    else if(arg.initialization == hipblas_initialization::trig_float)
    {
        hipblas_init_trig(hx, seedReset);
    }
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
