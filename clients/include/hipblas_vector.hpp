/* ************************************************************************
 * Copyright 2018-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#ifndef HIPBLAS_VECTOR_H_
#define HIPBLAS_VECTOR_H_

#include "hipblas.h"
#include "utility.h"
#include <cinttypes>
#include <cstdio>
#include <gtest/gtest.h>
#include <locale.h>
#include <vector>

/* ============================================================================================ */
/*! \brief  base-class to allocate/deallocate device memory */
template <typename T, size_t PAD, typename U>
class d_vector
{
protected:
    size_t size, bytes;

#ifdef GOOGLE_TEST
    U guard[PAD];
    d_vector(size_t s)
        : size(s)
        , bytes((s + PAD * 2) * sizeof(T))
    {
        // Initialize guard with random data
        if(PAD > 0)
        {
            hipblas_init_nan(guard, PAD);
        }
    }
#else
    d_vector(size_t s)
        : size(s)
        , bytes(s ? s * sizeof(T) : sizeof(T))
    {
    }
#endif

    T* device_vector_setup()
    {
        T* d;
        if((hipMalloc)(&d, bytes) != hipSuccess)
        {
            static char* lc = setlocale(LC_NUMERIC, "");
            fprintf(stderr, "Error allocating %'zu bytes (%zu GB)\n", bytes, bytes >> 30);
            d = nullptr;
        }
#ifdef GOOGLE_TEST
        else
        {
            if(PAD > 0)
            {
                // Copy guard to device memory before allocated memory
                hipMemcpy(d, guard, sizeof(guard), hipMemcpyHostToDevice);

                // Point to allocated block
                d += PAD;

                // Copy guard to device memory after allocated memory
                hipMemcpy(d + size, guard, sizeof(guard), hipMemcpyHostToDevice);
            }
        }
#endif
        return d;
    }

    void device_vector_teardown(T* d)
    {
        if(d != nullptr)
        {
#ifdef GOOGLE_TEST
            if(PAD > 0)
            {
                U host[PAD];

                // Copy device memory after allocated memory to host
                hipMemcpy(host, d + size, sizeof(guard), hipMemcpyDeviceToHost);

                // Make sure no corruption has occurred
                EXPECT_EQ(memcmp(host, guard, sizeof(guard)), 0);

                // Point to guard before allocated memory
                d -= PAD;

                // Copy device memory after allocated memory to host
                hipMemcpy(host, d, sizeof(guard), hipMemcpyDeviceToHost);

                // Make sure no corruption has occurred
                EXPECT_EQ(memcmp(host, guard, sizeof(guard)), 0);
            }
#endif
            // Free device memory
            CHECK_HIP_ERROR((hipFree)(d));
        }
    }
};

/* ============================================================================================ */
/*! \brief  pseudo-vector subclass which uses a batch of device memory pointers and
            an array of pointers in host memory*/
template <typename T, size_t PAD = 4096, typename U = T>
class device_batch_vector : private d_vector<T, PAD, U>
{
public:
    explicit device_batch_vector(size_t b, size_t s)
        : batch(b)
        , d_vector<T, PAD, U>(s)
    {
        data = (T**)malloc(batch * sizeof(T*));
        for(int b = 0; b < batch; ++b)
            data[b] = this->device_vector_setup();
    }

    ~device_batch_vector()
    {
        if(data != nullptr)
        {
            for(int b = 0; b < batch; ++b)
                this->device_vector_teardown(data[b]);
            free(data);
        }
    }

    T* operator[](int n)
    {
        return data[n];
    }

    // clang-format off
    operator T**()
    {
        return data;
    }
    // clang-format on

    // Disallow copying or assigning
    device_batch_vector(const device_batch_vector&) = delete;
    device_batch_vector& operator=(const device_batch_vector&) = delete;

private:
    T**    data;
    size_t batch;
};

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
//! @brief Implementation of the batch vector on host.
//!
template <typename T>
class host_batch_vector
{
public:
    //!
    //! @brief Delete copy constructor.
    //!
    host_batch_vector(const host_batch_vector<T>& that) = delete;

    //!
    //! @brief Delete copy assignement.
    //!
    host_batch_vector& operator=(const host_batch_vector<T>& that) = delete;

    //!
    //! @brief Constructor.
    //! @param n           The length of the vector.
    //! @param inc         The increment.
    //! @param batch_count The batch count.
    //!
    explicit host_batch_vector(int n, int inc, int batch_count)
        : m_n(n)
        , m_inc(inc)
        , m_batch_count(batch_count)
    {
        if(false == this->try_initialize_memory())
        {
            this->free_memory();
        }
    }

    //!
    //! @brief Constructor.
    //! @param n           The length of the vector.
    //! @param inc         The increment.
    //! @param stride      (UNUSED) The stride.
    //! @param batch_count The batch count.
    //!
    explicit host_batch_vector(int n, int inc, ptrdiff_t stride, int batch_count)
        : host_batch_vector(n, inc, batch_count)
    {
    }

    //!
    //! @brief Destructor.
    //!
    ~host_batch_vector()
    {
        this->free_memory();
    }

    //!
    //! @brief Returns the length of the vector.
    //!
    int n() const
    {
        return this->m_n;
    }

    //!
    //! @brief Returns the increment of the vector.
    //!
    int inc() const
    {
        return this->m_inc;
    }

    //!
    //! @brief Returns the batch count.
    //!
    int batch_count() const
    {
        return this->m_batch_count;
    }

    //!
    //! @brief Returns the stride value.
    //!
    ptrdiff_t stride() const
    {
        return 0;
    }

    //!
    //! @brief Random access to the vectors.
    //! @param batch_index the batch index.
    //! @return The mutable pointer.
    //!
    T* operator[](int batch_index)
    {

        return this->m_data[batch_index];
    }

    //!
    //! @brief Constant random access to the vectors.
    //! @param batch_index the batch index.
    //! @return The non-mutable pointer.
    //!
    const T* operator[](int batch_index) const
    {

        return this->m_data[batch_index];
    }

    //!
    //! @brief Cast to a double pointer.
    //!
    // clang-format off
    operator T**()
    // clang-format on
    {
        return this->m_data;
    }

    //!
    //! @brief Constant cast to a double pointer.
    //!
    operator const T* const *()
    {
        return this->m_data;
    }

    //!
    //! @brief Copy from a host batched vector.
    //! @param that the vector the data is copied from.
    //! @return true if the copy is done successfully, false otherwise.
    //!
    bool copy_from(const host_batch_vector<T>& that)
    {
        if((this->batch_count() == that.batch_count()) && (this->n() == that.n())
           && (this->inc() == that.inc()))
        {
            size_t num_bytes = this->n() * std::abs(this->inc()) * sizeof(T);
            for(int batch_index = 0; batch_index < this->m_batch_count; ++batch_index)
            {
                memcpy((*this)[batch_index], that[batch_index], num_bytes);
            }
            return true;
        }
        else
        {
            return false;
        }
    }

    //!
    //! @brief Transfer from a device batched vector.
    //! @param that the vector the data is copied from.
    //! @return the hip error.
    //!
    hipError_t transfer_from(const device_batch_vector<T>& that)
    {
        hipError_t hip_err;
        size_t     num_bytes = size_t(this->m_n) * std::abs(this->m_inc) * sizeof(T);
        for(int batch_index = 0; batch_index < this->m_batch_count; ++batch_index)
        {
            if(hipSuccess
               != (hip_err = hipMemcpy(
                       (*this)[batch_index], that[batch_index], num_bytes, hipMemcpyDeviceToHost)))
            {
                return hip_err;
            }
        }
        return hipSuccess;
    }

    //!
    //! @brief Check if memory exists.
    //! @return hipSuccess if memory exists, hipErrorOutOfMemory otherwise.
    //!
    hipError_t memcheck() const
    {
        return (nullptr != this->m_data) ? hipSuccess : hipErrorOutOfMemory;
    }

private:
    int m_n{};
    int m_inc{};
    int m_batch_count{};
    T** m_data{};

    bool try_initialize_memory()
    {
        bool success = (nullptr != (this->m_data = (T**)calloc(this->m_batch_count, sizeof(T*))));
        if(success)
        {
            size_t nmemb = size_t(this->m_n) * std::abs(this->m_inc);
            for(int batch_index = 0; batch_index < this->m_batch_count; ++batch_index)
            {
                success = (nullptr != (this->m_data[batch_index] = (T*)calloc(nmemb, sizeof(T))));
                if(false == success)
                {
                    break;
                }
            }
        }
        return success;
    }

    void free_memory()
    {
        if(nullptr != this->m_data)
        {
            for(int batch_index = 0; batch_index < this->m_batch_count; ++batch_index)
            {
                if(nullptr != this->m_data[batch_index])
                {
                    free(this->m_data[batch_index]);
                    this->m_data[batch_index] = nullptr;
                }
            }

            free(this->m_data);
            this->m_data = nullptr;
        }
    }
};

#endif
