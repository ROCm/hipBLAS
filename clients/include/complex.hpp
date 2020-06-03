/* ************************************************************************
 * Copyright 2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#ifndef HIPBLAS_COMPLEX_HPP
#define HIPBLAS_COMPLEX_HPP

#include "hipblas.h"
#include <complex>

inline hipblasComplex& operator+=(hipblasComplex& lhs, const hipblasComplex& rhs)
{
    reinterpret_cast<std::complex<float>&>(lhs)
        += reinterpret_cast<const std::complex<float>&>(rhs);
    return lhs;
}

inline hipblasDoubleComplex& operator+=(hipblasDoubleComplex& lhs, const hipblasDoubleComplex& rhs)
{
    reinterpret_cast<std::complex<double>&>(lhs)
        += reinterpret_cast<const std::complex<double>&>(rhs);
    return lhs;
}

inline hipblasComplex operator+(hipblasComplex lhs, const hipblasComplex& rhs)
{
    return lhs += rhs;
}

inline hipblasDoubleComplex operator+(hipblasDoubleComplex lhs, const hipblasDoubleComplex& rhs)
{
    return lhs += rhs;
}

inline hipblasComplex& operator-=(hipblasComplex& lhs, const hipblasComplex& rhs)
{
    reinterpret_cast<std::complex<float>&>(lhs)
        -= reinterpret_cast<const std::complex<float>&>(rhs);
    return lhs;
}

inline hipblasDoubleComplex& operator-=(hipblasDoubleComplex& lhs, const hipblasDoubleComplex& rhs)
{
    reinterpret_cast<std::complex<double>&>(lhs)
        -= reinterpret_cast<const std::complex<double>&>(rhs);
    return lhs;
}

inline hipblasComplex operator-(hipblasComplex lhs, const hipblasComplex& rhs)
{
    return lhs -= rhs;
}

inline hipblasDoubleComplex operator-(hipblasDoubleComplex lhs, const hipblasDoubleComplex& rhs)
{
    return lhs -= rhs;
}

inline hipblasComplex& operator*=(hipblasComplex& lhs, const hipblasComplex& rhs)
{
    reinterpret_cast<std::complex<float>&>(lhs)
        *= reinterpret_cast<const std::complex<float>&>(rhs);
    return lhs;
}

inline hipblasDoubleComplex& operator*=(hipblasDoubleComplex& lhs, const hipblasDoubleComplex& rhs)
{
    reinterpret_cast<std::complex<double>&>(lhs)
        *= reinterpret_cast<const std::complex<double>&>(rhs);
    return lhs;
}

inline hipblasComplex operator*(hipblasComplex lhs, const hipblasComplex& rhs)
{
    return lhs *= rhs;
}

inline hipblasDoubleComplex operator*(hipblasDoubleComplex lhs, const hipblasDoubleComplex& rhs)
{
    return lhs *= rhs;
}

inline hipblasComplex& operator/=(hipblasComplex& lhs, const hipblasComplex& rhs)
{
    reinterpret_cast<std::complex<float>&>(lhs)
        /= reinterpret_cast<const std::complex<float>&>(rhs);
    return lhs;
}

inline hipblasDoubleComplex& operator/=(hipblasDoubleComplex& lhs, const hipblasDoubleComplex& rhs)
{
    reinterpret_cast<std::complex<double>&>(lhs)
        /= reinterpret_cast<const std::complex<double>&>(rhs);
    return lhs;
}

inline hipblasComplex operator/(hipblasComplex lhs, const hipblasComplex& rhs)
{
    return lhs /= rhs;
}

inline hipblasDoubleComplex operator/(hipblasDoubleComplex lhs, const hipblasDoubleComplex& rhs)
{
    return lhs /= rhs;
}

inline bool operator==(const hipblasComplex& lhs, const hipblasComplex& rhs)
{
    return reinterpret_cast<const std::complex<float>&>(lhs)
           == reinterpret_cast<const std::complex<float>&>(rhs);
}

inline bool operator!=(const hipblasComplex& lhs, const hipblasComplex& rhs)
{
    return !(lhs == rhs);
}

inline bool operator==(const hipblasDoubleComplex& lhs, const hipblasDoubleComplex& rhs)
{
    return reinterpret_cast<const std::complex<double>&>(lhs)
           == reinterpret_cast<const std::complex<double>&>(rhs);
}

inline bool operator!=(const hipblasDoubleComplex& lhs, const hipblasDoubleComplex& rhs)
{
    return !(lhs == rhs);
}

inline hipblasComplex operator-(const hipblasComplex& x)
{
    return {-x.real(), -x.imag()};
}

inline hipblasDoubleComplex operator-(const hipblasDoubleComplex& x)
{
    return {-x.real(), -x.imag()};
}

inline hipblasComplex operator+(const hipblasComplex& x)
{
    return x;
}

inline hipblasDoubleComplex operator+(const hipblasDoubleComplex& x)
{
    return x;
}

namespace std
{
    inline float real(const hipblasComplex& z)
    {
        return z.real();
    }

    inline double real(const hipblasDoubleComplex& z)
    {
        return z.real();
    }

    inline float imag(const hipblasComplex& z)
    {
        return z.imag();
    }

    inline double imag(const hipblasDoubleComplex& z)
    {
        return z.imag();
    }

    inline hipblasComplex conj(const hipblasComplex& z)
    {
        return {z.real(), -z.imag()};
    }

    inline hipblasDoubleComplex conj(const hipblasDoubleComplex& z)
    {
        return {z.real(), -z.imag()};
    }

    inline float abs(const hipblasComplex& z)
    {
        return abs(reinterpret_cast<const complex<float>&>(z));
    }

    inline double abs(const hipblasDoubleComplex& z)
    {
        return abs(reinterpret_cast<const complex<double>&>(z));
    }

    inline float conj(const float& r)
    {
        return r;
    }

    inline double conj(const double& r)
    {
        return r;
    }
}

#endif
