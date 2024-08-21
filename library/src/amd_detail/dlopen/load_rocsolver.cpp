/* ************************************************************************
 * Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "load_rocsolver.hpp"
#include "load_function.hpp"

#ifndef BUILD_WITH_SOLVER

#define LOAD_FN(FN_)                              \
    do                                            \
    {                                             \
        if(!load_function(handle, #FN_, g_##FN_)) \
            return false;                         \
    } while(0)

/**
 * function pointer variables to rocsolver functions used within hipBLAS
 */
fp_rocsolver_sgetrf                 g_rocsolver_sgetrf;
fp_rocsolver_dgetrf                 g_rocsolver_dgetrf;
fp_rocsolver_cgetrf                 g_rocsolver_cgetrf;
fp_rocsolver_zgetrf                 g_rocsolver_zgetrf;
fp_rocsolver_sgetrf_batched         g_rocsolver_sgetrf_batched;
fp_rocsolver_dgetrf_batched         g_rocsolver_dgetrf_batched;
fp_rocsolver_cgetrf_batched         g_rocsolver_cgetrf_batched;
fp_rocsolver_zgetrf_batched         g_rocsolver_zgetrf_batched;
fp_rocsolver_sgetrf_strided_batched g_rocsolver_sgetrf_strided_batched;
fp_rocsolver_dgetrf_strided_batched g_rocsolver_dgetrf_strided_batched;
fp_rocsolver_cgetrf_strided_batched g_rocsolver_cgetrf_strided_batched;
fp_rocsolver_zgetrf_strided_batched g_rocsolver_zgetrf_strided_batched;

fp_rocsolver_sgetrf_npvt                 g_rocsolver_sgetrf_npvt;
fp_rocsolver_dgetrf_npvt                 g_rocsolver_dgetrf_npvt;
fp_rocsolver_cgetrf_npvt                 g_rocsolver_cgetrf_npvt;
fp_rocsolver_zgetrf_npvt                 g_rocsolver_zgetrf_npvt;
fp_rocsolver_sgetrf_npvt_batched         g_rocsolver_sgetrf_npvt_batched;
fp_rocsolver_dgetrf_npvt_batched         g_rocsolver_dgetrf_npvt_batched;
fp_rocsolver_cgetrf_npvt_batched         g_rocsolver_cgetrf_npvt_batched;
fp_rocsolver_zgetrf_npvt_batched         g_rocsolver_zgetrf_npvt_batched;
fp_rocsolver_sgetrf_npvt_strided_batched g_rocsolver_sgetrf_npvt_strided_batched;
fp_rocsolver_dgetrf_npvt_strided_batched g_rocsolver_dgetrf_npvt_strided_batched;
fp_rocsolver_cgetrf_npvt_strided_batched g_rocsolver_cgetrf_npvt_strided_batched;
fp_rocsolver_zgetrf_npvt_strided_batched g_rocsolver_zgetrf_npvt_strided_batched;

fp_rocsolver_sgetrs                 g_rocsolver_sgetrs;
fp_rocsolver_dgetrs                 g_rocsolver_dgetrs;
fp_rocsolver_cgetrs                 g_rocsolver_cgetrs;
fp_rocsolver_zgetrs                 g_rocsolver_zgetrs;
fp_rocsolver_sgetrs_batched         g_rocsolver_sgetrs_batched;
fp_rocsolver_dgetrs_batched         g_rocsolver_dgetrs_batched;
fp_rocsolver_cgetrs_batched         g_rocsolver_cgetrs_batched;
fp_rocsolver_zgetrs_batched         g_rocsolver_zgetrs_batched;
fp_rocsolver_sgetrs_strided_batched g_rocsolver_sgetrs_strided_batched;
fp_rocsolver_dgetrs_strided_batched g_rocsolver_dgetrs_strided_batched;
fp_rocsolver_cgetrs_strided_batched g_rocsolver_cgetrs_strided_batched;
fp_rocsolver_zgetrs_strided_batched g_rocsolver_zgetrs_strided_batched;

fp_rocsolver_sgetri_outofplace_batched      g_rocsolver_sgetri_outofplace_batched;
fp_rocsolver_dgetri_outofplace_batched      g_rocsolver_dgetri_outofplace_batched;
fp_rocsolver_cgetri_outofplace_batched      g_rocsolver_cgetri_outofplace_batched;
fp_rocsolver_zgetri_outofplace_batched      g_rocsolver_zgetri_outofplace_batched;
fp_rocsolver_sgetri_npvt_outofplace_batched g_rocsolver_sgetri_npvt_outofplace_batched;
fp_rocsolver_dgetri_npvt_outofplace_batched g_rocsolver_dgetri_npvt_outofplace_batched;
fp_rocsolver_cgetri_npvt_outofplace_batched g_rocsolver_cgetri_npvt_outofplace_batched;
fp_rocsolver_zgetri_npvt_outofplace_batched g_rocsolver_zgetri_npvt_outofplace_batched;

fp_rocsolver_sgeqrf                 g_rocsolver_sgeqrf;
fp_rocsolver_dgeqrf                 g_rocsolver_dgeqrf;
fp_rocsolver_cgeqrf                 g_rocsolver_cgeqrf;
fp_rocsolver_zgeqrf                 g_rocsolver_zgeqrf;
fp_rocsolver_sgeqrf_ptr_batched     g_rocsolver_sgeqrf_ptr_batched;
fp_rocsolver_dgeqrf_ptr_batched     g_rocsolver_dgeqrf_ptr_batched;
fp_rocsolver_cgeqrf_ptr_batched     g_rocsolver_cgeqrf_ptr_batched;
fp_rocsolver_zgeqrf_ptr_batched     g_rocsolver_zgeqrf_ptr_batched;
fp_rocsolver_sgeqrf_strided_batched g_rocsolver_sgeqrf_strided_batched;
fp_rocsolver_dgeqrf_strided_batched g_rocsolver_dgeqrf_strided_batched;
fp_rocsolver_cgeqrf_strided_batched g_rocsolver_cgeqrf_strided_batched;
fp_rocsolver_zgeqrf_strided_batched g_rocsolver_zgeqrf_strided_batched;

fp_rocsolver_sgels                 g_rocsolver_sgels;
fp_rocsolver_dgels                 g_rocsolver_dgels;
fp_rocsolver_cgels                 g_rocsolver_cgels;
fp_rocsolver_zgels                 g_rocsolver_zgels;
fp_rocsolver_sgels_batched         g_rocsolver_sgels_batched;
fp_rocsolver_dgels_batched         g_rocsolver_dgels_batched;
fp_rocsolver_cgels_batched         g_rocsolver_cgels_batched;
fp_rocsolver_zgels_batched         g_rocsolver_zgels_batched;
fp_rocsolver_sgels_strided_batched g_rocsolver_sgels_strided_batched;
fp_rocsolver_dgels_strided_batched g_rocsolver_dgels_strided_batched;
fp_rocsolver_cgels_strided_batched g_rocsolver_cgels_strided_batched;
fp_rocsolver_zgels_strided_batched g_rocsolver_zgels_strided_batched;

static bool load_rocsolver()
{
#ifdef WIN32
    void* handle = LoadLibraryW(L"rocsolver.dll");
#else

    // RTLD_NOW to load symbols now to avoid performance hit later
    // RTLD_GLOBAL to load geqrf_ptr_batched declared symbols in hipblas.cpp
    void* handle = dlopen("librocsolver.so.0", RTLD_NOW | RTLD_GLOBAL);
    char* err    = dlerror();
#endif

    if(!handle)
        return false;

    LOAD_FN(rocsolver_sgetrf);
    LOAD_FN(rocsolver_dgetrf);
    LOAD_FN(rocsolver_cgetrf);
    LOAD_FN(rocsolver_zgetrf);
    LOAD_FN(rocsolver_sgetrf_batched);
    LOAD_FN(rocsolver_dgetrf_batched);
    LOAD_FN(rocsolver_cgetrf_batched);
    LOAD_FN(rocsolver_zgetrf_batched);
    LOAD_FN(rocsolver_sgetrf_strided_batched);
    LOAD_FN(rocsolver_dgetrf_strided_batched);
    LOAD_FN(rocsolver_cgetrf_strided_batched);
    LOAD_FN(rocsolver_zgetrf_strided_batched);

    LOAD_FN(rocsolver_sgetrf_npvt);
    LOAD_FN(rocsolver_dgetrf_npvt);
    LOAD_FN(rocsolver_cgetrf_npvt);
    LOAD_FN(rocsolver_zgetrf_npvt);
    LOAD_FN(rocsolver_sgetrf_npvt_batched);
    LOAD_FN(rocsolver_dgetrf_npvt_batched);
    LOAD_FN(rocsolver_cgetrf_npvt_batched);
    LOAD_FN(rocsolver_zgetrf_npvt_batched);
    LOAD_FN(rocsolver_sgetrf_npvt_strided_batched);
    LOAD_FN(rocsolver_dgetrf_npvt_strided_batched);
    LOAD_FN(rocsolver_cgetrf_npvt_strided_batched);
    LOAD_FN(rocsolver_zgetrf_npvt_strided_batched);

    LOAD_FN(rocsolver_sgetrs);
    LOAD_FN(rocsolver_dgetrs);
    LOAD_FN(rocsolver_cgetrs);
    LOAD_FN(rocsolver_zgetrs);
    LOAD_FN(rocsolver_sgetrs_batched);
    LOAD_FN(rocsolver_dgetrs_batched);
    LOAD_FN(rocsolver_cgetrs_batched);
    LOAD_FN(rocsolver_zgetrs_batched);
    LOAD_FN(rocsolver_sgetrs_strided_batched);
    LOAD_FN(rocsolver_dgetrs_strided_batched);
    LOAD_FN(rocsolver_cgetrs_strided_batched);
    LOAD_FN(rocsolver_zgetrs_strided_batched);

    LOAD_FN(rocsolver_sgetri_outofplace_batched);
    LOAD_FN(rocsolver_dgetri_outofplace_batched);
    LOAD_FN(rocsolver_cgetri_outofplace_batched);
    LOAD_FN(rocsolver_zgetri_outofplace_batched);
    LOAD_FN(rocsolver_sgetri_npvt_outofplace_batched);
    LOAD_FN(rocsolver_dgetri_npvt_outofplace_batched);
    LOAD_FN(rocsolver_cgetri_npvt_outofplace_batched);
    LOAD_FN(rocsolver_zgetri_npvt_outofplace_batched);

    LOAD_FN(rocsolver_sgeqrf);
    LOAD_FN(rocsolver_dgeqrf);
    LOAD_FN(rocsolver_cgeqrf);
    LOAD_FN(rocsolver_zgeqrf);
    LOAD_FN(rocsolver_sgeqrf_ptr_batched);
    LOAD_FN(rocsolver_dgeqrf_ptr_batched);
    LOAD_FN(rocsolver_cgeqrf_ptr_batched);
    LOAD_FN(rocsolver_zgeqrf_ptr_batched);
    LOAD_FN(rocsolver_sgeqrf_strided_batched);
    LOAD_FN(rocsolver_dgeqrf_strided_batched);
    LOAD_FN(rocsolver_cgeqrf_strided_batched);
    LOAD_FN(rocsolver_zgeqrf_strided_batched);

    LOAD_FN(rocsolver_sgels);
    LOAD_FN(rocsolver_dgels);
    LOAD_FN(rocsolver_cgels);
    LOAD_FN(rocsolver_zgels);
    LOAD_FN(rocsolver_sgels_batched);
    LOAD_FN(rocsolver_dgels_batched);
    LOAD_FN(rocsolver_cgels_batched);
    LOAD_FN(rocsolver_zgels_batched);
    LOAD_FN(rocsolver_sgels_strided_batched);
    LOAD_FN(rocsolver_dgels_strided_batched);
    LOAD_FN(rocsolver_cgels_strided_batched);
    LOAD_FN(rocsolver_zgels_strided_batched);

    return true;
}

#undef LOAD_FN

#endif // BUILD_WITH_SOLVER

bool try_load_rocsolver()
{
#ifndef BUILD_WITH_SOLVER
    static bool result = load_rocsolver();
    return result;
#else
    return true;
#endif
}
