
// Copyright (C) 2022 Advanced Micro Devices, Inc. All rights reserved.

// Script-generated file -- do not edit

#pragma once

#include "hipblas_arguments.hpp"

template <typename T>
inline inline hipblasStatus_t testing_asum(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_asum_batched(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_asum_strided_batched(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_axpy(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_axpy_batched(const Arguments& arg);

template <typename Ta, typename Txa, typename Tyx>
inline inline hipblasStatus_t testing_axpy_batched_ex_template(const Arguments& arg);

template <typename Ta, typename Txa, typename Tyx>
inline inline hipblasStatus_t testing_axpy_ex_template(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_axpy_strided_batched(const Arguments& arg);

template <typename Ta, typename Txa, typename Tyx>
inline inline hipblasStatus_t testing_axpy_strided_batched_ex_template(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_copy(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_copy_batched(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_copy_strided_batched(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_dgmm(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_dgmm_batched(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_dgmm_strided_batched(const Arguments& arg);

template <typename T, bool CONJ>
inline inline hipblasStatus_t testing_dot(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_dotc(const Arguments& arg);

template <typename T, bool CONJ>
inline inline hipblasStatus_t testing_dot_batched(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_dotc_batched(const Arguments& arg);

template <typename Tx, typename Tyx, typename Try, typename Texr, bool CONJ>
inline inline hipblasStatus_t testing_dot_batched_ex_template(const Arguments& arg);

template <typename Tx, typename Tyx, typename Try, typename Texr, bool CONJ>
inline inline hipblasStatus_t testing_dot_ex_template(const Arguments& arg);

template <typename T, bool CONJ>
inline inline hipblasStatus_t testing_dot_strided_batched(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_dotc_strided_batched(const Arguments& arg);

template <typename Tx, typename Tyx, typename Try, typename Texr, bool CONJ>
inline inline hipblasStatus_t testing_dot_strided_batched_ex_template(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_gbmv(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_gbmv_batched(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_gbmv_strided_batched(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_geam(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_geam_batched(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_geam_strided_batched(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_gels(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_gels_batched(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_gels_strided_batched(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_gemm(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_gemm_batched(const Arguments& arg);

template <typename Ta, typename Tba, typename Tcb, typename Texc>
inline inline hipblasStatus_t testing_gemm_batched_ex_template(const Arguments& arg);

template <typename Ta, typename Tba, typename Tcb, typename Texc>
inline inline hipblasStatus_t testing_gemm_ex_template(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_gemm_strided_batched(const Arguments& arg);

template <typename Ta, typename Tba, typename Tcb, typename Texc>
inline inline hipblasStatus_t testing_gemm_strided_batched_ex_template(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_gemv(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_gemv_batched(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_gemv_strided_batched(const Arguments& arg);

template <typename T, bool CONJ>
inline inline hipblasStatus_t testing_ger(const Arguments& arg);

template <typename T, bool CONJ>
inline inline hipblasStatus_t testing_ger_batched(const Arguments& arg);

template <typename T, bool CONJ>
inline inline hipblasStatus_t testing_ger_strided_batched(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_hbmv(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_hbmv_batched(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_hbmv_strided_batched(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_hemm(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_hemm_batched(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_hemm_strided_batched(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_hemv(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_hemv_batched(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_hemv_strided_batched(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_her(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_her2(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_her2_batched(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_her2_strided_batched(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_her2k(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_her2k_batched(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_her2k_strided_batched(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_her_batched(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_her_strided_batched(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_herk(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_herk_batched(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_herk_strided_batched(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_herkx(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_herkx_batched(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_herkx_strided_batched(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_hpmv(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_hpmv_batched(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_hpmv_strided_batched(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_hpr(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_hpr2(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_hpr2_batched(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_hpr2_strided_batched(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_hpr_batched(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_hpr_strided_batched(const Arguments& arg);

template <typename T>
using hipblas_iamax_iamin_t
    = hipblasStatus_t (*)(hipblasHandle_t handle, int n, const T* x, int incx, int* result);

template <typename T, void REFBLAS_FUNC(int, const T*, int, int*)>
inline inline hipblasStatus_t testing_iamax_iamin(const Arguments&         arg,
                                                  hipblas_iamax_iamin_t<T> func);

template <typename T>
inline inline hipblasStatus_t testing_amax(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_amin(const Arguments& arg);

template <typename T>
using hipblas_iamax_iamin_batched_t = hipblasStatus_t (*)(
    hipblasHandle_t handle, int n, const T* const x[], int incx, int batch_count, int* result);

template <typename T, void REFBLAS_FUNC(int, const T*, int, int*)>
inline inline hipblasStatus_t testing_iamax_iamin_batched(const Arguments&                 arg,
                                                          hipblas_iamax_iamin_batched_t<T> func);

template <typename T>
inline inline hipblasStatus_t testing_amax_batched(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_amin_batched(const Arguments& arg);

template <typename T>
using hipblas_iamax_iamin_strided_batched_t = hipblasStatus_t (*)(hipblasHandle_t handle,
                                                                  int             n,
                                                                  const T*        x,
                                                                  int             incx,
                                                                  hipblasStride   stridex,
                                                                  int             batch_count,
                                                                  int*            result);

template <typename T, void REFBLAS_FUNC(int, const T*, int, int*)>
inline inline hipblasStatus_t
    testing_iamax_iamin_strided_batched(const Arguments&                         arg,
                                        hipblas_iamax_iamin_strided_batched_t<T> func);

template <typename T>
inline inline hipblasStatus_t testing_amax_strided_batched(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_amin_strided_batched(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_nrm2(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_nrm2_batched(const Arguments& arg);

template <typename Tx, typename Trx, typename Texr>
inline inline hipblasStatus_t testing_nrm2_batched_ex_template(const Arguments& arg);

template <typename Tx, typename Trx, typename Texr>
inline inline hipblasStatus_t testing_nrm2_ex_template(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_nrm2_strided_batched(const Arguments& arg);

template <typename Tx, typename Trx, typename Texr>
inline inline hipblasStatus_t testing_nrm2_strided_batched_ex_template(const Arguments& arg);

template <typename T, typename U, typename V>
inline inline hipblasStatus_t testing_rot(const Arguments& arg);

template <typename T, typename U, typename V>
inline inline hipblasStatus_t testing_rot_batched(const Arguments& arg);

template <typename Tex, typename Txex, typename Tcsx>
inline inline hipblasStatus_t testing_rot_batched_ex_template(const Arguments& arg);

template <typename Tex, typename Txex, typename Tcsx>
inline inline hipblasStatus_t testing_rot_ex_template(const Arguments& arg);

template <typename T, typename U, typename V>
inline inline hipblasStatus_t testing_rot_strided_batched(const Arguments& arg);

template <typename Tex, typename Txex, typename Tcsx>
inline inline hipblasStatus_t testing_rot_strided_batched_ex_template(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_rotg(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_rotg_batched(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_rotg_strided_batched(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_rotm(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_rotm_batched(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_rotm_strided_batched(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_rotmg(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_rotmg_batched(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_rotmg_strided_batched(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_sbmv(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_sbmv_batched(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_sbmv_strided_batched(const Arguments& arg);

template <typename T, typename U>
inline inline hipblasStatus_t testing_scal(const Arguments& arg);

template <typename T, typename U>
inline inline hipblasStatus_t testing_scal_batched(const Arguments& arg);

template <typename Ta, typename Txa, typename Texx>
inline inline hipblasStatus_t testing_scal_batched_ex_template(const Arguments& arg);

template <typename Ta, typename Txa, typename Texx>
inline inline hipblasStatus_t testing_scal_ex_template(const Arguments& arg);

template <typename T, typename U>
inline inline hipblasStatus_t testing_scal_strided_batched(const Arguments& arg);

template <typename Ta, typename Txa, typename Texx>
inline inline hipblasStatus_t testing_scal_strided_batched_ex_template(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_set_get_matrix(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_set_get_matrix_async(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_set_get_vector(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_set_get_vector_async(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_spmv(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_spmv_batched(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_spmv_strided_batched(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_spr(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_spr2(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_spr2_batched(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_spr2_strided_batched(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_spr_batched(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_spr_strided_batched(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_swap(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_swap_batched(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_swap_strided_batched(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_symm(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_symm_batched(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_symm_strided_batched(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_symv(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_symv_batched(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_symv_strided_batched(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_syr(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_syr2(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_syr2_batched(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_syr2_strided_batched(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_syr2k(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_syr2k_batched(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_syr2k_strided_batched(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_syr_batched(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_syr_strided_batched(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_syrk(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_syrk_batched(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_syrk_strided_batched(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_syrkx(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_syrkx_batched(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_syrkx_strided_batched(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_tbmv(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_tbmv_batched(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_tbmv_strided_batched(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_tbsv(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_tbsv_batched(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_tbsv_strided_batched(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_tpmv(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_tpmv_batched(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_tpmv_strided_batched(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_tpsv(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_tpsv_batched(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_tpsv_strided_batched(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_trmm(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_trmm_batched(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_trmm_strided_batched(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_trmv(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_trmv_batched(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_trmv_strided_batched(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_trsm(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_trsm_batched(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_trsm_batched_ex(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_trsm_ex(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_trsm_strided_batched(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_trsm_strided_batched_ex(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_trsv(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_trsv_batched(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_trsv_strided_batched(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_trtri(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_trtri_batched(const Arguments& arg);

template <typename T>
inline inline hipblasStatus_t testing_trtri_strided_batched(const Arguments& arg);

// template <typename T>
// inline inline hipblasStatus_t testing_geqrf(const Arguments& arg);

// template <typename T>
// inline inline hipblasStatus_t testing_geqrf_batched(const Arguments& arg);

// template <typename T>
// inline inline hipblasStatus_t testing_geqrf_strided_batched(const Arguments& arg);

// template <typename T>
// inline inline hipblasStatus_t testing_getrf(const Arguments& arg);

// template <typename T>
// inline inline hipblasStatus_t testing_getrf_batched(const Arguments& arg);

// template <typename T>
// inline inline hipblasStatus_t testing_getrf_npvt(const Arguments& arg);

// template <typename T>
// inline inline hipblasStatus_t testing_getrf_npvt_batched(const Arguments& arg);

// template <typename T>
// inline inline hipblasStatus_t testing_getrf_npvt_strided_batched(const Arguments& arg);

// template <typename T>
// inline inline hipblasStatus_t testing_getrf_strided_batched(const Arguments& arg);

// template <typename T>
// inline inline hipblasStatus_t testing_getri_batched(const Arguments& arg);

// template <typename T>
// inline inline hipblasStatus_t testing_getri_npvt_batched(const Arguments& arg);

// template <typename T>
// inline inline hipblasStatus_t testing_getrs(const Arguments& arg);

// template <typename T>
// inline inline hipblasStatus_t testing_getrs_batched(const Arguments& arg);

// template <typename T>
// inline inline hipblasStatus_t testing_getrs_strided_batched(const Arguments& arg);
