!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
!
! Permission is hereby granted, free of charge, to any person obtaining a copy
! of this software and associated documentation files (the "Software"), to deal
! in the Software without restriction, including without limitation the rights
! to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
! copies of the Software, and to permit persons to whom the Software is
! furnished to do so, subject to the following conditions:
!
! The above copyright notice and this permission notice shall be included in
! all copies or substantial portions of the Software.
!
! THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
! IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
! FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
! AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
! LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
! OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
! THE SOFTWARE.
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

!--------!
! blas 1 !
!--------!

! amax

! amaxBatched

! amaxStridedBatched

! amin

! aminBathced

! aminStridedBatched

! asum

! asumBatched

! asumStridedBatched

! axpy

! axpyBatched

! axpyStridedBatched

! copy
function hipblasScopy_64Fortran(handle, n, x, incx, y, incy) &
    bind(c, name='hipblasScopy_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasScopy_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    type(c_ptr), value :: y
    integer(c_int64_t), value :: incy
            hipblasScopy_64Fortran = &
        hipblasScopy(handle, n, x, incx, y, incy)
    return
end function hipblasScopy_64Fortran

function hipblasDcopy_64Fortran(handle, n, x, incx, y, incy) &
    bind(c, name='hipblasDcopy_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDcopy_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    type(c_ptr), value :: y
    integer(c_int64_t), value :: incy
            hipblasDcopy_64Fortran = &
        hipblasDcopy(handle, n, x, incx, y, incy)
    return
end function hipblasDcopy_64Fortran

function hipblasCcopy_64Fortran(handle, n, x, incx, y, incy) &
    bind(c, name='hipblasCcopy_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCcopy_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    type(c_ptr), value :: y
    integer(c_int64_t), value :: incy
            hipblasCcopy_64Fortran = &
        hipblasCcopy(handle, n, x, incx, y, incy)
    return
end function hipblasCcopy_64Fortran

function hipblasZcopy_64Fortran(handle, n, x, incx, y, incy) &
    bind(c, name='hipblasZcopy_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZcopy_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    type(c_ptr), value :: y
    integer(c_int64_t), value :: incy
            hipblasZcopy_64Fortran = &
        hipblasZcopy(handle, n, x, incx, y, incy)
    return
end function hipblasZcopy_64Fortran

! copyBatched
function hipblasScopyBatched_64Fortran(handle, n, x, incx, y, incy, batch_count) &
    bind(c, name='hipblasScopyBatched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasScopyBatched_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    type(c_ptr), value :: y
    integer(c_int64_t), value :: incy
    integer(c_int64_t), value :: batch_count
            hipblasScopyBatched_64Fortran = &
        hipblasScopyBatched(handle, n, x, incx, y, incy, batch_count)
    return
end function hipblasScopyBatched_64Fortran

function hipblasDcopyBatched_64Fortran(handle, n, x, incx, y, incy, batch_count) &
    bind(c, name='hipblasDcopyBatched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDcopyBatched_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    type(c_ptr), value :: y
    integer(c_int64_t), value :: incy
    integer(c_int64_t), value :: batch_count
            hipblasDcopyBatched_64Fortran = &
        hipblasDcopyBatched(handle, n, x, incx, y, incy, batch_count)
    return
end function hipblasDcopyBatched_64Fortran

function hipblasCcopyBatched_64Fortran(handle, n, x, incx, y, incy, batch_count) &
    bind(c, name='hipblasCcopyBatched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCcopyBatched_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    type(c_ptr), value :: y
    integer(c_int64_t), value :: incy
    integer(c_int64_t), value :: batch_count
            hipblasCcopyBatched_64Fortran = &
        hipblasCcopyBatched(handle, n, x, incx, y, incy, batch_count)
    return
end function hipblasCcopyBatched_64Fortran

function hipblasZcopyBatched_64Fortran(handle, n, x, incx, y, incy, batch_count) &
    bind(c, name='hipblasZcopyBatched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZcopyBatched_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    type(c_ptr), value :: y
    integer(c_int64_t), value :: incy
    integer(c_int64_t), value :: batch_count
            hipblasZcopyBatched_64Fortran = &
        hipblasZcopyBatched(handle, n, x, incx, y, incy, batch_count)
    return
end function hipblasZcopyBatched_64Fortran

! copyStridedBatched
function hipblasScopyStridedBatched_64Fortran(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count) &
    bind(c, name='hipblasScopyStridedBatched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasScopyStridedBatched_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    integer(c_int64_t), value :: stride_x
    type(c_ptr), value :: y
    integer(c_int64_t), value :: incy
    integer(c_int64_t), value :: stride_y
    integer(c_int64_t), value :: batch_count
            hipblasScopyStridedBatched_64Fortran = &
        hipblasScopyStridedBatched(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count)
    return
end function hipblasScopyStridedBatched_64Fortran

function hipblasDcopyStridedBatched_64Fortran(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count) &
    bind(c, name='hipblasDcopyStridedBatched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDcopyStridedBatched_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    integer(c_int64_t), value :: stride_x
    type(c_ptr), value :: y
    integer(c_int64_t), value :: incy
    integer(c_int64_t), value :: stride_y
    integer(c_int64_t), value :: batch_count
            hipblasDcopyStridedBatched_64Fortran = &
        hipblasDcopyStridedBatched(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count)
    return
end function hipblasDcopyStridedBatched_64Fortran

function hipblasCcopyStridedBatched_64Fortran(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count) &
    bind(c, name='hipblasCcopyStridedBatched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCcopyStridedBatched_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    integer(c_int64_t), value :: stride_x
    type(c_ptr), value :: y
    integer(c_int64_t), value :: incy
    integer(c_int64_t), value :: stride_y
    integer(c_int64_t), value :: batch_count
            hipblasCcopyStridedBatched_64Fortran = &
        hipblasCcopyStridedBatched(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count)
    return
end function hipblasCcopyStridedBatched_64Fortran

function hipblasZcopyStridedBatched_64Fortran(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count) &
    bind(c, name='hipblasZcopyStridedBatched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZcopyStridedBatched_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    integer(c_int64_t), value :: stride_x
    type(c_ptr), value :: y
    integer(c_int64_t), value :: incy
    integer(c_int64_t), value :: stride_y
    integer(c_int64_t), value :: batch_count
            hipblasZcopyStridedBatched_64Fortran = &
        hipblasZcopyStridedBatched(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count)
    return
end function hipblasZcopyStridedBatched_64Fortran

! dot

! dotBatched

! dotStridedBatched

! nrm2

! nrm2Batched

! nrm2StridedBatched

! rot

! rotBatched

! rotStridedBatched

! rotg

! rotgBatched

! rotgStridedBatched

! rotm

! rotmBatched

! rotmStridedBatched

! rotmg

! rotmgBatchced

! rotmgStridedBatched

! scal

! scalBatched

! scalStridedBatched

! swap

! swapBatched

! swapStridedBatched
