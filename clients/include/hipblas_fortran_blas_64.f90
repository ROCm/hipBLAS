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
function hipblasSasum_64Fortran(handle, n, x, incx, result) &
    bind(c, name='hipblasSasum_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSasum_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    type(c_ptr), value :: result
            hipblasSasum_64Fortran = &
        hipblasSasum_64(handle, n, x, incx, result)
    return
end function hipblasSasum_64Fortran

function hipblasDasum_64Fortran(handle, n, x, incx, result) &
    bind(c, name='hipblasDasum_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDasum_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    type(c_ptr), value :: result
            hipblasDasum_64Fortran = &
        hipblasDasum_64(handle, n, x, incx, result)
    return
end function hipblasDasum_64Fortran

function hipblasScasum_64Fortran(handle, n, x, incx, result) &
    bind(c, name='hipblasScasum_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasScasum_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    type(c_ptr), value :: result
            hipblasScasum_64Fortran = &
        hipblasScasum_64(handle, n, x, incx, result)
    return
end function hipblasScasum_64Fortran

function hipblasDzasum_64Fortran(handle, n, x, incx, result) &
    bind(c, name='hipblasDzasum_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDzasum_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    type(c_ptr), value :: result
            hipblasDzasum_64Fortran = &
        hipblasDzasum_64(handle, n, x, incx, result)
    return
end function hipblasDzasum_64Fortran

! asumBatched
function hipblasSasumBatched_64Fortran(handle, n, x, incx, batch_count, result) &
    bind(c, name='hipblasSasumBatched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSasumBatched_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    integer(c_int64_t), value :: batch_count
    type(c_ptr), value :: result
            hipblasSasumBatched_64Fortran = &
        hipblasSasumBatched_64(handle, n, x, incx, batch_count, result)
    return
end function hipblasSasumBatched_64Fortran

function hipblasDasumBatched_64Fortran(handle, n, x, incx, batch_count, result) &
    bind(c, name='hipblasDasumBatched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDasumBatched_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    integer(c_int64_t), value :: batch_count
    type(c_ptr), value :: result
            hipblasDasumBatched_64Fortran = &
        hipblasDasumBatched_64(handle, n, x, incx, batch_count, result)
    return
end function hipblasDasumBatched_64Fortran

function hipblasScasumBatched_64Fortran(handle, n, x, incx, batch_count, result) &
    bind(c, name='hipblasScasumBatched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasScasumBatched_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    integer(c_int64_t), value :: batch_count
    type(c_ptr), value :: result
            hipblasScasumBatched_64Fortran = &
        hipblasScasumBatched_64(handle, n, x, incx, batch_count, result)
    return
end function hipblasScasumBatched_64Fortran

function hipblasDzasumBatched_64Fortran(handle, n, x, incx, batch_count, result) &
    bind(c, name='hipblasDzasumBatched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDzasumBatched_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    integer(c_int64_t), value :: batch_count
    type(c_ptr), value :: result
            hipblasDzasumBatched_64Fortran = &
        hipblasDzasumBatched_64(handle, n, x, incx, batch_count, result)
    return
end function hipblasDzasumBatched_64Fortran

! asumStridedBatched
function hipblasSasumStridedBatched_64Fortran(handle, n, x, incx, stride_x, batch_count, result) &
    bind(c, name='hipblasSasumStridedBatched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSasumStridedBatched_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    integer(c_int64_t), value :: stride_x
    integer(c_int64_t), value :: batch_count
    type(c_ptr), value :: result
            hipblasSasumStridedBatched_64Fortran = &
        hipblasSasumStridedBatched_64(handle, n, x, incx, stride_x, batch_count, result)
    return
end function hipblasSasumStridedBatched_64Fortran

function hipblasDasumStridedBatched_64Fortran(handle, n, x, incx, stride_x, batch_count, result) &
    bind(c, name='hipblasDasumStridedBatched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDasumStridedBatched_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    integer(c_int64_t), value :: stride_x
    integer(c_int64_t), value :: batch_count
    type(c_ptr), value :: result
            hipblasDasumStridedBatched_64Fortran = &
        hipblasDasumStridedBatched_64(handle, n, x, incx, stride_x, batch_count, result)
    return
end function hipblasDasumStridedBatched_64Fortran

function hipblasScasumStridedBatched_64Fortran(handle, n, x, incx, stride_x, batch_count, result) &
    bind(c, name='hipblasScasumStridedBatched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasScasumStridedBatched_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    integer(c_int64_t), value :: stride_x
    integer(c_int64_t), value :: batch_count
    type(c_ptr), value :: result
            hipblasScasumStridedBatched_64Fortran = &
        hipblasScasumStridedBatched_64(handle, n, x, incx, stride_x, batch_count, result)
    return
end function hipblasScasumStridedBatched_64Fortran

function hipblasDzasumStridedBatched_64Fortran(handle, n, x, incx, stride_x, batch_count, result) &
    bind(c, name='hipblasDzasumStridedBatched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDzasumStridedBatched_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    integer(c_int64_t), value :: stride_x
    integer(c_int64_t), value :: batch_count
    type(c_ptr), value :: result
            hipblasDzasumStridedBatched_64Fortran = &
        hipblasDzasumStridedBatched_64(handle, n, x, incx, stride_x, batch_count, result)
    return
end function hipblasDzasumStridedBatched_64Fortran

! axpy
function hipblasHaxpy_64Fortran(handle, n, alpha, x, incx, y, incy) &
    bind(c, name='hipblasHaxpy_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasHaxpy_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    type(c_ptr), value :: y
    integer(c_int64_t), value :: incy
            hipblasHaxpy_64Fortran = &
        hipblasHaxpy_64(handle, n, alpha, x, incx, y, incy)
    return
end function hipblasHaxpy_64Fortran

function hipblasSaxpy_64Fortran(handle, n, alpha, x, incx, y, incy) &
    bind(c, name='hipblasSaxpy_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSaxpy_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    type(c_ptr), value :: y
    integer(c_int64_t), value :: incy
            hipblasSaxpy_64Fortran = &
        hipblasSaxpy_64(handle, n, alpha, x, incx, y, incy)
    return
end function hipblasSaxpy_64Fortran

function hipblasDaxpy_64Fortran(handle, n, alpha, x, incx, y, incy) &
    bind(c, name='hipblasDaxpy_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDaxpy_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    type(c_ptr), value :: y
    integer(c_int64_t), value :: incy
            hipblasDaxpy_64Fortran = &
        hipblasDaxpy_64(handle, n, alpha, x, incx, y, incy)
    return
end function hipblasDaxpy_64Fortran

function hipblasCaxpy_64Fortran(handle, n, alpha, x, incx, y, incy) &
    bind(c, name='hipblasCaxpy_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCaxpy_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    type(c_ptr), value :: y
    integer(c_int64_t), value :: incy
            hipblasCaxpy_64Fortran = &
        hipblasCaxpy_64(handle, n, alpha, x, incx, y, incy)
    return
end function hipblasCaxpy_64Fortran

function hipblasZaxpy_64Fortran(handle, n, alpha, x, incx, y, incy) &
    bind(c, name='hipblasZaxpy_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZaxpy_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    type(c_ptr), value :: y
    integer(c_int64_t), value :: incy
            hipblasZaxpy_64Fortran = &
        hipblasZaxpy_64(handle, n, alpha, x, incx, y, incy)
    return
end function hipblasZaxpy_64Fortran

! axpyBatched
function hipblasHaxpyBatched_64Fortran(handle, n, alpha, x, incx, y, incy, batch_count) &
    bind(c, name='hipblasHaxpyBatched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasHaxpyBatched_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    type(c_ptr), value :: y
    integer(c_int64_t), value :: incy
    integer(c_int64_t), value :: batch_count
            hipblasHaxpyBatched_64Fortran = &
        hipblasHaxpyBatched_64(handle, n, alpha, x, incx, y, incy, batch_count)
    return
end function hipblasHaxpyBatched_64Fortran

function hipblasSaxpyBatched_64Fortran(handle, n, alpha, x, incx, y, incy, batch_count) &
    bind(c, name='hipblasSaxpyBatched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSaxpyBatched_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    type(c_ptr), value :: y
    integer(c_int64_t), value :: incy
    integer(c_int64_t), value :: batch_count
            hipblasSaxpyBatched_64Fortran = &
        hipblasSaxpyBatched_64(handle, n, alpha, x, incx, y, incy, batch_count)
    return
end function hipblasSaxpyBatched_64Fortran

function hipblasDaxpyBatched_64Fortran(handle, n, alpha, x, incx, y, incy, batch_count) &
    bind(c, name='hipblasDaxpyBatched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDaxpyBatched_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    type(c_ptr), value :: y
    integer(c_int64_t), value :: incy
    integer(c_int64_t), value :: batch_count
            hipblasDaxpyBatched_64Fortran = &
        hipblasDaxpyBatched_64(handle, n, alpha, x, incx, y, incy, batch_count)
    return
end function hipblasDaxpyBatched_64Fortran

function hipblasCaxpyBatched_64Fortran(handle, n, alpha, x, incx, y, incy, batch_count) &
    bind(c, name='hipblasCaxpyBatched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCaxpyBatched_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    type(c_ptr), value :: y
    integer(c_int64_t), value :: incy
    integer(c_int64_t), value :: batch_count
            hipblasCaxpyBatched_64Fortran = &
        hipblasCaxpyBatched_64(handle, n, alpha, x, incx, y, incy, batch_count)
    return
end function hipblasCaxpyBatched_64Fortran

function hipblasZaxpyBatched_64Fortran(handle, n, alpha, x, incx, y, incy, batch_count) &
    bind(c, name='hipblasZaxpyBatched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZaxpyBatched_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    type(c_ptr), value :: y
    integer(c_int64_t), value :: incy
    integer(c_int64_t), value :: batch_count
            hipblasZaxpyBatched_64Fortran = &
        hipblasZaxpyBatched_64(handle, n, alpha, x, incx, y, incy, batch_count)
    return
end function hipblasZaxpyBatched_64Fortran

! axpyStridedBatched
function hipblasHaxpyStridedBatched_64Fortran(handle, n, alpha, x, incx, stride_x, y, incy, stride_y, batch_count) &
    bind(c, name='hipblasHaxpyStridedBatched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasHaxpyStridedBatched_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    integer(c_int64_t), value :: stride_x
    type(c_ptr), value :: y
    integer(c_int64_t), value :: incy
    integer(c_int64_t), value :: stride_y
    integer(c_int64_t), value :: batch_count
            hipblasHaxpyStridedBatched_64Fortran = &
        hipblasHaxpyStridedBatched_64(handle, n, alpha, x, incx, stride_x, y, incy, stride_y, batch_count)
    return
end function hipblasHaxpyStridedBatched_64Fortran

function hipblasSaxpyStridedBatched_64Fortran(handle, n, alpha, x, incx, stride_x, y, incy, stride_y, batch_count) &
    bind(c, name='hipblasSaxpyStridedBatched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSaxpyStridedBatched_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    integer(c_int64_t), value :: stride_x
    type(c_ptr), value :: y
    integer(c_int64_t), value :: incy
    integer(c_int64_t), value :: stride_y
    integer(c_int64_t), value :: batch_count
            hipblasSaxpyStridedBatched_64Fortran = &
        hipblasSaxpyStridedBatched_64(handle, n, alpha, x, incx, stride_x, y, incy, stride_y, batch_count)
    return
end function hipblasSaxpyStridedBatched_64Fortran

function hipblasDaxpyStridedBatched_64Fortran(handle, n, alpha, x, incx, stride_x, y, incy, stride_y, batch_count) &
    bind(c, name='hipblasDaxpyStridedBatched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDaxpyStridedBatched_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    integer(c_int64_t), value :: stride_x
    type(c_ptr), value :: y
    integer(c_int64_t), value :: incy
    integer(c_int64_t), value :: stride_y
    integer(c_int64_t), value :: batch_count
            hipblasDaxpyStridedBatched_64Fortran = &
        hipblasDaxpyStridedBatched_64(handle, n, alpha, x, incx, stride_x, y, incy, stride_y, batch_count)
    return
end function hipblasDaxpyStridedBatched_64Fortran

function hipblasCaxpyStridedBatched_64Fortran(handle, n, alpha, x, incx, stride_x, y, incy, stride_y, batch_count) &
    bind(c, name='hipblasCaxpyStridedBatched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCaxpyStridedBatched_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    integer(c_int64_t), value :: stride_x
    type(c_ptr), value :: y
    integer(c_int64_t), value :: incy
    integer(c_int64_t), value :: stride_y
    integer(c_int64_t), value :: batch_count
            hipblasCaxpyStridedBatched_64Fortran = &
        hipblasCaxpyStridedBatched_64(handle, n, alpha, x, incx, stride_x, y, incy, stride_y, batch_count)
    return
end function hipblasCaxpyStridedBatched_64Fortran

function hipblasZaxpyStridedBatched_64Fortran(handle, n, alpha, x, incx, stride_x, y, incy, stride_y, batch_count) &
    bind(c, name='hipblasZaxpyStridedBatched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZaxpyStridedBatched_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    integer(c_int64_t), value :: stride_x
    type(c_ptr), value :: y
    integer(c_int64_t), value :: incy
    integer(c_int64_t), value :: stride_y
    integer(c_int64_t), value :: batch_count
            hipblasZaxpyStridedBatched_64Fortran = &
        hipblasZaxpyStridedBatched_64(handle, n, alpha, x, incx, stride_x, y, incy, stride_y, batch_count)
    return
end function hipblasZaxpyStridedBatched_64Fortran

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
        hipblasScopy_64(handle, n, x, incx, y, incy)
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
        hipblasDcopy_64(handle, n, x, incx, y, incy)
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
        hipblasCcopy_64(handle, n, x, incx, y, incy)
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
        hipblasZcopy_64(handle, n, x, incx, y, incy)
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
        hipblasScopyBatched_64(handle, n, x, incx, y, incy, batch_count)
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
        hipblasDcopyBatched_64(handle, n, x, incx, y, incy, batch_count)
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
        hipblasCcopyBatched_64(handle, n, x, incx, y, incy, batch_count)
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
        hipblasZcopyBatched_64(handle, n, x, incx, y, incy, batch_count)
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
        hipblasScopyStridedBatched_64(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count)
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
        hipblasDcopyStridedBatched_64(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count)
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
        hipblasCcopyStridedBatched_64(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count)
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
        hipblasZcopyStridedBatched_64(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count)
    return
end function hipblasZcopyStridedBatched_64Fortran

! dot
function hipblasSdot_64Fortran(handle, n, x, incx, y, incy, result) &
    bind(c, name='hipblasSdot_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSdot_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    type(c_ptr), value :: y
    integer(c_int64_t), value :: incy
    type(c_ptr), value :: result
            hipblasSdot_64Fortran = &
        hipblasSdot_64(handle, n, x, incx, y, incy, result)
    return
end function hipblasSdot_64Fortran

function hipblasDdot_64Fortran(handle, n, x, incx, y, incy, result) &
    bind(c, name='hipblasDdot_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDdot_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    type(c_ptr), value :: y
    integer(c_int64_t), value :: incy
    type(c_ptr), value :: result
            hipblasDdot_64Fortran = &
        hipblasDdot_64(handle, n, x, incx, y, incy, result)
    return
end function hipblasDdot_64Fortran

function hipblasHdot_64Fortran(handle, n, x, incx, y, incy, result) &
    bind(c, name='hipblasHdot_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasHdot_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    type(c_ptr), value :: y
    integer(c_int64_t), value :: incy
    type(c_ptr), value :: result
            hipblasHdot_64Fortran = &
        hipblasHdot_64(handle, n, x, incx, y, incy, result)
    return
end function hipblasHdot_64Fortran

function hipblasBfdot_64Fortran(handle, n, x, incx, y, incy, result) &
    bind(c, name='hipblasBfdot_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasBfdot_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    type(c_ptr), value :: y
    integer(c_int64_t), value :: incy
    type(c_ptr), value :: result
            hipblasBfdot_64Fortran = &
        hipblasBfdot_64(handle, n, x, incx, y, incy, result)
    return
end function hipblasBfdot_64Fortran

function hipblasCdotu_64Fortran(handle, n, x, incx, y, incy, result) &
    bind(c, name='hipblasCdotu_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCdotu_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    type(c_ptr), value :: y
    integer(c_int64_t), value :: incy
    type(c_ptr), value :: result
            hipblasCdotu_64Fortran = &
        hipblasCdotu_64(handle, n, x, incx, y, incy, result)
    return
end function hipblasCdotu_64Fortran

function hipblasCdotc_64Fortran(handle, n, x, incx, y, incy, result) &
    bind(c, name='hipblasCdotc_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCdotc_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    type(c_ptr), value :: y
    integer(c_int64_t), value :: incy
    type(c_ptr), value :: result
            hipblasCdotc_64Fortran = &
        hipblasCdotc_64(handle, n, x, incx, y, incy, result)
    return
end function hipblasCdotc_64Fortran

function hipblasZdotu_64Fortran(handle, n, x, incx, y, incy, result) &
    bind(c, name='hipblasZdotu_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZdotu_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    type(c_ptr), value :: y
    integer(c_int64_t), value :: incy
    type(c_ptr), value :: result
            hipblasZdotu_64Fortran = &
        hipblasZdotu_64(handle, n, x, incx, y, incy, result)
    return
end function hipblasZdotu_64Fortran

function hipblasZdotc_64Fortran(handle, n, x, incx, y, incy, result) &
    bind(c, name='hipblasZdotc_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZdotc_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    type(c_ptr), value :: y
    integer(c_int64_t), value :: incy
    type(c_ptr), value :: result
            hipblasZdotc_64Fortran = &
        hipblasZdotc_64(handle, n, x, incx, y, incy, result)
    return
end function hipblasZdotc_64Fortran

! dotBatched
function hipblasSdotBatched_64Fortran(handle, n, x, incx, y, incy, batch_count, result) &
    bind(c, name='hipblasSdotBatched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSdotBatched_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    type(c_ptr), value :: y
    integer(c_int64_t), value :: incy
    integer(c_int64_t), value :: batch_count
    type(c_ptr), value :: result
            hipblasSdotBatched_64Fortran = &
        hipblasSdotBatched_64(handle, n, x, incx, y, incy, batch_count, result)
    return
end function hipblasSdotBatched_64Fortran

function hipblasDdotBatched_64Fortran(handle, n, x, incx, y, incy, batch_count, result) &
    bind(c, name='hipblasDdotBatched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDdotBatched_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    type(c_ptr), value :: y
    integer(c_int64_t), value :: incy
    integer(c_int64_t), value :: batch_count
    type(c_ptr), value :: result
            hipblasDdotBatched_64Fortran = &
        hipblasDdotBatched_64(handle, n, x, incx, y, incy, batch_count, result)
    return
end function hipblasDdotBatched_64Fortran

function hipblasHdotBatched_64Fortran(handle, n, x, incx, y, incy, batch_count, result) &
    bind(c, name='hipblasHdotBatched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasHdotBatched_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    type(c_ptr), value :: y
    integer(c_int64_t), value :: incy
    integer(c_int64_t), value :: batch_count
    type(c_ptr), value :: result
            hipblasHdotBatched_64Fortran = &
        hipblasHdotBatched_64(handle, n, x, incx, y, incy, batch_count, result)
    return
end function hipblasHdotBatched_64Fortran

function hipblasBfdotBatched_64Fortran(handle, n, x, incx, y, incy, batch_count, result) &
    bind(c, name='hipblasBfdotBatched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasBfdotBatched_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    type(c_ptr), value :: y
    integer(c_int64_t), value :: incy
    integer(c_int64_t), value :: batch_count
    type(c_ptr), value :: result
            hipblasBfdotBatched_64Fortran = &
        hipblasBfdotBatched_64(handle, n, x, incx, y, incy, batch_count, result)
    return
end function hipblasBfdotBatched_64Fortran

function hipblasCdotuBatched_64Fortran(handle, n, x, incx, y, incy, batch_count, result) &
    bind(c, name='hipblasCdotuBatched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCdotuBatched_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    type(c_ptr), value :: y
    integer(c_int64_t), value :: incy
    integer(c_int64_t), value :: batch_count
    type(c_ptr), value :: result
            hipblasCdotuBatched_64Fortran = &
        hipblasCdotuBatched_64(handle, n, x, incx, y, incy, batch_count, result)
    return
end function hipblasCdotuBatched_64Fortran

function hipblasCdotcBatched_64Fortran(handle, n, x, incx, y, incy, batch_count, result) &
    bind(c, name='hipblasCdotcBatched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCdotcBatched_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    type(c_ptr), value :: y
    integer(c_int64_t), value :: incy
    integer(c_int64_t), value :: batch_count
    type(c_ptr), value :: result
            hipblasCdotcBatched_64Fortran = &
        hipblasCdotcBatched_64(handle, n, x, incx, y, incy, batch_count, result)
    return
end function hipblasCdotcBatched_64Fortran

function hipblasZdotuBatched_64Fortran(handle, n, x, incx, y, incy, batch_count, result) &
    bind(c, name='hipblasZdotuBatched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZdotuBatched_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    type(c_ptr), value :: y
    integer(c_int64_t), value :: incy
    integer(c_int64_t), value :: batch_count
    type(c_ptr), value :: result
            hipblasZdotuBatched_64Fortran = &
        hipblasZdotuBatched_64(handle, n, x, incx, y, incy, batch_count, result)
    return
end function hipblasZdotuBatched_64Fortran

function hipblasZdotcBatched_64Fortran(handle, n, x, incx, y, incy, batch_count, result) &
    bind(c, name='hipblasZdotcBatched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZdotcBatched_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    type(c_ptr), value :: y
    integer(c_int64_t), value :: incy
    integer(c_int64_t), value :: batch_count
    type(c_ptr), value :: result
            hipblasZdotcBatched_64Fortran = &
        hipblasZdotcBatched_64(handle, n, x, incx, y, incy, batch_count, result)
    return
end function hipblasZdotcBatched_64Fortran

! dotStridedBatched
function hipblasSdotStridedBatched_64Fortran(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count, result) &
    bind(c, name='hipblasSdotStridedBatched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSdotStridedBatched_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    integer(c_int64_t), value :: stride_x
    type(c_ptr), value :: y
    integer(c_int64_t), value :: incy
    integer(c_int64_t), value :: stride_y
    integer(c_int64_t), value :: batch_count
    type(c_ptr), value :: result
            hipblasSdotStridedBatched_64Fortran = &
        hipblasSdotStridedBatched_64(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count, result)
    return
end function hipblasSdotStridedBatched_64Fortran

function hipblasDdotStridedBatched_64Fortran(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count, result) &
    bind(c, name='hipblasDdotStridedBatched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDdotStridedBatched_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    integer(c_int64_t), value :: stride_x
    type(c_ptr), value :: y
    integer(c_int64_t), value :: incy
    integer(c_int64_t), value :: stride_y
    integer(c_int64_t), value :: batch_count
    type(c_ptr), value :: result
            hipblasDdotStridedBatched_64Fortran = &
        hipblasDdotStridedBatched_64(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count, result)
    return
end function hipblasDdotStridedBatched_64Fortran

function hipblasHdotStridedBatched_64Fortran(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count, result) &
    bind(c, name='hipblasHdotStridedBatched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasHdotStridedBatched_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    integer(c_int64_t), value :: stride_x
    type(c_ptr), value :: y
    integer(c_int64_t), value :: incy
    integer(c_int64_t), value :: stride_y
    integer(c_int64_t), value :: batch_count
    type(c_ptr), value :: result
            hipblasHdotStridedBatched_64Fortran = &
        hipblasHdotStridedBatched_64(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count, result)
    return
end function hipblasHdotStridedBatched_64Fortran

function hipblasBfdotStridedBatched_64Fortran(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count, result) &
    bind(c, name='hipblasBfdotStridedBatched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasBfdotStridedBatched_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    integer(c_int64_t), value :: stride_x
    type(c_ptr), value :: y
    integer(c_int64_t), value :: incy
    integer(c_int64_t), value :: stride_y
    integer(c_int64_t), value :: batch_count
    type(c_ptr), value :: result
            hipblasBfdotStridedBatched_64Fortran = &
        hipblasBfdotStridedBatched_64(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count, result)
    return
end function hipblasBfdotStridedBatched_64Fortran

function hipblasCdotuStridedBatched_64Fortran(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count, result) &
    bind(c, name='hipblasCdotuStridedBatched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCdotuStridedBatched_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    integer(c_int64_t), value :: stride_x
    type(c_ptr), value :: y
    integer(c_int64_t), value :: incy
    integer(c_int64_t), value :: stride_y
    integer(c_int64_t), value :: batch_count
    type(c_ptr), value :: result
            hipblasCdotuStridedBatched_64Fortran = &
        hipblasCdotuStridedBatched_64(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count, result)
    return
end function hipblasCdotuStridedBatched_64Fortran

function hipblasCdotcStridedBatched_64Fortran(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count, result) &
    bind(c, name='hipblasCdotcStridedBatched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCdotcStridedBatched_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    integer(c_int64_t), value :: stride_x
    type(c_ptr), value :: y
    integer(c_int64_t), value :: incy
    integer(c_int64_t), value :: stride_y
    integer(c_int64_t), value :: batch_count
    type(c_ptr), value :: result
            hipblasCdotcStridedBatched_64Fortran = &
        hipblasCdotcStridedBatched_64(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count, result)
    return
end function hipblasCdotcStridedBatched_64Fortran

function hipblasZdotuStridedBatched_64Fortran(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count, result) &
    bind(c, name='hipblasZdotuStridedBatched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZdotuStridedBatched_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    integer(c_int64_t), value :: stride_x
    type(c_ptr), value :: y
    integer(c_int64_t), value :: incy
    integer(c_int64_t), value :: stride_y
    integer(c_int64_t), value :: batch_count
    type(c_ptr), value :: result
            hipblasZdotuStridedBatched_64Fortran = &
        hipblasZdotuStridedBatched_64(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count, result)
    return
end function hipblasZdotuStridedBatched_64Fortran

function hipblasZdotcStridedBatched_64Fortran(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count, result) &
    bind(c, name='hipblasZdotcStridedBatched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZdotcStridedBatched_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    integer(c_int64_t), value :: stride_x
    type(c_ptr), value :: y
    integer(c_int64_t), value :: incy
    integer(c_int64_t), value :: stride_y
    integer(c_int64_t), value :: batch_count
    type(c_ptr), value :: result
            hipblasZdotcStridedBatched_64Fortran = &
        hipblasZdotcStridedBatched_64(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count, result)
    return
end function hipblasZdotcStridedBatched_64Fortran

! nrm2

! nrm2Batched

! nrm2StridedBatched

! rot
function hipblasSrot_64Fortran(handle, n, x, incx, y, incy, c, s) &
    bind(c, name='hipblasSrot_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSrot_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    type(c_ptr), value :: y
    integer(c_int64_t), value :: incy
    type(c_ptr), value :: c
    type(c_ptr), value :: s
            hipblasSrot_64Fortran = &
        hipblasSrot_64(handle, n, x, incx, y, incy, c, s)
    return
end function hipblasSrot_64Fortran

function hipblasDrot_64Fortran(handle, n, x, incx, y, incy, c, s) &
    bind(c, name='hipblasDrot_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDrot_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    type(c_ptr), value :: y
    integer(c_int64_t), value :: incy
    type(c_ptr), value :: c
    type(c_ptr), value :: s
            hipblasDrot_64Fortran = &
        hipblasDrot_64(handle, n, x, incx, y, incy, c, s)
    return
end function hipblasDrot_64Fortran

function hipblasCrot_64Fortran(handle, n, x, incx, y, incy, c, s) &
    bind(c, name='hipblasCrot_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCrot_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    type(c_ptr), value :: y
    integer(c_int64_t), value :: incy
    type(c_ptr), value :: c
    type(c_ptr), value :: s
            hipblasCrot_64Fortran = &
        hipblasCrot_64(handle, n, x, incx, y, incy, c, s)
    return
end function hipblasCrot_64Fortran

function hipblasCsrot_64Fortran(handle, n, x, incx, y, incy, c, s) &
    bind(c, name='hipblasCsrot_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCsrot_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    type(c_ptr), value :: y
    integer(c_int64_t), value :: incy
    type(c_ptr), value :: c
    type(c_ptr), value :: s
            hipblasCsrot_64Fortran = &
        hipblasCsrot_64(handle, n, x, incx, y, incy, c, s)
    return
end function hipblasCsrot_64Fortran

function hipblasZrot_64Fortran(handle, n, x, incx, y, incy, c, s) &
    bind(c, name='hipblasZrot_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZrot_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    type(c_ptr), value :: y
    integer(c_int64_t), value :: incy
    type(c_ptr), value :: c
    type(c_ptr), value :: s
            hipblasZrot_64Fortran = &
        hipblasZrot_64(handle, n, x, incx, y, incy, c, s)
    return
end function hipblasZrot_64Fortran

function hipblasZdrot_64Fortran(handle, n, x, incx, y, incy, c, s) &
    bind(c, name='hipblasZdrot_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZdrot_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    type(c_ptr), value :: y
    integer(c_int64_t), value :: incy
    type(c_ptr), value :: c
    type(c_ptr), value :: s
            hipblasZdrot_64Fortran = &
        hipblasZdrot_64(handle, n, x, incx, y, incy, c, s)
    return
end function hipblasZdrot_64Fortran

! rotBatched
function hipblasSrotBatched_64Fortran(handle, n, x, incx, y, incy, c, s, batch_count) &
    bind(c, name='hipblasSrotBatched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSrotBatched_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    type(c_ptr), value :: y
    integer(c_int64_t), value :: incy
    type(c_ptr), value :: c
    type(c_ptr), value :: s
    integer(c_int64_t), value :: batch_count
            hipblasSrotBatched_64Fortran = &
        hipblasSrotBatched_64(handle, n, x, incx, y, incy, c, s, batch_count)
    return
end function hipblasSrotBatched_64Fortran

function hipblasDrotBatched_64Fortran(handle, n, x, incx, y, incy, c, s, batch_count) &
    bind(c, name='hipblasDrotBatched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDrotBatched_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    type(c_ptr), value :: y
    integer(c_int64_t), value :: incy
    type(c_ptr), value :: c
    type(c_ptr), value :: s
    integer(c_int64_t), value :: batch_count
            hipblasDrotBatched_64Fortran = &
        hipblasDrotBatched_64(handle, n, x, incx, y, incy, c, s, batch_count)
    return
end function hipblasDrotBatched_64Fortran

function hipblasCrotBatched_64Fortran(handle, n, x, incx, y, incy, c, s, batch_count) &
    bind(c, name='hipblasCrotBatched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCrotBatched_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    type(c_ptr), value :: y
    integer(c_int64_t), value :: incy
    type(c_ptr), value :: c
    type(c_ptr), value :: s
    integer(c_int64_t), value :: batch_count
            hipblasCrotBatched_64Fortran = &
        hipblasCrotBatched_64(handle, n, x, incx, y, incy, c, s, batch_count)
    return
end function hipblasCrotBatched_64Fortran

function hipblasCsrotBatched_64Fortran(handle, n, x, incx, y, incy, c, s, batch_count) &
    bind(c, name='hipblasCsrotBatched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCsrotBatched_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    type(c_ptr), value :: y
    integer(c_int64_t), value :: incy
    type(c_ptr), value :: c
    type(c_ptr), value :: s
    integer(c_int64_t), value :: batch_count
            hipblasCsrotBatched_64Fortran = &
        hipblasCsrotBatched_64(handle, n, x, incx, y, incy, c, s, batch_count)
    return
end function hipblasCsrotBatched_64Fortran

function hipblasZrotBatched_64Fortran(handle, n, x, incx, y, incy, c, s, batch_count) &
    bind(c, name='hipblasZrotBatched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZrotBatched_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    type(c_ptr), value :: y
    integer(c_int64_t), value :: incy
    type(c_ptr), value :: c
    type(c_ptr), value :: s
    integer(c_int64_t), value :: batch_count
            hipblasZrotBatched_64Fortran = &
        hipblasZrotBatched_64(handle, n, x, incx, y, incy, c, s, batch_count)
    return
end function hipblasZrotBatched_64Fortran

function hipblasZdrotBatched_64Fortran(handle, n, x, incx, y, incy, c, s, batch_count) &
    bind(c, name='hipblasZdrotBatched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZdrotBatched_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    type(c_ptr), value :: y
    integer(c_int64_t), value :: incy
    type(c_ptr), value :: c
    type(c_ptr), value :: s
    integer(c_int64_t), value :: batch_count
            hipblasZdrotBatched_64Fortran = &
        hipblasZdrotBatched_64(handle, n, x, incx, y, incy, c, s, batch_count)
    return
end function hipblasZdrotBatched_64Fortran

! rotStridedBatched
function hipblasSrotStridedBatched_64Fortran(handle, n, x, incx, stride_x, y, incy, stride_y, c, s, batch_count) &
    bind(c, name='hipblasSrotStridedBatched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSrotStridedBatched_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    integer(c_int64_t), value :: stride_x
    type(c_ptr), value :: y
    integer(c_int64_t), value :: incy
    integer(c_int64_t), value :: stride_y
    type(c_ptr), value :: c
    type(c_ptr), value :: s
    integer(c_int64_t), value :: batch_count
            hipblasSrotStridedBatched_64Fortran = &
        hipblasSrotStridedBatched_64(handle, n, x, incx, stride_x, y, incy, stride_y, c, s, batch_count)
    return
end function hipblasSrotStridedBatched_64Fortran

function hipblasDrotStridedBatched_64Fortran(handle, n, x, incx, stride_x, y, incy, stride_y, c, s, batch_count) &
    bind(c, name='hipblasDrotStridedBatched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDrotStridedBatched_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    integer(c_int64_t), value :: stride_x
    type(c_ptr), value :: y
    integer(c_int64_t), value :: incy
    integer(c_int64_t), value :: stride_y
    type(c_ptr), value :: c
    type(c_ptr), value :: s
    integer(c_int64_t), value :: batch_count
            hipblasDrotStridedBatched_64Fortran = &
        hipblasDrotStridedBatched_64(handle, n, x, incx, stride_x, y, incy, stride_y, c, s, batch_count)
    return
end function hipblasDrotStridedBatched_64Fortran

function hipblasCrotStridedBatched_64Fortran(handle, n, x, incx, stride_x, y, incy, stride_y, c, s, batch_count) &
    bind(c, name='hipblasCrotStridedBatched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCrotStridedBatched_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    integer(c_int64_t), value :: stride_x
    type(c_ptr), value :: y
    integer(c_int64_t), value :: incy
    integer(c_int64_t), value :: stride_y
    type(c_ptr), value :: c
    type(c_ptr), value :: s
    integer(c_int64_t), value :: batch_count
            hipblasCrotStridedBatched_64Fortran = &
        hipblasCrotStridedBatched_64(handle, n, x, incx, stride_x, y, incy, stride_y, c, s, batch_count)
    return
end function hipblasCrotStridedBatched_64Fortran

function hipblasCsrotStridedBatched_64Fortran(handle, n, x, incx, stride_x, y, incy, stride_y, c, s, batch_count) &
    bind(c, name='hipblasCsrotStridedBatched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCsrotStridedBatched_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    integer(c_int64_t), value :: stride_x
    type(c_ptr), value :: y
    integer(c_int64_t), value :: incy
    integer(c_int64_t), value :: stride_y
    type(c_ptr), value :: c
    type(c_ptr), value :: s
    integer(c_int64_t), value :: batch_count
            hipblasCsrotStridedBatched_64Fortran = &
        hipblasCsrotStridedBatched_64(handle, n, x, incx, stride_x, y, incy, stride_y, c, s, batch_count)
    return
end function hipblasCsrotStridedBatched_64Fortran

function hipblasZrotStridedBatched_64Fortran(handle, n, x, incx, stride_x, y, incy, stride_y, c, s, batch_count) &
    bind(c, name='hipblasZrotStridedBatched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZrotStridedBatched_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    integer(c_int64_t), value :: stride_x
    type(c_ptr), value :: y
    integer(c_int64_t), value :: incy
    integer(c_int64_t), value :: stride_y
    type(c_ptr), value :: c
    type(c_ptr), value :: s
    integer(c_int64_t), value :: batch_count
            hipblasZrotStridedBatched_64Fortran = &
        hipblasZrotStridedBatched_64(handle, n, x, incx, stride_x, y, incy, stride_y, c, s, batch_count)
    return
end function hipblasZrotStridedBatched_64Fortran

function hipblasZdrotStridedBatched_64Fortran(handle, n, x, incx, stride_x, y, incy, stride_y, c, s, batch_count) &
    bind(c, name='hipblasZdrotStridedBatched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZdrotStridedBatched_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    integer(c_int64_t), value :: stride_x
    type(c_ptr), value :: y
    integer(c_int64_t), value :: incy
    integer(c_int64_t), value :: stride_y
    type(c_ptr), value :: c
    type(c_ptr), value :: s
    integer(c_int64_t), value :: batch_count
            hipblasZdrotStridedBatched_64Fortran = &
        hipblasZdrotStridedBatched_64(handle, n, x, incx, stride_x, y, incy, stride_y, c, s, batch_count)
    return
end function hipblasZdrotStridedBatched_64Fortran

! rotg
function hipblasSrotg_64Fortran(handle, a, b, c, s) &
    bind(c, name='hipblasSrotg_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSrotg_64Fortran
    type(c_ptr), value :: handle
    type(c_ptr), value :: a
    type(c_ptr), value :: b
    type(c_ptr), value :: c
    type(c_ptr), value :: s
            hipblasSrotg_64Fortran = &
        hipblasSrotg_64(handle, a, b, c, s)
    return
end function hipblasSrotg_64Fortran

function hipblasDrotg_64Fortran(handle, a, b, c, s) &
    bind(c, name='hipblasDrotg_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDrotg_64Fortran
    type(c_ptr), value :: handle
    type(c_ptr), value :: a
    type(c_ptr), value :: b
    type(c_ptr), value :: c
    type(c_ptr), value :: s
            hipblasDrotg_64Fortran = &
        hipblasDrotg_64(handle, a, b, c, s)
    return
end function hipblasDrotg_64Fortran

function hipblasCrotg_64Fortran(handle, a, b, c, s) &
    bind(c, name='hipblasCrotg_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCrotg_64Fortran
    type(c_ptr), value :: handle
    type(c_ptr), value :: a
    type(c_ptr), value :: b
    type(c_ptr), value :: c
    type(c_ptr), value :: s
            hipblasCrotg_64Fortran = &
        hipblasCrotg_64(handle, a, b, c, s)
    return
end function hipblasCrotg_64Fortran

function hipblasZrotg_64Fortran(handle, a, b, c, s) &
    bind(c, name='hipblasZrotg_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZrotg_64Fortran
    type(c_ptr), value :: handle
    type(c_ptr), value :: a
    type(c_ptr), value :: b
    type(c_ptr), value :: c
    type(c_ptr), value :: s
            hipblasZrotg_64Fortran = &
        hipblasZrotg_64(handle, a, b, c, s)
    return
end function hipblasZrotg_64Fortran

! rotgBatched
function hipblasSrotgBatched_64Fortran(handle, a, b, c, s, batch_count) &
    bind(c, name='hipblasSrotgBatched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSrotgBatched_64Fortran
    type(c_ptr), value :: handle
    type(c_ptr), value :: a
    type(c_ptr), value :: b
    type(c_ptr), value :: c
    type(c_ptr), value :: s
    integer(c_int64_t), value :: batch_count
            hipblasSrotgBatched_64Fortran = &
        hipblasSrotgBatched_64(handle, a, b, c, s, batch_count)
    return
end function hipblasSrotgBatched_64Fortran

function hipblasDrotgBatched_64Fortran(handle, a, b, c, s, batch_count) &
    bind(c, name='hipblasDrotgBatched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDrotgBatched_64Fortran
    type(c_ptr), value :: handle
    type(c_ptr), value :: a
    type(c_ptr), value :: b
    type(c_ptr), value :: c
    type(c_ptr), value :: s
    integer(c_int64_t), value :: batch_count
            hipblasDrotgBatched_64Fortran = &
        hipblasDrotgBatched_64(handle, a, b, c, s, batch_count)
    return
end function hipblasDrotgBatched_64Fortran

function hipblasCrotgBatched_64Fortran(handle, a, b, c, s, batch_count) &
    bind(c, name='hipblasCrotgBatched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCrotgBatched_64Fortran
    type(c_ptr), value :: handle
    type(c_ptr), value :: a
    type(c_ptr), value :: b
    type(c_ptr), value :: c
    type(c_ptr), value :: s
    integer(c_int64_t), value :: batch_count
            hipblasCrotgBatched_64Fortran = &
        hipblasCrotgBatched_64(handle, a, b, c, s, batch_count)
    return
end function hipblasCrotgBatched_64Fortran

function hipblasZrotgBatched_64Fortran(handle, a, b, c, s, batch_count) &
    bind(c, name='hipblasZrotgBatched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZrotgBatched_64Fortran
    type(c_ptr), value :: handle
    type(c_ptr), value :: a
    type(c_ptr), value :: b
    type(c_ptr), value :: c
    type(c_ptr), value :: s
    integer(c_int64_t), value :: batch_count
            hipblasZrotgBatched_64Fortran = &
        hipblasZrotgBatched_64(handle, a, b, c, s, batch_count)
    return
end function hipblasZrotgBatched_64Fortran

! rotgStridedBatched
function hipblasSrotgStridedBatched_64Fortran(handle, a, stride_a, b, stride_b, c, stride_c, s, stride_s, batch_count) &
    bind(c, name='hipblasSrotgStridedBatched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSrotgStridedBatched_64Fortran
    type(c_ptr), value :: handle
    type(c_ptr), value :: a
    integer(c_int64_t), value :: stride_a
    type(c_ptr), value :: b
    integer(c_int64_t), value :: stride_b
    type(c_ptr), value :: c
    integer(c_int64_t), value :: stride_c
    type(c_ptr), value :: s
    integer(c_int64_t), value :: stride_s
    integer(c_int64_t), value :: batch_count
            hipblasSrotgStridedBatched_64Fortran = &
        hipblasSrotgStridedBatched_64(handle, a, stride_a, b, stride_b, c, stride_c, s, stride_s, batch_count)
    return
end function hipblasSrotgStridedBatched_64Fortran

function hipblasDrotgStridedBatched_64Fortran(handle, a, stride_a, b, stride_b, c, stride_c, s, stride_s, batch_count) &
    bind(c, name='hipblasDrotgStridedBatched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDrotgStridedBatched_64Fortran
    type(c_ptr), value :: handle
    type(c_ptr), value :: a
    integer(c_int64_t), value :: stride_a
    type(c_ptr), value :: b
    integer(c_int64_t), value :: stride_b
    type(c_ptr), value :: c
    integer(c_int64_t), value :: stride_c
    type(c_ptr), value :: s
    integer(c_int64_t), value :: stride_s
    integer(c_int64_t), value :: batch_count
            hipblasDrotgStridedBatched_64Fortran = &
        hipblasDrotgStridedBatched_64(handle, a, stride_a, b, stride_b, c, stride_c, s, stride_s, batch_count)
    return
end function hipblasDrotgStridedBatched_64Fortran

function hipblasCrotgStridedBatched_64Fortran(handle, a, stride_a, b, stride_b, c, stride_c, s, stride_s, batch_count) &
    bind(c, name='hipblasCrotgStridedBatched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCrotgStridedBatched_64Fortran
    type(c_ptr), value :: handle
    type(c_ptr), value :: a
    integer(c_int64_t), value :: stride_a
    type(c_ptr), value :: b
    integer(c_int64_t), value :: stride_b
    type(c_ptr), value :: c
    integer(c_int64_t), value :: stride_c
    type(c_ptr), value :: s
    integer(c_int64_t), value :: stride_s
    integer(c_int64_t), value :: batch_count
            hipblasCrotgStridedBatched_64Fortran = &
        hipblasCrotgStridedBatched_64(handle, a, stride_a, b, stride_b, c, stride_c, s, stride_s, batch_count)
    return
end function hipblasCrotgStridedBatched_64Fortran

function hipblasZrotgStridedBatched_64Fortran(handle, a, stride_a, b, stride_b, c, stride_c, s, stride_s, batch_count) &
    bind(c, name='hipblasZrotgStridedBatched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZrotgStridedBatched_64Fortran
    type(c_ptr), value :: handle
    type(c_ptr), value :: a
    integer(c_int64_t), value :: stride_a
    type(c_ptr), value :: b
    integer(c_int64_t), value :: stride_b
    type(c_ptr), value :: c
    integer(c_int64_t), value :: stride_c
    type(c_ptr), value :: s
    integer(c_int64_t), value :: stride_s
    integer(c_int64_t), value :: batch_count
            hipblasZrotgStridedBatched_64Fortran = &
        hipblasZrotgStridedBatched_64(handle, a, stride_a, b, stride_b, c, stride_c, s, stride_s, batch_count)
    return
end function hipblasZrotgStridedBatched_64Fortran

! rotm
function hipblasSrotm_64Fortran(handle, n, x, incx, y, incy, param) &
    bind(c, name='hipblasSrotm_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSrotm_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    type(c_ptr), value :: y
    integer(c_int64_t), value :: incy
    type(c_ptr), value :: param
            hipblasSrotm_64Fortran = &
        hipblasSrotm_64(handle, n, x, incx, y, incy, param)
    return
end function hipblasSrotm_64Fortran

function hipblasDrotm_64Fortran(handle, n, x, incx, y, incy, param) &
    bind(c, name='hipblasDrotm_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDrotm_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    type(c_ptr), value :: y
    integer(c_int64_t), value :: incy
    type(c_ptr), value :: param
            hipblasDrotm_64Fortran = &
        hipblasDrotm_64(handle, n, x, incx, y, incy, param)
    return
end function hipblasDrotm_64Fortran

! rotmBatched
function hipblasSrotmBatched_64Fortran(handle, n, x, incx, y, incy, param, batch_count) &
    bind(c, name='hipblasSrotmBatched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSrotmBatched_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    type(c_ptr), value :: y
    integer(c_int64_t), value :: incy
    type(c_ptr), value :: param
    integer(c_int64_t), value :: batch_count
            hipblasSrotmBatched_64Fortran = &
        hipblasSrotmBatched_64(handle, n, x, incx, y, incy, param, batch_count)
    return
end function hipblasSrotmBatched_64Fortran

function hipblasDrotmBatched_64Fortran(handle, n, x, incx, y, incy, param, batch_count) &
    bind(c, name='hipblasDrotmBatched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDrotmBatched_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    type(c_ptr), value :: y
    integer(c_int64_t), value :: incy
    type(c_ptr), value :: param
    integer(c_int64_t), value :: batch_count
            hipblasDrotmBatched_64Fortran = &
        hipblasDrotmBatched_64(handle, n, x, incx, y, incy, param, batch_count)
    return
end function hipblasDrotmBatched_64Fortran

! rotmStridedBatched
function hipblasSrotmStridedBatched_64Fortran(handle, n, x, incx, stride_x, y, incy, stride_y, param, &
                                            stride_param, batch_count) &
    bind(c, name='hipblasSrotmStridedBatched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSrotmStridedBatched_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    integer(c_int64_t), value :: stride_x
    type(c_ptr), value :: y
    integer(c_int64_t), value :: incy
    integer(c_int64_t), value :: stride_y
    type(c_ptr), value :: param
    integer(c_int64_t), value :: stride_param
    integer(c_int64_t), value :: batch_count
            hipblasSrotmStridedBatched_64Fortran = &
        hipblasSrotmStridedBatched_64(handle, n, x, incx, stride_x, y, incy, stride_y, param, &
                                    stride_param, batch_count)
    return
end function hipblasSrotmStridedBatched_64Fortran

function hipblasDrotmStridedBatched_64Fortran(handle, n, x, incx, stride_x, y, incy, stride_y, param, &
                                            stride_param, batch_count) &
    bind(c, name='hipblasDrotmStridedBatched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDrotmStridedBatched_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    integer(c_int64_t), value :: stride_x
    type(c_ptr), value :: y
    integer(c_int64_t), value :: incy
    integer(c_int64_t), value :: stride_y
    type(c_ptr), value :: param
    integer(c_int64_t), value :: stride_param
    integer(c_int64_t), value :: batch_count
            hipblasDrotmStridedBatched_64Fortran = &
        hipblasDrotmStridedBatched_64(handle, n, x, incx, stride_x, y, incy, stride_y, param, &
                                    stride_param, batch_count)
    return
end function hipblasDrotmStridedBatched_64Fortran

! rotmg
function hipblasSrotmg_64Fortran(handle, d1, d2, x1, y1, param) &
    bind(c, name='hipblasSrotmg_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSrotmg_64Fortran
    type(c_ptr), value :: handle
    type(c_ptr), value :: d1
    type(c_ptr), value :: d2
    type(c_ptr), value :: x1
    type(c_ptr), value :: y1
    type(c_ptr), value :: param
            hipblasSrotmg_64Fortran = &
        hipblasSrotmg_64(handle, d1, d2, x1, y1, param)
    return
end function hipblasSrotmg_64Fortran

function hipblasDrotmg_64Fortran(handle, d1, d2, x1, y1, param) &
    bind(c, name='hipblasDrotmg_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDrotmg_64Fortran
    type(c_ptr), value :: handle
    type(c_ptr), value :: d1
    type(c_ptr), value :: d2
    type(c_ptr), value :: x1
    type(c_ptr), value :: y1
    type(c_ptr), value :: param
            hipblasDrotmg_64Fortran = &
        hipblasDrotmg_64(handle, d1, d2, x1, y1, param)
    return
end function hipblasDrotmg_64Fortran

! rotmgBatched
function hipblasSrotmgBatched_64Fortran(handle, d1, d2, x1, y1, param, batch_count) &
    bind(c, name='hipblasSrotmgBatched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSrotmgBatched_64Fortran
    type(c_ptr), value :: handle
    type(c_ptr), value :: d1
    type(c_ptr), value :: d2
    type(c_ptr), value :: x1
    type(c_ptr), value :: y1
    type(c_ptr), value :: param
    integer(c_int64_t), value :: batch_count
            hipblasSrotmgBatched_64Fortran = &
        hipblasSrotmgBatched_64(handle, d1, d2, x1, y1, param, batch_count)
    return
end function hipblasSrotmgBatched_64Fortran

function hipblasDrotmgBatched_64Fortran(handle, d1, d2, x1, y1, param, batch_count) &
    bind(c, name='hipblasDrotmgBatched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDrotmgBatched_64Fortran
    type(c_ptr), value :: handle
    type(c_ptr), value :: d1
    type(c_ptr), value :: d2
    type(c_ptr), value :: x1
    type(c_ptr), value :: y1
    type(c_ptr), value :: param
    integer(c_int64_t), value :: batch_count
            hipblasDrotmgBatched_64Fortran = &
        hipblasDrotmgBatched_64(handle, d1, d2, x1, y1, param, batch_count)
    return
end function hipblasDrotmgBatched_64Fortran

! rotmgStridedBatched
function hipblasSrotmgStridedBatched_64Fortran(handle, d1, stride_d1, d2, stride_d2, x1, stride_x1, &
                                            y1, stride_y1, param, stride_param, batch_count) &
    bind(c, name='hipblasSrotmgStridedBatched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSrotmgStridedBatched_64Fortran
    type(c_ptr), value :: handle
    type(c_ptr), value :: d1
    integer(c_int64_t), value :: stride_d1
    type(c_ptr), value :: d2
    integer(c_int64_t), value :: stride_d2
    type(c_ptr), value :: x1
    integer(c_int64_t), value :: stride_x1
    type(c_ptr), value :: y1
    integer(c_int64_t), value :: stride_y1
    type(c_ptr), value :: param
    integer(c_int64_t), value :: stride_param
    integer(c_int64_t), value :: batch_count
            hipblasSrotmgStridedBatched_64Fortran = &
        hipblasSrotmgStridedBatched_64(handle, d1, stride_d1, d2, stride_d2, x1, stride_x1, y1, stride_y1, &
                                    param, stride_param, batch_count)
    return
end function hipblasSrotmgStridedBatched_64Fortran

function hipblasDrotmgStridedBatched_64Fortran(handle, d1, stride_d1, d2, stride_d2, x1, stride_x1, &
                                            y1, stride_y1, param, stride_param, batch_count) &
    bind(c, name='hipblasDrotmgStridedBatched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDrotmgStridedBatched_64Fortran
    type(c_ptr), value :: handle
    type(c_ptr), value :: d1
    integer(c_int64_t), value :: stride_d1
    type(c_ptr), value :: d2
    integer(c_int64_t), value :: stride_d2
    type(c_ptr), value :: x1
    integer(c_int64_t), value :: stride_x1
    type(c_ptr), value :: y1
    integer(c_int64_t), value :: stride_y1
    type(c_ptr), value :: param
    integer(c_int64_t), value :: stride_param
    integer(c_int64_t), value :: batch_count
            hipblasDrotmgStridedBatched_64Fortran = &
        hipblasDrotmgStridedBatched_64(handle, d1, stride_d1, d2, stride_d2, x1, stride_x1, y1, stride_y1, &
                                    param, stride_param, batch_count)
    return
end function hipblasDrotmgStridedBatched_64Fortran

! scal

! scalBatched

! scalStridedBatched

! swap

! swapBatched

! swapStridedBatched
