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
function hipblasIsamax_64Fortran(handle, n, x, incx, result) &
    bind(c, name='hipblasIsamax_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasIsamax_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    type(c_ptr), value :: result
            hipblasIsamax_64Fortran = &
        hipblasIsamax_64(handle, n, x, incx, result)
    return
end function hipblasIsamax_64Fortran

function hipblasIdamax_64Fortran(handle, n, x, incx, result) &
    bind(c, name='hipblasIdamax_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasIdamax_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    type(c_ptr), value :: result
            hipblasIdamax_64Fortran = &
        hipblasIdamax_64(handle, n, x, incx, result)
    return
end function hipblasIdamax_64Fortran

function hipblasIcamax_64Fortran(handle, n, x, incx, result) &
    bind(c, name='hipblasIcamax_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasIcamax_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    type(c_ptr), value :: result
            hipblasIcamax_64Fortran = &
        hipblasIcamax_64(handle, n, x, incx, result)
    return
end function hipblasIcamax_64Fortran

function hipblasIzamax_64Fortran(handle, n, x, incx, result) &
    bind(c, name='hipblasIzamax_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasIzamax_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    type(c_ptr), value :: result
            hipblasIzamax_64Fortran = &
        hipblasIzamax_64(handle, n, x, incx, result)
    return
end function hipblasIzamax_64Fortran

! amaxBatched
function hipblasIsamaxBatched_64Fortran(handle, n, x, incx, batch_count, result) &
    bind(c, name='hipblasIsamaxBatched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasIsamaxBatched_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    integer(c_int64_t), value :: batch_count
    type(c_ptr), value :: result
            hipblasIsamaxBatched_64Fortran = &
        hipblasIsamaxBatched_64(handle, n, x, incx, batch_count, result)
    return
end function hipblasIsamaxBatched_64Fortran

function hipblasIdamaxBatched_64Fortran(handle, n, x, incx, batch_count, result) &
    bind(c, name='hipblasIdamaxBatched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasIdamaxBatched_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    integer(c_int64_t), value :: batch_count
    type(c_ptr), value :: result
            hipblasIdamaxBatched_64Fortran = &
        hipblasIdamaxBatched_64(handle, n, x, incx, batch_count, result)
    return
end function hipblasIdamaxBatched_64Fortran

function hipblasIcamaxBatched_64Fortran(handle, n, x, incx, batch_count, result) &
    bind(c, name='hipblasIcamaxBatched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasIcamaxBatched_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    integer(c_int64_t), value :: batch_count
    type(c_ptr), value :: result
            hipblasIcamaxBatched_64Fortran = &
        hipblasIcamaxBatched_64(handle, n, x, incx, batch_count, result)
    return
end function hipblasIcamaxBatched_64Fortran

function hipblasIzamaxBatched_64Fortran(handle, n, x, incx, batch_count, result) &
    bind(c, name='hipblasIzamaxBatched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasIzamaxBatched_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    integer(c_int64_t), value :: batch_count
    type(c_ptr), value :: result
            hipblasIzamaxBatched_64Fortran = &
        hipblasIzamaxBatched_64(handle, n, x, incx, batch_count, result)
    return
end function hipblasIzamaxBatched_64Fortran

! amaxStridedBatched
function hipblasIsamaxStridedBatched_64Fortran(handle, n, x, incx, stride_x, batch_count, result) &
    bind(c, name='hipblasIsamaxStridedBatched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasIsamaxStridedBatched_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    integer(c_int64_t), value :: stride_x
    integer(c_int64_t), value :: batch_count
    type(c_ptr), value :: result
            hipblasIsamaxStridedBatched_64Fortran = &
        hipblasIsamaxStridedBatched_64(handle, n, x, incx, stride_x, batch_count, result)
    return
end function hipblasIsamaxStridedBatched_64Fortran

function hipblasIdamaxStridedBatched_64Fortran(handle, n, x, incx, stride_x, batch_count, result) &
    bind(c, name='hipblasIdamaxStridedBatched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasIdamaxStridedBatched_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    integer(c_int64_t), value :: stride_x
    integer(c_int64_t), value :: batch_count
    type(c_ptr), value :: result
            hipblasIdamaxStridedBatched_64Fortran = &
        hipblasIdamaxStridedBatched_64(handle, n, x, incx, stride_x, batch_count, result)
    return
end function hipblasIdamaxStridedBatched_64Fortran

function hipblasIcamaxStridedBatched_64Fortran(handle, n, x, incx, stride_x, batch_count, result) &
    bind(c, name='hipblasIcamaxStridedBatched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasIcamaxStridedBatched_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    integer(c_int64_t), value :: stride_x
    integer(c_int64_t), value :: batch_count
    type(c_ptr), value :: result
            hipblasIcamaxStridedBatched_64Fortran = &
        hipblasIcamaxStridedBatched_64(handle, n, x, incx, stride_x, batch_count, result)
    return
end function hipblasIcamaxStridedBatched_64Fortran

function hipblasIzamaxStridedBatched_64Fortran(handle, n, x, incx, stride_x, batch_count, result) &
    bind(c, name='hipblasIzamaxStridedBatched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasIzamaxStridedBatched_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    integer(c_int64_t), value :: stride_x
    integer(c_int64_t), value :: batch_count
    type(c_ptr), value :: result
            hipblasIzamaxStridedBatched_64Fortran = &
        hipblasIzamaxStridedBatched_64(handle, n, x, incx, stride_x, batch_count, result)
    return
end function hipblasIzamaxStridedBatched_64Fortran

! amin
function hipblasIsamin_64Fortran(handle, n, x, incx, result) &
    bind(c, name='hipblasIsamin_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasIsamin_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    type(c_ptr), value :: result
            hipblasIsamin_64Fortran = &
        hipblasIsamin_64(handle, n, x, incx, result)
    return
end function hipblasIsamin_64Fortran

function hipblasIdamin_64Fortran(handle, n, x, incx, result) &
    bind(c, name='hipblasIdamin_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasIdamin_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    type(c_ptr), value :: result
            hipblasIdamin_64Fortran = &
        hipblasIdamin_64(handle, n, x, incx, result)
    return
end function hipblasIdamin_64Fortran

function hipblasIcamin_64Fortran(handle, n, x, incx, result) &
    bind(c, name='hipblasIcamin_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasIcamin_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    type(c_ptr), value :: result
            hipblasIcamin_64Fortran = &
        hipblasIcamin_64(handle, n, x, incx, result)
    return
end function hipblasIcamin_64Fortran

function hipblasIzamin_64Fortran(handle, n, x, incx, result) &
    bind(c, name='hipblasIzamin_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasIzamin_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    type(c_ptr), value :: result
            hipblasIzamin_64Fortran = &
        hipblasIzamin_64(handle, n, x, incx, result)
    return
end function hipblasIzamin_64Fortran

! aminBatched
function hipblasIsaminBatched_64Fortran(handle, n, x, incx, batch_count, result) &
    bind(c, name='hipblasIsaminBatched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasIsaminBatched_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    integer(c_int64_t), value :: batch_count
    type(c_ptr), value :: result
            hipblasIsaminBatched_64Fortran = &
        hipblasIsaminBatched_64(handle, n, x, incx, batch_count, result)
    return
end function hipblasIsaminBatched_64Fortran

function hipblasIdaminBatched_64Fortran(handle, n, x, incx, batch_count, result) &
    bind(c, name='hipblasIdaminBatched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasIdaminBatched_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    integer(c_int64_t), value :: batch_count
    type(c_ptr), value :: result
            hipblasIdaminBatched_64Fortran = &
        hipblasIdaminBatched_64(handle, n, x, incx, batch_count, result)
    return
end function hipblasIdaminBatched_64Fortran

function hipblasIcaminBatched_64Fortran(handle, n, x, incx, batch_count, result) &
    bind(c, name='hipblasIcaminBatched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasIcaminBatched_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    integer(c_int64_t), value :: batch_count
    type(c_ptr), value :: result
            hipblasIcaminBatched_64Fortran = &
        hipblasIcaminBatched_64(handle, n, x, incx, batch_count, result)
    return
end function hipblasIcaminBatched_64Fortran

function hipblasIzaminBatched_64Fortran(handle, n, x, incx, batch_count, result) &
    bind(c, name='hipblasIzaminBatched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasIzaminBatched_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    integer(c_int64_t), value :: batch_count
    type(c_ptr), value :: result
            hipblasIzaminBatched_64Fortran = &
        hipblasIzaminBatched_64(handle, n, x, incx, batch_count, result)
    return
end function hipblasIzaminBatched_64Fortran

! aminStridedBatched
function hipblasIsaminStridedBatched_64Fortran(handle, n, x, incx, stride_x, batch_count, result) &
    bind(c, name='hipblasIsaminStridedBatched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasIsaminStridedBatched_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    integer(c_int64_t), value :: stride_x
    integer(c_int64_t), value :: batch_count
    type(c_ptr), value :: result
            hipblasIsaminStridedBatched_64Fortran = &
        hipblasIsaminStridedBatched_64(handle, n, x, incx, stride_x, batch_count, result)
    return
end function hipblasIsaminStridedBatched_64Fortran

function hipblasIdaminStridedBatched_64Fortran(handle, n, x, incx, stride_x, batch_count, result) &
    bind(c, name='hipblasIdaminStridedBatched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasIdaminStridedBatched_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    integer(c_int64_t), value :: stride_x
    integer(c_int64_t), value :: batch_count
    type(c_ptr), value :: result
            hipblasIdaminStridedBatched_64Fortran = &
        hipblasIdaminStridedBatched_64(handle, n, x, incx, stride_x, batch_count, result)
    return
end function hipblasIdaminStridedBatched_64Fortran

function hipblasIcaminStridedBatched_64Fortran(handle, n, x, incx, stride_x, batch_count, result) &
    bind(c, name='hipblasIcaminStridedBatched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasIcaminStridedBatched_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    integer(c_int64_t), value :: stride_x
    integer(c_int64_t), value :: batch_count
    type(c_ptr), value :: result
            hipblasIcaminStridedBatched_64Fortran = &
        hipblasIcaminStridedBatched_64(handle, n, x, incx, stride_x, batch_count, result)
    return
end function hipblasIcaminStridedBatched_64Fortran

function hipblasIzaminStridedBatched_64Fortran(handle, n, x, incx, stride_x, batch_count, result) &
    bind(c, name='hipblasIzaminStridedBatched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasIzaminStridedBatched_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    integer(c_int64_t), value :: stride_x
    integer(c_int64_t), value :: batch_count
    type(c_ptr), value :: result
            hipblasIzaminStridedBatched_64Fortran = &
        hipblasIzaminStridedBatched_64(handle, n, x, incx, stride_x, batch_count, result)
    return
end function hipblasIzaminStridedBatched_64Fortran

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
function hipblasSnrm2_64Fortran(handle, n, x, incx, result) &
    bind(c, name='hipblasSnrm2_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSnrm2_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    type(c_ptr), value :: result
            hipblasSnrm2_64Fortran = &
        hipblasSnrm2_64(handle, n, x, incx, result)
    return
end function hipblasSnrm2_64Fortran

function hipblasDnrm2_64Fortran(handle, n, x, incx, result) &
    bind(c, name='hipblasDnrm2_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDnrm2_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    type(c_ptr), value :: result
            hipblasDnrm2_64Fortran = &
        hipblasDnrm2_64(handle, n, x, incx, result)
    return
end function hipblasDnrm2_64Fortran

function hipblasScnrm2_64Fortran(handle, n, x, incx, result) &
    bind(c, name='hipblasScnrm2_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasScnrm2_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    type(c_ptr), value :: result
            hipblasScnrm2_64Fortran = &
        hipblasScnrm2_64(handle, n, x, incx, result)
    return
end function hipblasScnrm2_64Fortran

function hipblasDznrm2_64Fortran(handle, n, x, incx, result) &
    bind(c, name='hipblasDznrm2_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDznrm2_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    type(c_ptr), value :: result
            hipblasDznrm2_64Fortran = &
        hipblasDznrm2_64(handle, n, x, incx, result)
    return
end function hipblasDznrm2_64Fortran

! nrm2Batched
function hipblasSnrm2Batched_64Fortran(handle, n, x, incx, batch_count, result) &
    bind(c, name='hipblasSnrm2Batched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSnrm2Batched_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    integer(c_int64_t), value :: batch_count
    type(c_ptr), value :: result
            hipblasSnrm2Batched_64Fortran = &
        hipblasSnrm2Batched_64(handle, n, x, incx, batch_count, result)
    return
end function hipblasSnrm2Batched_64Fortran

function hipblasDnrm2Batched_64Fortran(handle, n, x, incx, batch_count, result) &
    bind(c, name='hipblasDnrm2Batched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDnrm2Batched_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    integer(c_int64_t), value :: batch_count
    type(c_ptr), value :: result
            hipblasDnrm2Batched_64Fortran = &
        hipblasDnrm2Batched_64(handle, n, x, incx, batch_count, result)
    return
end function hipblasDnrm2Batched_64Fortran

function hipblasScnrm2Batched_64Fortran(handle, n, x, incx, batch_count, result) &
    bind(c, name='hipblasScnrm2Batched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasScnrm2Batched_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    integer(c_int64_t), value :: batch_count
    type(c_ptr), value :: result
            hipblasScnrm2Batched_64Fortran = &
        hipblasScnrm2Batched_64(handle, n, x, incx, batch_count, result)
    return
end function hipblasScnrm2Batched_64Fortran

function hipblasDznrm2Batched_64Fortran(handle, n, x, incx, batch_count, result) &
    bind(c, name='hipblasDznrm2Batched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDznrm2Batched_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    integer(c_int64_t), value :: batch_count
    type(c_ptr), value :: result
            hipblasDznrm2Batched_64Fortran = &
        hipblasDznrm2Batched_64(handle, n, x, incx, batch_count, result)
    return
end function hipblasDznrm2Batched_64Fortran

! nrm2StridedBatched
function hipblasSnrm2StridedBatched_64Fortran(handle, n, x, incx, stride_x, batch_count, result) &
    bind(c, name='hipblasSnrm2StridedBatched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSnrm2StridedBatched_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    integer(c_int64_t), value :: stride_x
    integer(c_int64_t), value :: batch_count
    type(c_ptr), value :: result
            hipblasSnrm2StridedBatched_64Fortran = &
        hipblasSnrm2StridedBatched_64(handle, n, x, incx, stride_x, batch_count, result)
    return
end function hipblasSnrm2StridedBatched_64Fortran

function hipblasDnrm2StridedBatched_64Fortran(handle, n, x, incx, stride_x, batch_count, result) &
    bind(c, name='hipblasDnrm2StridedBatched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDnrm2StridedBatched_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    integer(c_int64_t), value :: stride_x
    integer(c_int64_t), value :: batch_count
    type(c_ptr), value :: result
            hipblasDnrm2StridedBatched_64Fortran = &
        hipblasDnrm2StridedBatched_64(handle, n, x, incx, stride_x, batch_count, result)
    return
end function hipblasDnrm2StridedBatched_64Fortran

function hipblasScnrm2StridedBatched_64Fortran(handle, n, x, incx, stride_x, batch_count, result) &
    bind(c, name='hipblasScnrm2StridedBatched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasScnrm2StridedBatched_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    integer(c_int64_t), value :: stride_x
    integer(c_int64_t), value :: batch_count
    type(c_ptr), value :: result
            hipblasScnrm2StridedBatched_64Fortran = &
        hipblasScnrm2StridedBatched_64(handle, n, x, incx, stride_x, batch_count, result)
    return
end function hipblasScnrm2StridedBatched_64Fortran

function hipblasDznrm2StridedBatched_64Fortran(handle, n, x, incx, stride_x, batch_count, result) &
    bind(c, name='hipblasDznrm2StridedBatched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDznrm2StridedBatched_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    integer(c_int64_t), value :: stride_x
    integer(c_int64_t), value :: batch_count
    type(c_ptr), value :: result
            hipblasDznrm2StridedBatched_64Fortran = &
        hipblasDznrm2StridedBatched_64(handle, n, x, incx, stride_x, batch_count, result)
    return
end function hipblasDznrm2StridedBatched_64Fortran

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
function hipblasSscal_64Fortran(handle, n, alpha, x, incx) &
    bind(c, name='hipblasSscal_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSscal_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
            hipblasSscal_64Fortran = &
        hipblasSscal_64(handle, n, alpha, x, incx)
    return
end function hipblasSscal_64Fortran

function hipblasDscal_64Fortran(handle, n, alpha, x, incx) &
    bind(c, name='hipblasDscal_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDscal_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
            hipblasDscal_64Fortran = &
        hipblasDscal_64(handle, n, alpha, x, incx)
    return
end function hipblasDscal_64Fortran

function hipblasCscal_64Fortran(handle, n, alpha, x, incx) &
    bind(c, name='hipblasCscal_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCscal_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
            hipblasCscal_64Fortran = &
        hipblasCscal_64(handle, n, alpha, x, incx)
    return
end function hipblasCscal_64Fortran

function hipblasZscal_64Fortran(handle, n, alpha, x, incx) &
    bind(c, name='hipblasZscal_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZscal_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
            hipblasZscal_64Fortran = &
        hipblasZscal_64(handle, n, alpha, x, incx)
    return
end function hipblasZscal_64Fortran

function hipblasCsscal_64Fortran(handle, n, alpha, x, incx) &
    bind(c, name='hipblasCsscal_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCsscal_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
            hipblasCsscal_64Fortran = &
        hipblasCsscal_64(handle, n, alpha, x, incx)
    return
end function hipblasCsscal_64Fortran

function hipblasZdscal_64Fortran(handle, n, alpha, x, incx) &
    bind(c, name='hipblasZdscal_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZdscal_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
            hipblasZdscal_64Fortran = &
        hipblasZdscal_64(handle, n, alpha, x, incx)
    return
end function hipblasZdscal_64Fortran

! scalBatched
function hipblasSscalBatched_64Fortran(handle, n, alpha, x, incx, batch_count) &
    bind(c, name='hipblasSscalBatched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSscalBatched_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    integer(c_int64_t), value :: batch_count
            hipblasSscalBatched_64Fortran = &
        hipblasSscalBatched_64(handle, n, alpha, x, incx, batch_count)
    return
end function hipblasSscalBatched_64Fortran

function hipblasDscalBatched_64Fortran(handle, n, alpha, x, incx, batch_count) &
    bind(c, name='hipblasDscalBatched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDscalBatched_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    integer(c_int64_t), value :: batch_count
            hipblasDscalBatched_64Fortran = &
        hipblasDscalBatched_64(handle, n, alpha, x, incx, batch_count)
    return
end function hipblasDscalBatched_64Fortran

function hipblasCscalBatched_64Fortran(handle, n, alpha, x, incx, batch_count) &
    bind(c, name='hipblasCscalBatched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCscalBatched_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    integer(c_int64_t), value :: batch_count
            hipblasCscalBatched_64Fortran = &
        hipblasCscalBatched_64(handle, n, alpha, x, incx, batch_count)
    return
end function hipblasCscalBatched_64Fortran

function hipblasZscalBatched_64Fortran(handle, n, alpha, x, incx, batch_count) &
    bind(c, name='hipblasZscalBatched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZscalBatched_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    integer(c_int64_t), value :: batch_count
            hipblasZscalBatched_64Fortran = &
        hipblasZscalBatched_64(handle, n, alpha, x, incx, batch_count)
    return
end function hipblasZscalBatched_64Fortran

function hipblasCsscalBatched_64Fortran(handle, n, alpha, x, incx, batch_count) &
    bind(c, name='hipblasCsscalBatched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCsscalBatched_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    integer(c_int64_t), value :: batch_count
            hipblasCsscalBatched_64Fortran = &
        hipblasCsscalBatched_64(handle, n, alpha, x, incx, batch_count)
    return
end function hipblasCsscalBatched_64Fortran

function hipblasZdscalBatched_64Fortran(handle, n, alpha, x, incx, batch_count) &
    bind(c, name='hipblasZdscalBatched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZdscalBatched_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    integer(c_int64_t), value :: batch_count
            hipblasZdscalBatched_64Fortran = &
        hipblasZdscalBatched_64(handle, n, alpha, x, incx, batch_count)
    return
end function hipblasZdscalBatched_64Fortran

! scalStridedBatched
function hipblasSscalStridedBatched_64Fortran(handle, n, alpha, x, incx, stride_x, batch_count) &
    bind(c, name='hipblasSscalStridedBatched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSscalStridedBatched_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    integer(c_int64_t), value :: stride_x
    integer(c_int64_t), value :: batch_count
            hipblasSscalStridedBatched_64Fortran = &
        hipblasSscalStridedBatched_64(handle, n, alpha, x, incx, stride_x, batch_count)
    return
end function hipblasSscalStridedBatched_64Fortran

function hipblasDscalStridedBatched_64Fortran(handle, n, alpha, x, incx, stride_x, batch_count) &
    bind(c, name='hipblasDscalStridedBatched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDscalStridedBatched_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    integer(c_int64_t), value :: stride_x
    integer(c_int64_t), value :: batch_count
            hipblasDscalStridedBatched_64Fortran = &
        hipblasDscalStridedBatched_64(handle, n, alpha, x, incx, stride_x, batch_count)
    return
end function hipblasDscalStridedBatched_64Fortran

function hipblasCscalStridedBatched_64Fortran(handle, n, alpha, x, incx, stride_x, batch_count) &
    bind(c, name='hipblasCscalStridedBatched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCscalStridedBatched_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    integer(c_int64_t), value :: stride_x
    integer(c_int64_t), value :: batch_count
            hipblasCscalStridedBatched_64Fortran = &
        hipblasCscalStridedBatched_64(handle, n, alpha, x, incx, stride_x, batch_count)
    return
end function hipblasCscalStridedBatched_64Fortran

function hipblasZscalStridedBatched_64Fortran(handle, n, alpha, x, incx, stride_x, batch_count) &
    bind(c, name='hipblasZscalStridedBatched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZscalStridedBatched_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    integer(c_int64_t), value :: stride_x
    integer(c_int64_t), value :: batch_count
            hipblasZscalStridedBatched_64Fortran = &
        hipblasZscalStridedBatched_64(handle, n, alpha, x, incx, stride_x, batch_count)
    return
end function hipblasZscalStridedBatched_64Fortran

function hipblasCsscalStridedBatched_64Fortran(handle, n, alpha, x, incx, stride_x, batch_count) &
    bind(c, name='hipblasCsscalStridedBatched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCsscalStridedBatched_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    integer(c_int64_t), value :: stride_x
    integer(c_int64_t), value :: batch_count
            hipblasCsscalStridedBatched_64Fortran = &
        hipblasCsscalStridedBatched_64(handle, n, alpha, x, incx, stride_x, batch_count)
    return
end function hipblasCsscalStridedBatched_64Fortran

function hipblasZdscalStridedBatched_64Fortran(handle, n, alpha, x, incx, stride_x, batch_count) &
    bind(c, name='hipblasZdscalStridedBatched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZdscalStridedBatched_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    integer(c_int64_t), value :: stride_x
    integer(c_int64_t), value :: batch_count
            hipblasZdscalStridedBatched_64Fortran = &
        hipblasZdscalStridedBatched_64(handle, n, alpha, x, incx, stride_x, batch_count)
    return
end function hipblasZdscalStridedBatched_64Fortran

! swap
function hipblasSswap_64Fortran(handle, n, x, incx, y, incy) &
    bind(c, name='hipblasSswap_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSswap_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    type(c_ptr), value :: y
    integer(c_int64_t), value :: incy
            hipblasSswap_64Fortran = &
        hipblasSswap_64(handle, n, x, incx, y, incy)
    return
end function hipblasSswap_64Fortran

function hipblasDswap_64Fortran(handle, n, x, incx, y, incy) &
    bind(c, name='hipblasDswap_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDswap_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    type(c_ptr), value :: y
    integer(c_int64_t), value :: incy
            hipblasDswap_64Fortran = &
        hipblasDswap_64(handle, n, x, incx, y, incy)
    return
end function hipblasDswap_64Fortran

function hipblasCswap_64Fortran(handle, n, x, incx, y, incy) &
    bind(c, name='hipblasCswap_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCswap_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    type(c_ptr), value :: y
    integer(c_int64_t), value :: incy
            hipblasCswap_64Fortran = &
        hipblasCswap_64(handle, n, x, incx, y, incy)
    return
end function hipblasCswap_64Fortran

function hipblasZswap_64Fortran(handle, n, x, incx, y, incy) &
    bind(c, name='hipblasZswap_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZswap_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    type(c_ptr), value :: y
    integer(c_int64_t), value :: incy
            hipblasZswap_64Fortran = &
        hipblasZswap_64(handle, n, x, incx, y, incy)
    return
end function hipblasZswap_64Fortran

! swapBatched
function hipblasSswapBatched_64Fortran(handle, n, x, incx, y, incy, batch_count) &
    bind(c, name='hipblasSswapBatched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSswapBatched_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    type(c_ptr), value :: y
    integer(c_int64_t), value :: incy
    integer(c_int64_t), value :: batch_count
            hipblasSswapBatched_64Fortran = &
        hipblasSswapBatched_64(handle, n, x, incx, y, incy, batch_count)
    return
end function hipblasSswapBatched_64Fortran

function hipblasDswapBatched_64Fortran(handle, n, x, incx, y, incy, batch_count) &
    bind(c, name='hipblasDswapBatched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDswapBatched_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    type(c_ptr), value :: y
    integer(c_int64_t), value :: incy
    integer(c_int64_t), value :: batch_count
            hipblasDswapBatched_64Fortran = &
        hipblasDswapBatched_64(handle, n, x, incx, y, incy, batch_count)
    return
end function hipblasDswapBatched_64Fortran

function hipblasCswapBatched_64Fortran(handle, n, x, incx, y, incy, batch_count) &
    bind(c, name='hipblasCswapBatched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCswapBatched_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    type(c_ptr), value :: y
    integer(c_int64_t), value :: incy
    integer(c_int64_t), value :: batch_count
            hipblasCswapBatched_64Fortran = &
        hipblasCswapBatched_64(handle, n, x, incx, y, incy, batch_count)
    return
end function hipblasCswapBatched_64Fortran

function hipblasZswapBatched_64Fortran(handle, n, x, incx, y, incy, batch_count) &
    bind(c, name='hipblasZswapBatched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZswapBatched_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    type(c_ptr), value :: y
    integer(c_int64_t), value :: incy
    integer(c_int64_t), value :: batch_count
            hipblasZswapBatched_64Fortran = &
        hipblasZswapBatched_64(handle, n, x, incx, y, incy, batch_count)
    return
end function hipblasZswapBatched_64Fortran

! swapStridedBatched
function hipblasSswapStridedBatched_64Fortran(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count) &
    bind(c, name='hipblasSswapStridedBatched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSswapStridedBatched_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    integer(c_int64_t), value :: stride_x
    type(c_ptr), value :: y
    integer(c_int64_t), value :: incy
    integer(c_int64_t), value :: stride_y
    integer(c_int64_t), value :: batch_count
            hipblasSswapStridedBatched_64Fortran = &
        hipblasSswapStridedBatched_64(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count)
    return
end function hipblasSswapStridedBatched_64Fortran

function hipblasDswapStridedBatched_64Fortran(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count) &
    bind(c, name='hipblasDswapStridedBatched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDswapStridedBatched_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    integer(c_int64_t), value :: stride_x
    type(c_ptr), value :: y
    integer(c_int64_t), value :: incy
    integer(c_int64_t), value :: stride_y
    integer(c_int64_t), value :: batch_count
            hipblasDswapStridedBatched_64Fortran = &
        hipblasDswapStridedBatched_64(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count)
    return
end function hipblasDswapStridedBatched_64Fortran

function hipblasCswapStridedBatched_64Fortran(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count) &
    bind(c, name='hipblasCswapStridedBatched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCswapStridedBatched_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    integer(c_int64_t), value :: stride_x
    type(c_ptr), value :: y
    integer(c_int64_t), value :: incy
    integer(c_int64_t), value :: stride_y
    integer(c_int64_t), value :: batch_count
            hipblasCswapStridedBatched_64Fortran = &
        hipblasCswapStridedBatched_64(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count)
    return
end function hipblasCswapStridedBatched_64Fortran

function hipblasZswapStridedBatched_64Fortran(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count) &
    bind(c, name='hipblasZswapStridedBatched_64Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZswapStridedBatched_64Fortran
    type(c_ptr), value :: handle
    integer(c_int64_t), value :: n
    type(c_ptr), value :: x
    integer(c_int64_t), value :: incx
    integer(c_int64_t), value :: stride_x
    type(c_ptr), value :: y
    integer(c_int64_t), value :: incy
    integer(c_int64_t), value :: stride_y
    integer(c_int64_t), value :: batch_count
            hipblasZswapStridedBatched_64Fortran = &
        hipblasZswapStridedBatched_64(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count)
    return
end function hipblasZswapStridedBatched_64Fortran
