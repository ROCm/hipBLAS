module hipblas_interface
    use iso_c_binding
    use hipblas

    contains

    !--------!
    ! blas 1 !
    !--------!

    ! scal
    function hipblasSscalFortran(handle, n, alpha, x, incx) &
            result(res) &
            bind(c, name = 'hipblasSscalFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int) :: res
        res = hipblasSscal(handle, n, alpha, x, incx)
        return
    end function hipblasSscalFortran

    function hipblasDscalFortran(handle, n, alpha, x, incx) &
            result(res) &
            bind(c, name = 'hipblasDscalFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int) :: res
        res = hipblasDscal(handle, n, alpha, x, incx)
        return
    end function hipblasDscalFortran

    function hipblasCscalFortran(handle, n, alpha, x, incx) &
            result(res) &
            bind(c, name = 'hipblasCscalFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int) :: res
        res = hipblasCscal(handle, n, alpha, x, incx)
        return
    end function hipblasCscalFortran

    function hipblasZscalFortran(handle, n, alpha, x, incx) &
            result(res) &
            bind(c, name = 'hipblasZscalFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int) :: res
        res = hipblasZscal(handle, n, alpha, x, incx)
        return
    end function hipblasZscalFortran

    function hipblasCsscalFortran(handle, n, alpha, x, incx) &
            result(res) &
            bind(c, name = 'hipblasCsscalFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int) :: res
        res = hipblasCsscal(handle, n, alpha, x, incx)
        return
    end function hipblasCsscalFortran

    function hipblasZdscalFortran(handle, n, alpha, x, incx) &
            result(res) &
            bind(c, name = 'hipblasZdscalFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int) :: res
        res = hipblasZdscal(handle, n, alpha, x, incx)
        return
    end function hipblasZdscalFortran

    ! scalBatched
    function hipblasSscalBatchedFortran(handle, n, alpha, x, incx, batch_count) &
            result(res) &
            bind(c, name = 'hipblasSscalBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasSscalBatched(handle, n, alpha, x, incx, batch_count)
        return
    end function hipblasSscalBatchedFortran

    function hipblasDscalBatchedFortran(handle, n, alpha, x, incx, batch_count) &
            result(res) &
            bind(c, name = 'hipblasDscalBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasDscalBatched(handle, n, alpha, x, incx, batch_count)
        return
    end function hipblasDscalBatchedFortran

    function hipblasCscalBatchedFortran(handle, n, alpha, x, incx, batch_count) &
            result(res) &
            bind(c, name = 'hipblasCscalBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasCscalBatched(handle, n, alpha, x, incx, batch_count)
        return
    end function hipblasCscalBatchedFortran

    function hipblasZscalBatchedFortran(handle, n, alpha, x, incx, batch_count) &
            result(res) &
            bind(c, name = 'hipblasZscalBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasZscalBatched(handle, n, alpha, x, incx, batch_count)
        return
    end function hipblasZscalBatchedFortran

    function hipblasCsscalBatchedFortran(handle, n, alpha, x, incx, batch_count) &
            result(res) &
            bind(c, name = 'hipblasCsscalBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasCsscalBatched(handle, n, alpha, x, incx, batch_count)
        return
    end function hipblasCsscalBatchedFortran

    function hipblasZdscalBatchedFortran(handle, n, alpha, x, incx, batch_count) &
            result(res) &
            bind(c, name = 'hipblasZdscalBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasZdscalBatched(handle, n, alpha, x, incx, batch_count)
        return
    end function hipblasZdscalBatchedFortran

    ! scalStridedBatched
    function hipblasSscalStridedBatchedFortran(handle, n, alpha, x, incx, stride_x, batch_count) &
            result(res) &
            bind(c, name = 'hipblasSscalStridedBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasSscalStridedBatched(handle, n, alpha, x, incx, stride_x, batch_count)
        return
    end function hipblasSscalStridedBatchedFortran

    function hipblasDscalStridedBatchedFortran(handle, n, alpha, x, incx, stride_x, batch_count) &
            result(res) &
            bind(c, name = 'hipblasDscalStridedBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasDscalStridedBatched(handle, n, alpha, x, incx, stride_x, batch_count)
        return
    end function hipblasDscalStridedBatchedFortran

    function hipblasCscalStridedBatchedFortran(handle, n, alpha, x, incx, stride_x, batch_count) &
            result(res) &
            bind(c, name = 'hipblasCscalStridedBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasCscalStridedBatched(handle, n, alpha, x, incx, stride_x, batch_count)
        return
    end function hipblasCscalStridedBatchedFortran

    function hipblasZscalStridedBatchedFortran(handle, n, alpha, x, incx, stride_x, batch_count) &
            result(res) &
            bind(c, name = 'hipblasZscalStridedBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasZscalStridedBatched(handle, n, alpha, x, incx, stride_x, batch_count)
        return
    end function hipblasZscalStridedBatchedFortran

    function hipblasCsscalStridedBatchedFortran(handle, n, alpha, x, incx, stride_x, batch_count) &
            result(res) &
            bind(c, name = 'hipblasCsscalStridedBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasCsscalStridedBatched(handle, n, alpha, x, incx, stride_x, batch_count)
        return
    end function hipblasCsscalStridedBatchedFortran

    function hipblasZdscalStridedBatchedFortran(handle, n, alpha, x, incx, stride_x, batch_count) &
            result(res) &
            bind(c, name = 'hipblasZdscalStridedBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasZdscalStridedBatched(handle, n, alpha, x, incx, stride_x, batch_count)
        return
    end function hipblasZdscalStridedBatchedFortran

    ! copy
    function hipblasScopyFortran(handle, n, x, incx, y, incy) &
            result(res) &
            bind(c, name = 'hipblasScopyFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int) :: res
        res = hipblasScopy(handle, n, x, incx, y, incy)
        return
    end function hipblasScopyFortran

    function hipblasDcopyFortran(handle, n, x, incx, y, incy) &
            result(res) &
            bind(c, name = 'hipblasDcopyFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int) :: res
        res = hipblasDcopy(handle, n, x, incx, y, incy)
        return
    end function hipblasDcopyFortran

    function hipblasCcopyFortran(handle, n, x, incx, y, incy) &
            result(res) &
            bind(c, name = 'hipblasCcopyFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int) :: res
        res = hipblasCcopy(handle, n, x, incx, y, incy)
        return
    end function hipblasCcopyFortran

    function hipblasZcopyFortran(handle, n, x, incx, y, incy) &
            result(res) &
            bind(c, name = 'hipblasZcopyFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int) :: res
        res = hipblasZcopy(handle, n, x, incx, y, incy)
        return
    end function hipblasZcopyFortran

    ! copyBatched
    function hipblasScopyBatchedFortran(handle, n, x, incx, y, incy, batch_count) &
            result(res) &
            bind(c, name = 'hipblasScopyBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasScopyBatched(handle, n, x, incx, y, incy, batch_count)
        return
    end function hipblasScopyBatchedFortran

    function hipblasDcopyBatchedFortran(handle, n, x, incx, y, incy, batch_count) &
            result(res) &
            bind(c, name = 'hipblasDcopyBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasDcopyBatched(handle, n,  x, incx, y, incy, batch_count)
        return
    end function hipblasDcopyBatchedFortran

    function hipblasCcopyBatchedFortran(handle, n, x, incx, y, incy, batch_count) &
            result(res) &
            bind(c, name = 'hipblasCcopyBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasCcopyBatched(handle, n, x, incx, y, incy, batch_count)
        return
    end function hipblasCcopyBatchedFortran

    function hipblasZcopyBatchedFortran(handle, n, x, incx, y, incy, batch_count) &
            result(res) &
            bind(c, name = 'hipblasZcopyBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasZcopyBatched(handle, n, x, incx, y, incy, batch_count)
        return
    end function hipblasZcopyBatchedFortran

    ! copyStridedBatched
    function hipblasScopyStridedBatchedFortran(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count) &
            result(res) &
            bind(c, name = 'hipblasScopyStridedBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int64_t), value :: stride_y
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasScopyStridedBatched(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count)
        return
    end function hipblasScopyStridedBatchedFortran

    function hipblasDcopyStridedBatchedFortran(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count) &
            result(res) &
            bind(c, name = 'hipblasDcopyStridedBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int64_t), value :: stride_y
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasDcopyStridedBatched(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count)
        return
    end function hipblasDcopyStridedBatchedFortran

    function hipblasCcopyStridedBatchedFortran(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count) &
            result(res) &
            bind(c, name = 'hipblasCcopyStridedBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int64_t), value :: stride_y
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasCcopyStridedBatched(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count)
        return
    end function hipblasCcopyStridedBatchedFortran

    function hipblasZcopyStridedBatchedFortran(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count) &
            result(res) &
            bind(c, name = 'hipblasZcopyStridedBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int64_t), value :: stride_y
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasZcopyStridedBatched(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count)
        return
    end function hipblasZcopyStridedBatchedFortran

    ! dot
    function hipblasSdotFortran(handle, n, x, incx, y, incy, result) &
            result(res) &
            bind(c, name = 'hipblasSdotFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = hipblasSdot(handle, n, x, incx, y, incy, result)
        return
    end function hipblasSdotFortran

    function hipblasDdotFortran(handle, n, x, incx, y, incy, result) &
            result(res) &
            bind(c, name = 'hipblasDdotFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = hipblasDdot(handle, n, x, incx, y, incy, result)
        return
    end function hipblasDdotFortran

    function hipblasHdotFortran(handle, n, x, incx, y, incy, result) &
            result(res) &
            bind(c, name = 'hipblasHdotFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = hipblasHdot(handle, n, x, incx, y, incy, result)
        return
    end function hipblasHdotFortran

    function hipblasBfdotFortran(handle, n, x, incx, y, incy, result) &
            result(res) &
            bind(c, name = 'hipblasBfdotFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = hipblasBfdot(handle, n, x, incx, y, incy, result)
        return
    end function hipblasBfdotFortran

    function hipblasCdotuFortran(handle, n, x, incx, y, incy, result) &
            result(res) &
            bind(c, name = 'hipblasCdotuFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = hipblasCdotu(handle, n, x, incx, y, incy, result)
        return
    end function hipblasCdotuFortran

    function hipblasCdotcFortran(handle, n, x, incx, y, incy, result) &
            result(res) &
            bind(c, name = 'hipblasCdotcFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = hipblasCdotc(handle, n, x, incx, y, incy, result)
        return
    end function hipblasCdotcFortran

    function hipblasZdotuFortran(handle, n, x, incx, y, incy, result) &
            result(res) &
            bind(c, name = 'hipblasZdotuFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = hipblasZdotu(handle, n, x, incx, y, incy, result)
        return
    end function hipblasZdotuFortran

    function hipblasZdotcFortran(handle, n, x, incx, y, incy, result) &
            result(res) &
            bind(c, name = 'hipblasZdotcFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = hipblasZdotc(handle, n, x, incx, y, incy, result)
        return
    end function hipblasZdotcFortran

    ! dotBatched
    function hipblasSdotBatchedFortran(handle, n, x, incx, y, incy, batch_count, result) &
            result(res) &
            bind(c, name = 'hipblasSdotBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int), value :: batch_count
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = hipblasSdotBatched(handle, n, x, incx, y, incy, batch_count, result)
        return
    end function hipblasSdotBatchedFortran

    function hipblasDdotBatchedFortran(handle, n, x, incx, y, incy, batch_count, result) &
            result(res) &
            bind(c, name = 'hipblasDdotBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int), value :: batch_count
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = hipblasDdotBatched(handle, n, x, incx, y, incy, batch_count, result)
        return
    end function hipblasDdotBatchedFortran

    function hipblasHdotBatchedFortran(handle, n, x, incx, y, incy, batch_count, result) &
            result(res) &
            bind(c, name = 'hipblasHdotBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int), value :: batch_count
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = hipblasHdotBatched(handle, n, x, incx, y, incy, batch_count, result)
        return
    end function hipblasHdotBatchedFortran

    function hipblasBfdotBatchedFortran(handle, n, x, incx, y, incy, batch_count, result) &
            result(res) &
            bind(c, name = 'hipblasBfdotBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int), value :: batch_count
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = hipblasBfdotBatched(handle, n, x, incx, y, incy, batch_count, result)
        return
    end function hipblasBfdotBatchedFortran

    function hipblasCdotuBatchedFortran(handle, n, x, incx, y, incy, batch_count, result) &
            result(res) &
            bind(c, name = 'hipblasCdotuBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int), value :: batch_count
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = hipblasCdotuBatched(handle, n, x, incx, y, incy, batch_count, result)
        return
    end function hipblasCdotuBatchedFortran

    function hipblasCdotcBatchedFortran(handle, n, x, incx, y, incy, batch_count, result) &
            result(res) &
            bind(c, name = 'hipblasCdotcBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int), value :: batch_count
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = hipblasCdotcBatched(handle, n, x, incx, y, incy, batch_count, result)
        return
    end function hipblasCdotcBatchedFortran

    function hipblasZdotuBatchedFortran(handle, n, x, incx, y, incy, batch_count, result) &
            result(res) &
            bind(c, name = 'hipblasZdotuBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int), value :: batch_count
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = hipblasZdotuBatched(handle, n, x, incx, y, incy, batch_count, result)
        return
    end function hipblasZdotuBatchedFortran

    function hipblasZdotcBatchedFortran(handle, n, x, incx, y, incy, batch_count, result) &
            result(res) &
            bind(c, name = 'hipblasZdotcBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int), value :: batch_count
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = hipblasZdotcBatched(handle, n, x, incx, y, incy, batch_count, result)
        return
    end function hipblasZdotcBatchedFortran

    ! dotStridedBatched
    function hipblasSdotStridedBatchedFortran(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count, result) &
            result(res) &
            bind(c, name = 'hipblasSdotStridedBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int64_t), value :: stride_y
        integer(c_int), value :: batch_count
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = hipblasSdotStridedBatched(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count, result)
        return
    end function hipblasSdotStridedBatchedFortran

    function hipblasDdotStridedBatchedFortran(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count, result) &
            result(res) &
            bind(c, name = 'hipblasDdotStridedBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int64_t), value :: stride_y
        integer(c_int), value :: batch_count
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = hipblasDdotStridedBatched(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count, result)
        return
    end function hipblasDdotStridedBatchedFortran

    function hipblasHdotStridedBatchedFortran(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count, result) &
            result(res) &
            bind(c, name = 'hipblasHdotStridedBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int64_t), value :: stride_y
        integer(c_int), value :: batch_count
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = hipblasHdotStridedBatched(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count, result)
        return
    end function hipblasHdotStridedBatchedFortran

    function hipblasBfdotStridedBatchedFortran(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count, result) &
            result(res) &
            bind(c, name = 'hipblasBfdotStridedBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int64_t), value :: stride_y
        integer(c_int), value :: batch_count
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = hipblasBfdotStridedBatched(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count, result)
        return
    end function hipblasBfdotStridedBatchedFortran

    function hipblasCdotuStridedBatchedFortran(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count, result) &
            result(res) &
            bind(c, name = 'hipblasCdotuStridedBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int64_t), value :: stride_y
        integer(c_int), value :: batch_count
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = hipblasCdotuStridedBatched(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count, result)
        return
    end function hipblasCdotuStridedBatchedFortran

    function hipblasCdotcStridedBatchedFortran(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count, result) &
            result(res) &
            bind(c, name = 'hipblasCdotcStridedBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int64_t), value :: stride_y
        integer(c_int), value :: batch_count
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = hipblasCdotcStridedBatched(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count, result)
        return
    end function hipblasCdotcStridedBatchedFortran

    function hipblasZdotuStridedBatchedFortran(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count, result) &
            result(res) &
            bind(c, name = 'hipblasZdotuStridedBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int64_t), value :: stride_y
        integer(c_int), value :: batch_count
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = hipblasZdotuStridedBatched(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count, result)
        return
    end function hipblasZdotuStridedBatchedFortran

    function hipblasZdotcStridedBatchedFortran(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count, result) &
            result(res) &
            bind(c, name = 'hipblasZdotcStridedBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int64_t), value :: stride_y
        integer(c_int), value :: batch_count
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = hipblasZdotcStridedBatched(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count, result)
        return
    end function hipblasZdotcStridedBatchedFortran

    ! swap
    function hipblasSswapFortran(handle, n, x, incx, y, incy) &
            result(res) &
            bind(c, name = 'hipblasSswapFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int) :: res
        res = hipblasSswap(handle, n, x, incx, y, incy)
        return
    end function hipblasSswapFortran

    function hipblasDswapFortran(handle, n, x, incx, y, incy) &
            result(res) &
            bind(c, name = 'hipblasDswapFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int) :: res
        res = hipblasDswap(handle, n, x, incx, y, incy)
        return
    end function hipblasDswapFortran

    function hipblasCswapFortran(handle, n, x, incx, y, incy) &
            result(res) &
            bind(c, name = 'hipblasCswapFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int) :: res
        res = hipblasCswap(handle, n, x, incx, y, incy)
        return
    end function hipblasCswapFortran

    function hipblasZswapFortran(handle, n, x, incx, y, incy) &
            result(res) &
            bind(c, name = 'hipblasZswapFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int) :: res
        res = hipblasZswap(handle, n, x, incx, y, incy)
        return
    end function hipblasZswapFortran

    ! swapBatched
    function hipblasSswapBatchedFortran(handle, n, x, incx, y, incy, batch_count) &
            result(res) &
            bind(c, name = 'hipblasSswapBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasSswapBatched(handle, n, x, incx, y, incy, batch_count)
        return
    end function hipblasSswapBatchedFortran

    function hipblasDswapBatchedFortran(handle, n, x, incx, y, incy, batch_count) &
            result(res) &
            bind(c, name = 'hipblasDswapBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasDswapBatched(handle, n, x, incx, y, incy, batch_count)
        return
    end function hipblasDswapBatchedFortran

    function hipblasCswapBatchedFortran(handle, n, x, incx, y, incy, batch_count) &
            result(res) &
            bind(c, name = 'hipblasCswapBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasCswapBatched(handle, n, x, incx, y, incy, batch_count)
        return
    end function hipblasCswapBatchedFortran

    function hipblasZswapBatchedFortran(handle, n, x, incx, y, incy, batch_count) &
            result(res) &
            bind(c, name = 'hipblasZswapBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasZswapBatched(handle, n, x, incx, y, incy, batch_count)
        return
    end function hipblasZswapBatchedFortran

    ! swapStridedBatched
    function hipblasSswapStridedBatchedFortran(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count) &
            result(res) &
            bind(c, name = 'hipblasSswapStridedBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int64_t), value :: stride_y
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasSswapStridedBatched(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count)
        return
    end function hipblasSswapStridedBatchedFortran

    function hipblasDswapStridedBatchedFortran(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count) &
            result(res) &
            bind(c, name = 'hipblasDswapStridedBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int64_t), value :: stride_y
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasDswapStridedBatched(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count)
        return
    end function hipblasDswapStridedBatchedFortran

    function hipblasCswapStridedBatchedFortran(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count) &
            result(res) &
            bind(c, name = 'hipblasCswapStridedBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int64_t), value :: stride_y
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasCswapStridedBatched(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count)
        return
    end function hipblasCswapStridedBatchedFortran

    function hipblasZswapStridedBatchedFortran(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count) &
            result(res) &
            bind(c, name = 'hipblasZswapStridedBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int64_t), value :: stride_y
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasZswapStridedBatched(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count)
        return
    end function hipblasZswapStridedBatchedFortran

    ! axpy
    function hipblasHaxpyFortran(handle, n, alpha, x, incx, y, incy) &
            result(res) &
            bind(c, name = 'hipblasHaxpyFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int) :: res
        res = hipblasHaxpy(handle, n, alpha, x, incx, y, incy)
        return
    end function hipblasHaxpyFortran

    function hipblasSaxpyFortran(handle, n, alpha, x, incx, y, incy) &
            result(res) &
            bind(c, name = 'hipblasSaxpyFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int) :: res
        res = hipblasSaxpy(handle, n, alpha, x, incx, y, incy)
        return
    end function hipblasSaxpyFortran

    function hipblasDaxpyFortran(handle, n, alpha, x, incx, y, incy) &
            result(res) &
            bind(c, name = 'hipblasDaxpyFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int) :: res
        res = hipblasDaxpy(handle, n, alpha, x, incx, y, incy)
        return
    end function hipblasDaxpyFortran

    function hipblasCaxpyFortran(handle, n, alpha, x, incx, y, incy) &
            result(res) &
            bind(c, name = 'hipblasCaxpyFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int) :: res
        res = hipblasCaxpy(handle, n, alpha, x, incx, y, incy)
        return
    end function hipblasCaxpyFortran

    function hipblasZaxpyFortran(handle, n, alpha, x, incx, y, incy) &
            result(res) &
            bind(c, name = 'hipblasZaxpyFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int) :: res
        res = hipblasZaxpy(handle, n, alpha, x, incx, y, incy)
        return
    end function hipblasZaxpyFortran

    ! axpyBatched
    function hipblasHaxpyBatchedFortran(handle, n, alpha, x, incx, y, incy, batch_count) &
            result(res) &
            bind(c, name = 'hipblasHaxpyBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasHaxpyBatched(handle, n, alpha, x, incx, y, incy, batch_count)
        return
    end function hipblasHaxpyBatchedFortran

    function hipblasSaxpyBatchedFortran(handle, n, alpha, x, incx, y, incy, batch_count) &
            result(res) &
            bind(c, name = 'hipblasSaxpyBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasSaxpyBatched(handle, n, alpha, x, incx, y, incy, batch_count)
        return
    end function hipblasSaxpyBatchedFortran

    function hipblasDaxpyBatchedFortran(handle, n, alpha, x, incx, y, incy, batch_count) &
            result(res) &
            bind(c, name = 'hipblasDaxpyBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasDaxpyBatched(handle, n, alpha, x, incx, y, incy, batch_count)
        return
    end function hipblasDaxpyBatchedFortran

    function hipblasCaxpyBatchedFortran(handle, n, alpha, x, incx, y, incy, batch_count) &
            result(res) &
            bind(c, name = 'hipblasCaxpyBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasCaxpyBatched(handle, n, alpha, x, incx, y, incy, batch_count)
        return
    end function hipblasCaxpyBatchedFortran

    function hipblasZaxpyBatchedFortran(handle, n, alpha, x, incx, y, incy, batch_count) &
            result(res) &
            bind(c, name = 'hipblasZaxpyBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasZaxpyBatched(handle, n, alpha, x, incx, y, incy, batch_count)
        return
    end function hipblasZaxpyBatchedFortran

    ! axpyStridedBatched
    function hipblasHaxpyStridedBatchedFortran(handle, n, alpha, x, incx, stride_x, y, incy, stride_y, batch_count) &
            result(res) &
            bind(c, name = 'hipblasHaxpyStridedBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int64_t), value :: stride_y
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasHaxpyStridedBatched(handle, n, alpha, x, incx, stride_x, y, incy, stride_y, batch_count)
        return
    end function hipblasHaxpyStridedBatchedFortran

    function hipblasSaxpyStridedBatchedFortran(handle, n, alpha, x, incx, stride_x, y, incy, stride_y, batch_count) &
            result(res) &
            bind(c, name = 'hipblasSaxpyStridedBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int64_t), value :: stride_y
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasSaxpyStridedBatched(handle, n, alpha, x, incx, stride_x, y, incy, stride_y, batch_count)
        return
    end function hipblasSaxpyStridedBatchedFortran

    function hipblasDaxpyStridedBatchedFortran(handle, n, alpha, x, incx, stride_x, y, incy, stride_y, batch_count) &
            result(res) &
            bind(c, name = 'hipblasDaxpyStridedBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int64_t), value :: stride_y
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasDaxpyStridedBatched(handle, n, alpha, x, incx, stride_x, y, incy, stride_y, batch_count)
        return
    end function hipblasDaxpyStridedBatchedFortran

    function hipblasCaxpyStridedBatchedFortran(handle, n, alpha, x, incx, stride_x, y, incy, stride_y, batch_count) &
            result(res) &
            bind(c, name = 'hipblasCaxpyStridedBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int64_t), value :: stride_y
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasCaxpyStridedBatched(handle, n, alpha, x, incx, stride_x, y, incy, stride_y, batch_count)
        return
    end function hipblasCaxpyStridedBatchedFortran

    function hipblasZaxpyStridedBatchedFortran(handle, n, alpha, x, incx, stride_x, y, incy, stride_y, batch_count) &
            result(res) &
            bind(c, name = 'hipblasZaxpyStridedBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int64_t), value :: stride_y
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasZaxpyStridedBatched(handle, n, alpha, x, incx, stride_x, y, incy, stride_y, batch_count)
        return
    end function hipblasZaxpyStridedBatchedFortran

    ! asum
    function hipblasSasumFortran(handle, n, x, incx, result) &
            result(res) &
            bind(c, name = 'hipblasSasumFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = hipblasSasum(handle, n, x, incx, result)
        return
    end function hipblasSasumFortran

    function hipblasDasumFortran(handle, n, x, incx, result) &
            result(res) &
            bind(c, name = 'hipblasDasumFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = hipblasDasum(handle, n, x, incx, result)
        return
    end function hipblasDasumFortran

    function hipblasScasumFortran(handle, n, x, incx, result) &
            result(res) &
            bind(c, name = 'hipblasScasumFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = hipblasScasum(handle, n, x, incx, result)
        return
    end function hipblasScasumFortran

    function hipblasDzasumFortran(handle, n, x, incx, result) &
            result(res) &
            bind(c, name = 'hipblasDzasumFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = hipblasDzasum(handle, n, x, incx, result)
        return
    end function hipblasDzasumFortran

    ! asumBatched
    function hipblasSasumBatchedFortran(handle, n, x, incx, batch_count, result) &
            result(res) &
            bind(c, name = 'hipblasSasumBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int), value :: batch_count
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = hipblasSasumBatched(handle, n, x, incx, batch_count, result)
        return
    end function hipblasSasumBatchedFortran

    function hipblasDasumBatchedFortran(handle, n, x, incx, batch_count, result) &
            result(res) &
            bind(c, name = 'hipblasDasumBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int), value :: batch_count
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = hipblasDasumBatched(handle, n, x, incx, batch_count, result)
        return
    end function hipblasDasumBatchedFortran

    function hipblasScasumBatchedFortran(handle, n, x, incx, batch_count, result) &
            result(res) &
            bind(c, name = 'hipblasScasumBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int), value :: batch_count
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = hipblasScasumBatched(handle, n, x, incx, batch_count, result)
        return
    end function hipblasScasumBatchedFortran

    function hipblasDzasumBatchedFortran(handle, n, x, incx, batch_count, result) &
            result(res) &
            bind(c, name = 'hipblasDzasumBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int), value :: batch_count
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = hipblasDzasumBatched(handle, n, x, incx, batch_count, result)
        return
    end function hipblasDzasumBatchedFortran

    ! asumStridedBatched
    function hipblasSasumStridedBatchedFortran(handle, n, x, incx, stride_x, batch_count, result) &
            result(res) &
            bind(c, name = 'hipblasSasumStridedBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        integer(c_int), value :: batch_count
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = hipblasSasumStridedBatched(handle, n, x, incx, stride_x, batch_count, result)
        return
    end function hipblasSasumStridedBatchedFortran

    function hipblasDasumStridedBatchedFortran(handle, n, x, incx, stride_x, batch_count, result) &
            result(res) &
            bind(c, name = 'hipblasDasumStridedBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        integer(c_int), value :: batch_count
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = hipblasDasumStridedBatched(handle, n, x, incx, stride_x, batch_count, result)
        return
    end function hipblasDasumStridedBatchedFortran

    function hipblasScasumStridedBatchedFortran(handle, n, x, incx, stride_x, batch_count, result) &
            result(res) &
            bind(c, name = 'hipblasScasumStridedBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        integer(c_int), value :: batch_count
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = hipblasScasumStridedBatched(handle, n, x, incx, stride_x, batch_count, result)
        return
    end function hipblasScasumStridedBatchedFortran

    function hipblasDzasumStridedBatchedFortran(handle, n, x, incx, stride_x, batch_count, result) &
            result(res) &
            bind(c, name = 'hipblasDzasumStridedBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        integer(c_int), value :: batch_count
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = hipblasDzasumStridedBatched(handle, n, x, incx, stride_x, batch_count, result)
        return
    end function hipblasDzasumStridedBatchedFortran

    ! nrm2
    function hipblasSnrm2Fortran(handle, n, x, incx, result) &
            result(res) &
            bind(c, name = 'hipblasSnrm2Fortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = hipblasSnrm2(handle, n, x, incx, result)
        return
    end function hipblasSnrm2Fortran

    function hipblasDnrm2Fortran(handle, n, x, incx, result) &
            result(res) &
            bind(c, name = 'hipblasDnrm2Fortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = hipblasDnrm2(handle, n, x, incx, result)
        return
    end function hipblasDnrm2Fortran

    function hipblasScnrm2Fortran(handle, n, x, incx, result) &
            result(res) &
            bind(c, name = 'hipblasScnrm2Fortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = hipblasScnrm2(handle, n, x, incx, result)
        return
    end function hipblasScnrm2Fortran

    function hipblasDznrm2Fortran(handle, n, x, incx, result) &
            result(res) &
            bind(c, name = 'hipblasDznrm2Fortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = hipblasDznrm2(handle, n, x, incx, result)
        return
    end function hipblasDznrm2Fortran

    ! nrm2Batched
    function hipblasSnrm2BatchedFortran(handle, n, x, incx, batch_count, result) &
            result(res) &
            bind(c, name = 'hipblasSnrm2BatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int), value :: batch_count
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = hipblasSnrm2Batched(handle, n, x, incx, batch_count, result)
        return
    end function hipblasSnrm2BatchedFortran

    function hipblasDnrm2BatchedFortran(handle, n, x, incx, batch_count, result) &
            result(res) &
            bind(c, name = 'hipblasDnrm2BatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int), value :: batch_count
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = hipblasDnrm2Batched(handle, n, x, incx, batch_count, result)
        return
    end function hipblasDnrm2BatchedFortran

    function hipblasScnrm2BatchedFortran(handle, n, x, incx, batch_count, result) &
            result(res) &
            bind(c, name = 'hipblasScnrm2BatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int), value :: batch_count
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = hipblasScnrm2Batched(handle, n, x, incx, batch_count, result)
        return
    end function hipblasScnrm2BatchedFortran

    function hipblasDznrm2BatchedFortran(handle, n, x, incx, batch_count, result) &
            result(res) &
            bind(c, name = 'hipblasDznrm2BatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int), value :: batch_count
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = hipblasDznrm2Batched(handle, n, x, incx, batch_count, result)
        return
    end function hipblasDznrm2BatchedFortran

    ! nrm2StridedBatched
    function hipblasSnrm2StridedBatchedFortran(handle, n, x, incx, stride_x, batch_count, result) &
            result(res) &
            bind(c, name = 'hipblasSnrm2StridedBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        integer(c_int), value :: batch_count
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = hipblasSnrm2StridedBatched(handle, n, x, incx, stride_x, batch_count, result)
        return
    end function hipblasSnrm2StridedBatchedFortran

    function hipblasDnrm2StridedBatchedFortran(handle, n, x, incx, stride_x, batch_count, result) &
            result(res) &
            bind(c, name = 'hipblasDnrm2StridedBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        integer(c_int), value :: batch_count
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = hipblasDnrm2StridedBatched(handle, n, x, incx, stride_x, batch_count, result)
        return
    end function hipblasDnrm2StridedBatchedFortran

    function hipblasScnrm2StridedBatchedFortran(handle, n, x, incx, stride_x, batch_count, result) &
            result(res) &
            bind(c, name = 'hipblasScnrm2StridedBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        integer(c_int), value :: batch_count
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = hipblasScnrm2StridedBatched(handle, n, x, incx, stride_x, batch_count, result)
        return
    end function hipblasScnrm2StridedBatchedFortran

    function hipblasDznrm2StridedBatchedFortran(handle, n, x, incx, stride_x, batch_count, result) &
            result(res) &
            bind(c, name = 'hipblasDznrm2StridedBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        integer(c_int), value :: batch_count
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = hipblasDznrm2StridedBatched(handle, n, x, incx, stride_x, batch_count, result)
        return
    end function hipblasDznrm2StridedBatchedFortran

    ! amax
    function hipblasIsamaxFortran(handle, n, x, incx, result) &
            result(res) &
            bind(c, name = 'hipblasIsamaxFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = hipblasIsamax(handle, n, x, incx, result)
        return
    end function hipblasIsamaxFortran

    function hipblasIdamaxFortran(handle, n, x, incx, result) &
            result(res) &
            bind(c, name = 'hipblasIdamaxFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = hipblasIdamax(handle, n, x, incx, result)
        return
    end function hipblasIdamaxFortran

    function hipblasIcamaxFortran(handle, n, x, incx, result) &
            result(res) &
            bind(c, name = 'hipblasIcamaxFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = hipblasIcamax(handle, n, x, incx, result)
        return
    end function hipblasIcamaxFortran

    function hipblasIzamaxFortran(handle, n, x, incx, result) &
            result(res) &
            bind(c, name = 'hipblasIzamaxFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = hipblasIzamax(handle, n, x, incx, result)
        return
    end function hipblasIzamaxFortran

    ! amaxBatched
    function hipblasIsamaxBatchedFortran(handle, n, x, incx, batch_count, result) &
            result(res) &
            bind(c, name = 'hipblasIsamaxBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int), value :: batch_count
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = hipblasIsamaxBatched(handle, n, x, incx, batch_count, result)
        return
    end function hipblasIsamaxBatchedFortran

    function hipblasIdamaxBatchedFortran(handle, n, x, incx, batch_count, result) &
            result(res) &
            bind(c, name = 'hipblasIdamaxBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int), value :: batch_count
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = hipblasIdamaxBatched(handle, n, x, incx, batch_count, result)
        return
    end function hipblasIdamaxBatchedFortran

    function hipblasIcamaxBatchedFortran(handle, n, x, incx, batch_count, result) &
            result(res) &
            bind(c, name = 'hipblasIcamaxBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int), value :: batch_count
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = hipblasIcamaxBatched(handle, n, x, incx, batch_count, result)
        return
    end function hipblasIcamaxBatchedFortran

    function hipblasIzamaxBatchedFortran(handle, n, x, incx, batch_count, result) &
            result(res) &
            bind(c, name = 'hipblasIzamaxBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int), value :: batch_count
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = hipblasIzamaxBatched(handle, n, x, incx, batch_count, result)
        return
    end function hipblasIzamaxBatchedFortran

    ! amaxStridedBatched
    function hipblasIsamaxStridedBatchedFortran(handle, n, x, incx, stride_x, batch_count, result) &
            result(res) &
            bind(c, name = 'hipblasIsamaxStridedBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        integer(c_int), value :: batch_count
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = hipblasIsamaxStridedBatched(handle, n, x, incx, stride_x, batch_count, result)
        return
    end function hipblasIsamaxStridedBatchedFortran

    function hipblasIdamaxStridedBatchedFortran(handle, n, x, incx, stride_x, batch_count, result) &
            result(res) &
            bind(c, name = 'hipblasIdamaxStridedBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        integer(c_int), value :: batch_count
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = hipblasIdamaxStridedBatched(handle, n, x, incx, stride_x, batch_count, result)
        return
    end function hipblasIdamaxStridedBatchedFortran

    function hipblasIcamaxStridedBatchedFortran(handle, n, x, incx, stride_x, batch_count, result) &
            result(res) &
            bind(c, name = 'hipblasIcamaxStridedBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        integer(c_int), value :: batch_count
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = hipblasIcamaxStridedBatched(handle, n, x, incx, stride_x, batch_count, result)
        return
    end function hipblasIcamaxStridedBatchedFortran

    function hipblasIzamaxStridedBatchedFortran(handle, n, x, incx, stride_x, batch_count, result) &
            result(res) &
            bind(c, name = 'hipblasIzamaxStridedBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        integer(c_int), value :: batch_count
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = hipblasIzamaxStridedBatched(handle, n, x, incx, stride_x, batch_count, result)
        return
    end function hipblasIzamaxStridedBatchedFortran

    ! amin
    function hipblasIsaminFortran(handle, n, x, incx, result) &
            result(res) &
            bind(c, name = 'hipblasIsaminFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = hipblasIsamin(handle, n, x, incx, result)
        return
    end function hipblasIsaminFortran

    function hipblasIdaminFortran(handle, n, x, incx, result) &
            result(res) &
            bind(c, name = 'hipblasIdaminFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = hipblasIdamin(handle, n, x, incx, result)
        return
    end function hipblasIdaminFortran

    function hipblasIcaminFortran(handle, n, x, incx, result) &
            result(res) &
            bind(c, name = 'hipblasIcaminFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = hipblasIcamin(handle, n, x, incx, result)
        return
    end function hipblasIcaminFortran

    function hipblasIzaminFortran(handle, n, x, incx, result) &
            result(res) &
            bind(c, name = 'hipblasIzaminFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = hipblasIzamin(handle, n, x, incx, result)
        return
    end function hipblasIzaminFortran

    ! aminBatched
    function hipblasIsaminBatchedFortran(handle, n, x, incx, batch_count, result) &
            result(res) &
            bind(c, name = 'hipblasIsaminBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int), value :: batch_count
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = hipblasIsaminBatched(handle, n, x, incx, batch_count, result)
        return
    end function hipblasIsaminBatchedFortran

    function hipblasIdaminBatchedFortran(handle, n, x, incx, batch_count, result) &
            result(res) &
            bind(c, name = 'hipblasIdaminBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int), value :: batch_count
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = hipblasIdaminBatched(handle, n, x, incx, batch_count, result)
        return
    end function hipblasIdaminBatchedFortran

    function hipblasIcaminBatchedFortran(handle, n, x, incx, batch_count, result) &
            result(res) &
            bind(c, name = 'hipblasIcaminBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int), value :: batch_count
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = hipblasIcaminBatched(handle, n, x, incx, batch_count, result)
        return
    end function hipblasIcaminBatchedFortran

    function hipblasIzaminBatchedFortran(handle, n, x, incx, batch_count, result) &
            result(res) &
            bind(c, name = 'hipblasIzaminBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int), value :: batch_count
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = hipblasIzaminBatched(handle, n, x, incx, batch_count, result)
        return
    end function hipblasIzaminBatchedFortran

    ! aminStridedBatched
    function hipblasIsaminStridedBatchedFortran(handle, n, x, incx, stride_x, batch_count, result) &
            result(res) &
            bind(c, name = 'hipblasIsaminStridedBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        integer(c_int), value :: batch_count
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = hipblasIsaminStridedBatched(handle, n, x, incx, stride_x, batch_count, result)
        return
    end function hipblasIsaminStridedBatchedFortran

    function hipblasIdaminStridedBatchedFortran(handle, n, x, incx, stride_x, batch_count, result) &
            result(res) &
            bind(c, name = 'hipblasIdaminStridedBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        integer(c_int), value :: batch_count
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = hipblasIdaminStridedBatched(handle, n, x, incx, stride_x, batch_count, result)
        return
    end function hipblasIdaminStridedBatchedFortran

    function hipblasIcaminStridedBatchedFortran(handle, n, x, incx, stride_x, batch_count, result) &
            result(res) &
            bind(c, name = 'hipblasIcaminStridedBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        integer(c_int), value :: batch_count
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = hipblasIcaminStridedBatched(handle, n, x, incx, stride_x, batch_count, result)
        return
    end function hipblasIcaminStridedBatchedFortran

    function hipblasIzaminStridedBatchedFortran(handle, n, x, incx, stride_x, batch_count, result) &
            result(res) &
            bind(c, name = 'hipblasIzaminStridedBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        integer(c_int), value :: batch_count
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = hipblasIzaminStridedBatched(handle, n, x, incx, stride_x, batch_count, result)
        return
    end function hipblasIzaminStridedBatchedFortran

    ! rot
    function hipblasSrotFortran(handle, n, x, incx, y, incy, c, s) &
            result(res) &
            bind(c, name = 'hipblasSrotFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        type(c_ptr), value :: c
        type(c_ptr), value :: s
        integer(c_int) :: res
        res = hipblasSrot(handle, n, x, incx, y, incy, c, s)
        return
    end function hipblasSrotFortran

    function hipblasDrotFortran(handle, n, x, incx, y, incy, c, s) &
            result(res) &
            bind(c, name = 'hipblasDrotFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        type(c_ptr), value :: c
        type(c_ptr), value :: s
        integer(c_int) :: res
        res = hipblasDrot(handle, n, x, incx, y, incy, c, s)
        return
    end function hipblasDrotFortran

    function hipblasCrotFortran(handle, n, x, incx, y, incy, c, s) &
            result(res) &
            bind(c, name = 'hipblasCrotFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        type(c_ptr), value :: c
        type(c_ptr), value :: s
        integer(c_int) :: res
        res = hipblasCrot(handle, n, x, incx, y, incy, c, s)
        return
    end function hipblasCrotFortran

    function hipblasCsrotFortran(handle, n, x, incx, y, incy, c, s) &
            result(res) &
            bind(c, name = 'hipblasCsrotFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        type(c_ptr), value :: c
        type(c_ptr), value :: s
        integer(c_int) :: res
        res = hipblasCsrot(handle, n, x, incx, y, incy, c, s)
        return
    end function hipblasCsrotFortran

    function hipblasZrotFortran(handle, n, x, incx, y, incy, c, s) &
            result(res) &
            bind(c, name = 'hipblasZrotFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        type(c_ptr), value :: c
        type(c_ptr), value :: s
        integer(c_int) :: res
        res = hipblasZrot(handle, n, x, incx, y, incy, c, s)
        return
    end function hipblasZrotFortran

    function hipblasZdrotFortran(handle, n, x, incx, y, incy, c, s) &
            result(res) &
            bind(c, name = 'hipblasZdrotFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        type(c_ptr), value :: c
        type(c_ptr), value :: s
        integer(c_int) :: res
        res = hipblasZdrot(handle, n, x, incx, y, incy, c, s)
        return
    end function hipblasZdrotFortran

    ! rotBatched
    function hipblasSrotBatchedFortran(handle, n, x, incx, y, incy, c, s, batch_count) &
            result(res) &
            bind(c, name = 'hipblasSrotBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        type(c_ptr), value :: c
        type(c_ptr), value :: s
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasSrotBatched(handle, n, x, incx, y, incy, c, s, batch_count)
        return
    end function hipblasSrotBatchedFortran

    function hipblasDrotBatchedFortran(handle, n, x, incx, y, incy, c, s, batch_count) &
            result(res) &
            bind(c, name = 'hipblasDrotBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        type(c_ptr), value :: c
        type(c_ptr), value :: s
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasDrotBatched(handle, n, x, incx, y, incy, c, s, batch_count)
        return
    end function hipblasDrotBatchedFortran

    function hipblasCrotBatchedFortran(handle, n, x, incx, y, incy, c, s, batch_count) &
            result(res) &
            bind(c, name = 'hipblasCrotBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        type(c_ptr), value :: c
        type(c_ptr), value :: s
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasCrotBatched(handle, n, x, incx, y, incy, c, s, batch_count)
        return
    end function hipblasCrotBatchedFortran

    function hipblasCsrotBatchedFortran(handle, n, x, incx, y, incy, c, s, batch_count) &
            result(res) &
            bind(c, name = 'hipblasCsrotBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        type(c_ptr), value :: c
        type(c_ptr), value :: s
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasCsrotBatched(handle, n, x, incx, y, incy, c, s, batch_count)
        return
    end function hipblasCsrotBatchedFortran

    function hipblasZrotBatchedFortran(handle, n, x, incx, y, incy, c, s, batch_count) &
            result(res) &
            bind(c, name = 'hipblasZrotBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        type(c_ptr), value :: c
        type(c_ptr), value :: s
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasZrotBatched(handle, n, x, incx, y, incy, c, s, batch_count)
        return
    end function hipblasZrotBatchedFortran

    function hipblasZdrotBatchedFortran(handle, n, x, incx, y, incy, c, s, batch_count) &
            result(res) &
            bind(c, name = 'hipblasZdrotBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        type(c_ptr), value :: c
        type(c_ptr), value :: s
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasZdrotBatched(handle, n, x, incx, y, incy, c, s, batch_count)
        return
    end function hipblasZdrotBatchedFortran

    ! rotStridedBatched
    function hipblasSrotStridedBatchedFortran(handle, n, x, incx, stride_x, y, incy, stride_y, c, s, batch_count) &
            result(res) &
            bind(c, name = 'hipblasSrotStridedBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int64_t), value :: stride_y
        type(c_ptr), value :: c
        type(c_ptr), value :: s
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasSrotStridedBatched(handle, n, x, incx, stride_x, y, incy, stride_y, c, s, batch_count)
        return
    end function hipblasSrotStridedBatchedFortran

    function hipblasDrotStridedBatchedFortran(handle, n, x, incx, stride_x, y, incy, stride_y, c, s, batch_count) &
            result(res) &
            bind(c, name = 'hipblasDrotStridedBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int64_t), value :: stride_y
        type(c_ptr), value :: c
        type(c_ptr), value :: s
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasDrotStridedBatched(handle, n, x, incx, stride_x, y, incy, stride_y, c, s, batch_count)
        return
    end function hipblasDrotStridedBatchedFortran

    function hipblasCrotStridedBatchedFortran(handle, n, x, incx, stride_x, y, incy, stride_y, c, s, batch_count) &
            result(res) &
            bind(c, name = 'hipblasCrotStridedBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int64_t), value :: stride_y
        type(c_ptr), value :: c
        type(c_ptr), value :: s
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasCrotStridedBatched(handle, n, x, incx, stride_x, y, incy, stride_y, c, s, batch_count)
        return
    end function hipblasCrotStridedBatchedFortran

    function hipblasCsrotStridedBatchedFortran(handle, n, x, incx, stride_x, y, incy, stride_y, c, s, batch_count) &
            result(res) &
            bind(c, name = 'hipblasCsrotStridedBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int64_t), value :: stride_y
        type(c_ptr), value :: c
        type(c_ptr), value :: s
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasCsrotStridedBatched(handle, n, x, incx, stride_x, y, incy, stride_y, c, s, batch_count)
        return
    end function hipblasCsrotStridedBatchedFortran

    function hipblasZrotStridedBatchedFortran(handle, n, x, incx, stride_x, y, incy, stride_y, c, s, batch_count) &
            result(res) &
            bind(c, name = 'hipblasZrotStridedBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int64_t), value :: stride_y
        type(c_ptr), value :: c
        type(c_ptr), value :: s
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasZrotStridedBatched(handle, n, x, incx, stride_x, y, incy, stride_y, c, s, batch_count)
        return
    end function hipblasZrotStridedBatchedFortran

    function hipblasZdrotStridedBatchedFortran(handle, n, x, incx, stride_x, y, incy, stride_y, c, s, batch_count) &
            result(res) &
            bind(c, name = 'hipblasZdrotStridedBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int64_t), value :: stride_y
        type(c_ptr), value :: c
        type(c_ptr), value :: s
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasZdrotStridedBatched(handle, n, x, incx, stride_x, y, incy, stride_y, c, s, batch_count)
        return
    end function hipblasZdrotStridedBatchedFortran

    ! rotg
    function hipblasSrotgFortran(handle, a, b, c, s) &
            result(res) &
            bind(c, name = 'hipblasSrotgFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        type(c_ptr), value :: a
        type(c_ptr), value :: b
        type(c_ptr), value :: c
        type(c_ptr), value :: s
        integer(c_int) :: res
        res = hipblasSrotg(handle, a, b, c, s)
        return
    end function hipblasSrotgFortran

    function hipblasDrotgFortran(handle, a, b, c, s) &
            result(res) &
            bind(c, name = 'hipblasDrotgFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        type(c_ptr), value :: a
        type(c_ptr), value :: b
        type(c_ptr), value :: c
        type(c_ptr), value :: s
        integer(c_int) :: res
        res = hipblasDrotg(handle, a, b, c, s)
        return
    end function hipblasDrotgFortran

    function hipblasCrotgFortran(handle, a, b, c, s) &
            result(res) &
            bind(c, name = 'hipblasCrotgFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        type(c_ptr), value :: a
        type(c_ptr), value :: b
        type(c_ptr), value :: c
        type(c_ptr), value :: s
        integer(c_int) :: res
        res = hipblasCrotg(handle, a, b, c, s)
        return
    end function hipblasCrotgFortran

    function hipblasZrotgFortran(handle, a, b, c, s) &
            result(res) &
            bind(c, name = 'hipblasZrotgFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        type(c_ptr), value :: a
        type(c_ptr), value :: b
        type(c_ptr), value :: c
        type(c_ptr), value :: s
        integer(c_int) :: res
        res = hipblasZrotg(handle, a, b, c, s)
        return
    end function hipblasZrotgFortran

    ! rotgBatched
    function hipblasSrotgBatchedFortran(handle, a, b, c, s, batch_count) &
            result(res) &
            bind(c, name = 'hipblasSrotgBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        type(c_ptr), value :: a
        type(c_ptr), value :: b
        type(c_ptr), value :: c
        type(c_ptr), value :: s
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasSrotgBatched(handle, a, b, c, s, batch_count)
        return
    end function hipblasSrotgBatchedFortran

    function hipblasDrotgBatchedFortran(handle, a, b, c, s, batch_count) &
            result(res) &
            bind(c, name = 'hipblasDrotgBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        type(c_ptr), value :: a
        type(c_ptr), value :: b
        type(c_ptr), value :: c
        type(c_ptr), value :: s
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasDrotgBatched(handle, a, b, c, s, batch_count)
        return
    end function hipblasDrotgBatchedFortran

    function hipblasCrotgBatchedFortran(handle, a, b, c, s, batch_count) &
            result(res) &
            bind(c, name = 'hipblasCrotgBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        type(c_ptr), value :: a
        type(c_ptr), value :: b
        type(c_ptr), value :: c
        type(c_ptr), value :: s
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasCrotgBatched(handle, a, b, c, s, batch_count)
        return
    end function hipblasCrotgBatchedFortran

    function hipblasZrotgBatchedFortran(handle, a, b, c, s, batch_count) &
            result(res) &
            bind(c, name = 'hipblasZrotgBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        type(c_ptr), value :: a
        type(c_ptr), value :: b
        type(c_ptr), value :: c
        type(c_ptr), value :: s
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasZrotgBatched(handle, a, b, c, s, batch_count)
        return
    end function hipblasZrotgBatchedFortran

    ! rotgStridedBatched
    function hipblasSrotgStridedBatchedFortran(handle, a, stride_a, b, stride_b, c, stride_c, s, stride_s, batch_count) &
            result(res) &
            bind(c, name = 'hipblasSrotgStridedBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        type(c_ptr), value :: a
        integer(c_int64_t), value :: stride_a
        type(c_ptr), value :: b
        integer(c_int64_t), value :: stride_b
        type(c_ptr), value :: c
        integer(c_int64_t), value :: stride_c
        type(c_ptr), value :: s
        integer(c_int64_t), value :: stride_s
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasSrotgStridedBatched(handle, a, stride_a, b, stride_b, c, stride_c, s, stride_s, batch_count)
        return
    end function hipblasSrotgStridedBatchedFortran

    function hipblasDrotgStridedBatchedFortran(handle, a, stride_a, b, stride_b, c, stride_c, s, stride_s, batch_count) &
            result(res) &
            bind(c, name = 'hipblasDrotgStridedBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        type(c_ptr), value :: a
        integer(c_int64_t), value :: stride_a
        type(c_ptr), value :: b
        integer(c_int64_t), value :: stride_b
        type(c_ptr), value :: c
        integer(c_int64_t), value :: stride_c
        type(c_ptr), value :: s
        integer(c_int64_t), value :: stride_s
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasDrotgStridedBatched(handle, a, stride_a, b, stride_b, c, stride_c, s, stride_s, batch_count)
        return
    end function hipblasDrotgStridedBatchedFortran

    function hipblasCrotgStridedBatchedFortran(handle, a, stride_a, b, stride_b, c, stride_c, s, stride_s, batch_count) &
            result(res) &
            bind(c, name = 'hipblasCrotgStridedBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        type(c_ptr), value :: a
        integer(c_int64_t), value :: stride_a
        type(c_ptr), value :: b
        integer(c_int64_t), value :: stride_b
        type(c_ptr), value :: c
        integer(c_int64_t), value :: stride_c
        type(c_ptr), value :: s
        integer(c_int64_t), value :: stride_s
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasCrotgStridedBatched(handle, a, stride_a, b, stride_b, c, stride_c, s, stride_s, batch_count)
        return
    end function hipblasCrotgStridedBatchedFortran

    function hipblasZrotgStridedBatchedFortran(handle, a, stride_a, b, stride_b, c, stride_c, s, stride_s, batch_count) &
            result(res) &
            bind(c, name = 'hipblasZrotgStridedBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        type(c_ptr), value :: a
        integer(c_int64_t), value :: stride_a
        type(c_ptr), value :: b
        integer(c_int64_t), value :: stride_b
        type(c_ptr), value :: c
        integer(c_int64_t), value :: stride_c
        type(c_ptr), value :: s
        integer(c_int64_t), value :: stride_s
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasZrotgStridedBatched(handle, a, stride_a, b, stride_b, c, stride_c, s, stride_s, batch_count)
        return
    end function hipblasZrotgStridedBatchedFortran

    ! rotm
    function hipblasSrotmFortran(handle, n, x, incx, y, incy, param) &
            result(res) &
            bind(c, name = 'hipblasSrotmFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        type(c_ptr), value :: param
        integer(c_int) :: res
        res = hipblasSrotm(handle, n, x, incx, y, incy, param)
        return
    end function hipblasSrotmFortran

    function hipblasDrotmFortran(handle, n, x, incx, y, incy, param) &
            result(res) &
            bind(c, name = 'hipblasDrotmFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        type(c_ptr), value :: param
        integer(c_int) :: res
        res = hipblasDrotm(handle, n, x, incx, y, incy, param)
        return
    end function hipblasDrotmFortran

    ! rotmBatched
    function hipblasSrotmBatchedFortran(handle, n, x, incx, y, incy, param, batch_count) &
            result(res) &
            bind(c, name = 'hipblasSrotmBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        type(c_ptr), value :: param
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasSrotmBatched(handle, n, x, incx, y, incy, param, batch_count)
        return
    end function hipblasSrotmBatchedFortran

    function hipblasDrotmBatchedFortran(handle, n, x, incx, y, incy, param, batch_count) &
            result(res) &
            bind(c, name = 'hipblasDrotmBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        type(c_ptr), value :: param
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasDrotmBatched(handle, n, x, incx, y, incy, param, batch_count)
        return
    end function hipblasDrotmBatchedFortran

    ! rotmStridedBatched
    function hipblasSrotmStridedBatchedFortran(handle, n, x, incx, stride_x, y, incy, stride_y, param,&
        stride_param, batch_count) &
            result(res) &
            bind(c, name = 'hipblasSrotmStridedBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int64_t), value :: stride_y
        type(c_ptr), value :: param
        integer(c_int64_t), value :: stride_param
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasSrotmStridedBatched(handle, n, x, incx, stride_x, y, incy, stride_y, param,&
            stride_param, batch_count)
        return
    end function hipblasSrotmStridedBatchedFortran

    function hipblasDrotmStridedBatchedFortran(handle, n, x, incx, stride_x, y, incy, stride_y, param,&
        stride_param, batch_count) &
            result(res) &
            bind(c, name = 'hipblasDrotmStridedBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int64_t), value :: stride_y
        type(c_ptr), value :: param
        integer(c_int64_t), value :: stride_param
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasDrotmStridedBatched(handle, n, x, incx, stride_x, y, incy, stride_y, param,&
            stride_param, batch_count)
        return
    end function hipblasDrotmStridedBatchedFortran

    ! rotmg
    function hipblasSrotmgFortran(handle, d1, d2, x1, y1, param) &
            result(res) &
            bind(c, name = 'hipblasSrotmgFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        type(c_ptr), value :: d1
        type(c_ptr), value :: d2
        type(c_ptr), value :: x1
        type(c_ptr), value :: y1
        type(c_ptr), value :: param
        integer(c_int) :: res
        res = hipblasSrotmg(handle, d1, d2, x1, y1, param)
        return
    end function hipblasSrotmgFortran

    function hipblasDrotmgFortran(handle, d1, d2, x1, y1, param) &
            result(res) &
            bind(c, name = 'hipblasDrotmgFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        type(c_ptr), value :: d1
        type(c_ptr), value :: d2
        type(c_ptr), value :: x1
        type(c_ptr), value :: y1
        type(c_ptr), value :: param
        integer(c_int) :: res
        res = hipblasDrotmg(handle, d1, d2, x1, y1, param)
        return
    end function hipblasDrotmgFortran

    ! rotmgBatched
    function hipblasSrotmgBatchedFortran(handle, d1, d2, x1, y1, param, batch_count) &
            result(res) &
            bind(c, name = 'hipblasSrotmgBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        type(c_ptr), value :: d1
        type(c_ptr), value :: d2
        type(c_ptr), value :: x1
        type(c_ptr), value :: y1
        type(c_ptr), value :: param
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasSrotmgBatched(handle, d1, d2, x1, y1, param, batch_count)
        return
    end function hipblasSrotmgBatchedFortran

    function hipblasDrotmgBatchedFortran(handle, d1, d2, x1, y1, param, batch_count) &
            result(res) &
            bind(c, name = 'hipblasDrotmgBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        type(c_ptr), value :: d1
        type(c_ptr), value :: d2
        type(c_ptr), value :: x1
        type(c_ptr), value :: y1
        type(c_ptr), value :: param
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasDrotmgBatched(handle, d1, d2, x1, y1, param, batch_count)
        return
    end function hipblasDrotmgBatchedFortran

    ! rotmgStridedBatched
    function hipblasSrotmgStridedBatchedFortran(handle, d1, stride_d1, d2, stride_d2, x1, stride_x1, &
            y1, stride_y1, param, stride_param, batch_count) &
            result(res) &
            bind(c, name = 'hipblasSrotmgStridedBatchedFortran')
        use iso_c_binding
        implicit none
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
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasSrotmgStridedBatched(handle, d1, stride_d1, d2, stride_d2, x1, stride_x1, y1, stride_y1,&
            param, stride_param, batch_count)
        return
    end function hipblasSrotmgStridedBatchedFortran

    function hipblasDrotmgStridedBatchedFortran(handle, d1, stride_d1, d2, stride_d2, x1, stride_x1, &
            y1, stride_y1, param, stride_param, batch_count) &
            result(res) &
            bind(c, name = 'hipblasDrotmgStridedBatchedFortran')
        use iso_c_binding
        implicit none
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
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasDrotmgStridedBatched(handle, d1, stride_d1, d2, stride_d2, x1, stride_x1, y1, stride_y1,&
            param, stride_param, batch_count)
        return
    end function hipblasDrotmgStridedBatchedFortran

end module hipblas_interface