module hipblas_interface
    use iso_c_binding
    use hipblas

    contains

    !--------!
    !  Aux   !
    !--------!
    function hipblasSetVectorFortran(n, elemSize, x, incx, y, incy) &
            result(res) &
            bind(c, name = 'hipblasSetVectorFortran')
        use iso_c_binding
        implicit none
        integer(c_int), value :: n
        integer(c_int), value :: elemSize
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int) :: res
        res = hipblasSetVector(n, elemSize, x, incx, y, incy)
    end function hipblasSetVectorFortran

    function hipblasGetVectorFortran(n, elemSize, x, incx, y, incy) &
            result(res) &
            bind(c, name = 'hipblasGetVectorFortran')
        use iso_c_binding
        implicit none
        integer(c_int), value :: n
        integer(c_int), value :: elemSize
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int) :: res
        res = hipblasGetVector(n, elemSize, x, incx, y, incy)
    end function hipblasGetVectorFortran

    function hipblasSetMatrixFortran(rows, cols, elemSize, A, lda, B, ldb) &
            result(res) &
            bind(c, name = 'hipblasSetMatrixFortran')
        use iso_c_binding
        implicit none
        integer(c_int), value :: rows
        integer(c_int), value :: cols
        integer(c_int), value :: elemSize
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: B
        integer(c_int), value :: ldb
        integer(c_int) :: res
        res = hipblasSetMatrix(rows, cols, elemSize, A, lda, B, ldb)
    end function hipblasSetMatrixFortran

    function hipblasGetMatrixFortran(rows, cols, elemSize, A, lda, B, ldb) &
            result(res) &
            bind(c, name = 'hipblasGetMatrixFortran')
        use iso_c_binding
        implicit none
        integer(c_int), value :: rows
        integer(c_int), value :: cols
        integer(c_int), value :: elemSize
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: B
        integer(c_int), value :: ldb
        integer(c_int) :: res
        res = hipblasSetMatrix(rows, cols, elemSize, A, lda, B, ldb)
    end function hipblasGetMatrixFortran

    function hipblasSetVectorAsyncFortran(n, elemSize, x, incx, y, incy, stream) &
            result(res) &
            bind(c, name = 'hipblasSetVectorAsyncFortran')
        use iso_c_binding
        implicit none
        integer(c_int), value :: n
        integer(c_int), value :: elemSize
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        type(c_ptr), value :: stream
        integer(c_int) :: res
        res = hipblasSetVectorAsync(n, elemSize, x, incx, y, incy, stream)
    end function hipblasSetVectorAsyncFortran

    function hipblasGetVectorAsyncFortran(n, elemSize, x, incx, y, incy, stream) &
            result(res) &
            bind(c, name = 'hipblasGetVectorAsyncFortran')
        use iso_c_binding
        implicit none
        integer(c_int), value :: n
        integer(c_int), value :: elemSize
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        type(c_ptr), value :: stream
        integer(c_int) :: res
        res = hipblasGetVectorAsync(n, elemSize, x, incx, y, incy, stream)
    end function hipblasGetVectorAsyncFortran

    function hipblasSetMatrixAsyncFortran(rows, cols, elemSize, A, lda, B, ldb, stream) &
            result(res) &
            bind(c, name = 'hipblasSetMatrixAsyncFortran')
        use iso_c_binding
        implicit none
        integer(c_int), value :: rows
        integer(c_int), value :: cols
        integer(c_int), value :: elemSize
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: B
        integer(c_int), value :: ldb
        type(c_ptr), value :: stream
        integer(c_int) :: res
        res = hipblasSetMatrixAsync(rows, cols, elemSize, A, lda, B, ldb, stream)
    end function hipblasSetMatrixAsyncFortran

    function hipblasGetMatrixAsyncFortran(rows, cols, elemSize, A, lda, B, ldb, stream) &
            result(res) &
            bind(c, name = 'hipblasGetMatrixAsyncFortran')
        use iso_c_binding
        implicit none
        integer(c_int), value :: rows
        integer(c_int), value :: cols
        integer(c_int), value :: elemSize
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: B
        integer(c_int), value :: ldb
        type(c_ptr), value :: stream
        integer(c_int) :: res
        res = hipblasGetMatrixAsync(rows, cols, elemSize, A, lda, B, ldb, stream)
    end function hipblasGetMatrixAsyncFortran

    function hipblasSetAtomicsModeFortran(handle, atomics_mode) &
            result(res) &
            bind(c, name = 'hipblasSetAtomicsModeFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_ATOMICS_ALLOWED)), value :: atomics_mode
        integer(c_int) :: res
        res = hipblasSetAtomicsMode(handle, atomics_mode)
    end function hipblasSetAtomicsModeFortran

    function hipblasGetAtomicsModeFortran(handle, atomics_mode) &
            result(res) &
            bind(c, name = 'hipblasGetAtomicsModeFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        type(c_ptr), value :: atomics_mode
        integer(c_int) :: res
        res = hipblasGetAtomicsMode(handle, atomics_mode)
    end function hipblasGetAtomicsModeFortran

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

    !--------!
    ! blas 2 !
    !--------!

    ! gbmv
    function hipblasSgbmvFortran(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy) &
            result(res) &
            bind(c, name = 'hipblasSgbmvFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_OP_N)), value :: trans
        integer(c_int), value :: m
        integer(c_int), value :: n
        integer(c_int), value :: kl
        integer(c_int), value :: ku
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: beta
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int) :: res
        res = hipblasSgbmv(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy)
    end function hipblasSgbmvFortran

    function hipblasDgbmvFortran(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy) &
            result(res) &
            bind(c, name = 'hipblasDgbmvFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_OP_N)), value :: trans
        integer(c_int), value :: m
        integer(c_int), value :: n
        integer(c_int), value :: kl
        integer(c_int), value :: ku
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: beta
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int) :: res
        res = hipblasDgbmv(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy)
    end function hipblasDgbmvFortran

    function hipblasCgbmvFortran(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy) &
            result(res) &
            bind(c, name = 'hipblasCgbmvFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_OP_N)), value :: trans
        integer(c_int), value :: m
        integer(c_int), value :: n
        integer(c_int), value :: kl
        integer(c_int), value :: ku
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: beta
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int) :: res
        res = hipblasCgbmv(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy)
    end function hipblasCgbmvFortran

    function hipblasZgbmvFortran(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy) &
            result(res) &
            bind(c, name = 'hipblasZgbmvFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_OP_N)), value :: trans
        integer(c_int), value :: m
        integer(c_int), value :: n
        integer(c_int), value :: kl
        integer(c_int), value :: ku
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: beta
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int) :: res
        res = hipblasZgbmv(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy)
    end function hipblasZgbmvFortran

    ! gbmvBatched
    function hipblasSgbmvBatchedFortran(handle, trans, m, n, kl, ku, alpha, A, lda, &
            x, incx, beta, y, incy, batch_count) &
            result(res) &
            bind(c, name = 'hipblasSgbmvBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_OP_N)), value :: trans
        integer(c_int), value :: m
        integer(c_int), value :: n
        integer(c_int), value :: kl
        integer(c_int), value :: ku
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: beta
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasSgbmvBatched(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx,&
              beta, y, incy, batch_count)
    end function hipblasSgbmvBatchedFortran

    function hipblasDgbmvBatchedFortran(handle, trans, m, n, kl, ku, alpha, A, lda, &
            x, incx, beta, y, incy, batch_count) &
            result(res) &
            bind(c, name = 'hipblasDgbmvBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_OP_N)), value :: trans
        integer(c_int), value :: m
        integer(c_int), value :: n
        integer(c_int), value :: kl
        integer(c_int), value :: ku
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: beta
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasDgbmvBatched(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx,&
              beta, y, incy, batch_count)
    end function hipblasDgbmvBatchedFortran

    function hipblasCgbmvBatchedFortran(handle, trans, m, n, kl, ku, alpha, A, lda, &
            x, incx, beta, y, incy, batch_count) &
            result(res) &
            bind(c, name = 'hipblasCgbmvBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_OP_N)), value :: trans
        integer(c_int), value :: m
        integer(c_int), value :: n
        integer(c_int), value :: kl
        integer(c_int), value :: ku
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: beta
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasCgbmvBatched(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx,&
              beta, y, incy, batch_count)
    end function hipblasCgbmvBatchedFortran

    function hipblasZgbmvBatchedFortran(handle, trans, m, n, kl, ku, alpha, A, lda, &
            x, incx, beta, y, incy, batch_count) &
            result(res) &
            bind(c, name = 'hipblasZgbmvBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_OP_N)), value :: trans
        integer(c_int), value :: m
        integer(c_int), value :: n
        integer(c_int), value :: kl
        integer(c_int), value :: ku
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: beta
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasZgbmvBatched(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx,&
              beta, y, incy, batch_count)
    end function hipblasZgbmvBatchedFortran

    ! gbmvStridedBatched
    function hipblasSgbmvStridedBatchedFortran(handle, trans, m, n, kl, ku, alpha, A, lda, stride_A, &
            x, incx, stride_x, beta, y, incy, stride_y, batch_count) &
            result(res) &
            bind(c, name = 'hipblasSgbmvStridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_OP_N)), value :: trans
        integer(c_int), value :: m
        integer(c_int), value :: n
        integer(c_int), value :: kl
        integer(c_int), value :: ku
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int64_t), value :: stride_A
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        type(c_ptr), value :: beta
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int64_t), value :: stride_y
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasSgbmvStridedBatched(handle, trans, m, n, kl, ku, alpha, A, lda, stride_A, x, incx, stride_x,&
              beta, y, incy, stride_y, batch_count)
    end function hipblasSgbmvStridedBatchedFortran

    function hipblasDgbmvStridedBatchedFortran(handle, trans, m, n, kl, ku, alpha, A, lda, stride_A, &
            x, incx, stride_x, beta, y, incy, stride_y, batch_count) &
            result(res) &
            bind(c, name = 'hipblasDgbmvStridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_OP_N)), value :: trans
        integer(c_int), value :: m
        integer(c_int), value :: n
        integer(c_int), value :: kl
        integer(c_int), value :: ku
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int64_t), value :: stride_A
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        type(c_ptr), value :: beta
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int64_t), value :: stride_y
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasDgbmvStridedBatched(handle, trans, m, n, kl, ku, alpha, A, lda, stride_A, x, incx, stride_x,&
              beta, y, incy, stride_y, batch_count)
    end function hipblasDgbmvStridedBatchedFortran

    function hipblasCgbmvStridedBatchedFortran(handle, trans, m, n, kl, ku, alpha, A, lda, stride_A, &
            x, incx, stride_x, beta, y, incy, stride_y, batch_count) &
            result(res) &
            bind(c, name = 'hipblasCgbmvStridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_OP_N)), value :: trans
        integer(c_int), value :: m
        integer(c_int), value :: n
        integer(c_int), value :: kl
        integer(c_int), value :: ku
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int64_t), value :: stride_A
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        type(c_ptr), value :: beta
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int64_t), value :: stride_y
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasCgbmvStridedBatched(handle, trans, m, n, kl, ku, alpha, A, lda, stride_A, x, incx, stride_x,&
              beta, y, incy, stride_y, batch_count)
    end function hipblasCgbmvStridedBatchedFortran

    function hipblasZgbmvStridedBatchedFortran(handle, trans, m, n, kl, ku, alpha, A, lda, stride_A, &
            x, incx, stride_x, beta, y, incy, stride_y, batch_count) &
            result(res) &
            bind(c, name = 'hipblasZgbmvStridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_OP_N)), value :: trans
        integer(c_int), value :: m
        integer(c_int), value :: n
        integer(c_int), value :: kl
        integer(c_int), value :: ku
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int64_t), value :: stride_A
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        type(c_ptr), value :: beta
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int64_t), value :: stride_y
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasZgbmvStridedBatched(handle, trans, m, n, kl, ku, alpha, A, lda, stride_A, x, incx, stride_x,&
              beta, y, incy, stride_y, batch_count)
    end function hipblasZgbmvStridedBatchedFortran

    ! gemv
    function hipblasSgemvFortran(handle, trans, m, n, alpha, A, lda, &
            x, incx, beta, y, incy) &
            result(res) &
            bind(c, name = 'hipblasSgemvFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_OP_N)), value :: trans
        integer(c_int), value :: m
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: beta
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int) :: res
        res = hipblasSgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy)
    end function hipblasSgemvFortran

    function hipblasDgemvFortran(handle, trans, m, n, alpha, A, lda, &
            x, incx, beta, y, incy) &
            result(res) &
            bind(c, name = 'hipblasDgemvFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_OP_N)), value :: trans
        integer(c_int), value :: m
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: beta
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int) :: res
        res = hipblasDgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy)
    end function hipblasDgemvFortran

    function hipblasCgemvFortran(handle, trans, m, n, alpha, A, lda, &
            x, incx, beta, y, incy) &
            result(res) &
            bind(c, name = 'hipblasCgemvFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_OP_N)), value :: trans
        integer(c_int), value :: m
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: beta
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int) :: res
        res = hipblasCgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy)
    end function hipblasCgemvFortran

    function hipblasZgemvFortran(handle, trans, m, n, alpha, A, lda, &
            x, incx, beta, y, incy) &
            result(res) &
            bind(c, name = 'hipblasZgemvFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_OP_N)), value :: trans
        integer(c_int), value :: m
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: beta
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int) :: res
        res = hipblasZgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy)
    end function hipblasZgemvFortran

    ! gemvBatched
    function hipblasSgemvBatchedFortran(handle, trans, m, n, alpha, A, lda, &
            x, incx, beta, y, incy, batch_count) &
            result(res) &
            bind(c, name = 'hipblasSgemvBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_OP_N)), value :: trans
        integer(c_int), value :: m
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: beta
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasSgemvBatched(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy, batch_count)
    end function hipblasSgemvBatchedFortran

    function hipblasDgemvBatchedFortran(handle, trans, m, n, alpha, A, lda, &
            x, incx, beta, y, incy, batch_count) &
            result(res) &
            bind(c, name = 'hipblasDgemvBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_OP_N)), value :: trans
        integer(c_int), value :: m
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: beta
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasDgemvBatched(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy, batch_count)
    end function hipblasDgemvBatchedFortran

    function hipblasCgemvBatchedFortran(handle, trans, m, n, alpha, A, lda, &
            x, incx, beta, y, incy, batch_count) &
            result(res) &
            bind(c, name = 'hipblasCgemvBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_OP_N)), value :: trans
        integer(c_int), value :: m
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: beta
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasCgemvBatched(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy, batch_count)
    end function hipblasCgemvBatchedFortran

    function hipblasZgemvBatchedFortran(handle, trans, m, n, alpha, A, lda, &
            x, incx, beta, y, incy, batch_count) &
            result(res) &
            bind(c, name = 'hipblasZgemvBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_OP_N)), value :: trans
        integer(c_int), value :: m
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: beta
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasZgemvBatched(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy, batch_count)
    end function hipblasZgemvBatchedFortran

    ! gemvStridedBatched
    function hipblasSgemvStridedBatchedFortran(handle, trans, m, n, alpha, A, lda, stride_A, &
            x, incx, stride_x, beta, y, incy, stride_y, batch_count) &
            result(res) &
            bind(c, name = 'hipblasSgemvStridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_OP_N)), value :: trans
        integer(c_int), value :: m
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int64_t), value :: stride_A
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        type(c_ptr), value :: beta
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int64_t), value :: stride_y
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasSgemvStridedBatched(handle, trans, m, n, alpha, A, lda, stride_A,&
              x, incx, stride_x, beta, y, incy, stride_y, batch_count)
    end function hipblasSgemvStridedBatchedFortran

    function hipblasDgemvStridedBatchedFortran(handle, trans, m, n, alpha, A, lda, stride_A, &
            x, incx, stride_x, beta, y, incy, stride_y, batch_count) &
            result(res) &
            bind(c, name = 'hipblasDgemvStridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_OP_N)), value :: trans
        integer(c_int), value :: m
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int64_t), value :: stride_A
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        type(c_ptr), value :: beta
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int64_t), value :: stride_y
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasDgemvStridedBatched(handle, trans, m, n, alpha, A, lda, stride_A,&
              x, incx, stride_x, beta, y, incy, stride_y, batch_count)
    end function hipblasDgemvStridedBatchedFortran

    function hipblasCgemvStridedBatchedFortran(handle, trans, m, n, alpha, A, lda, stride_A, &
            x, incx, stride_x, beta, y, incy, stride_y, batch_count) &
            result(res) &
            bind(c, name = 'hipblasCgemvStridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_OP_N)), value :: trans
        integer(c_int), value :: m
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int64_t), value :: stride_A
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        type(c_ptr), value :: beta
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int64_t), value :: stride_y
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasCgemvStridedBatched(handle, trans, m, n, alpha, A, lda, stride_A,&
              x, incx, stride_x, beta, y, incy, stride_y, batch_count)
    end function hipblasCgemvStridedBatchedFortran

    function hipblasZgemvStridedBatchedFortran(handle, trans, m, n, alpha, A, lda, stride_A, &
            x, incx, stride_x, beta, y, incy, stride_y, batch_count) &
            result(res) &
            bind(c, name = 'hipblasZgemvStridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_OP_N)), value :: trans
        integer(c_int), value :: m
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int64_t), value :: stride_A
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        type(c_ptr), value :: beta
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int64_t), value :: stride_y
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasZgemvStridedBatched(handle, trans, m, n, alpha, A, lda, stride_A,&
              x, incx, stride_x, beta, y, incy, stride_y, batch_count)
    end function hipblasZgemvStridedBatchedFortran

    ! hbmv
    function hipblasChbmvFortran(handle, uplo, n, k, alpha, A, lda, &
            x, incx, beta, y, incy) &
            result(res) &
            bind(c, name = 'hipblasChbmvFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        integer(c_int), value :: k
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: beta
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int) :: res
        res = hipblasChbmv(handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy)
    end function hipblasChbmvFortran

    function hipblasZhbmvFortran(handle, uplo, n, k, alpha, A, lda, &
            x, incx, beta, y, incy) &
            result(res) &
            bind(c, name = 'hipblasZhbmvFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        integer(c_int), value :: k
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: beta
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int) :: res
        res = hipblasZhbmv(handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy)
    end function hipblasZhbmvFortran

    ! hbmvBatched
    function hipblasChbmvBatchedFortran(handle, uplo, n, k, alpha, A, lda, &
            x, incx, beta, y, incy, batch_count) &
            result(res) &
            bind(c, name = 'hipblasChbmvBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        integer(c_int), value :: k
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: beta
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasChbmvBatched(handle, uplo, n, k, alpha, A, lda,&
              x, incx, beta, y, incy, batch_count)
    end function hipblasChbmvBatchedFortran

    function hipblasZhbmvBatchedFortran(handle, uplo, n, k, alpha, A, lda, &
            x, incx, beta, y, incy, batch_count) &
            result(res) &
            bind(c, name = 'hipblasZhbmvBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        integer(c_int), value :: k
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: beta
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasZhbmvBatched(handle, uplo, n, k, alpha, A, lda,&
              x, incx, beta, y, incy, batch_count)
    end function hipblasZhbmvBatchedFortran

    ! hbmvStridedBatched
    function hipblasChbmvStridedBatchedFortran(handle, uplo, n, k, alpha, A, lda, stride_A, &
            x, incx, stride_x, beta, y, incy, stride_y, batch_count) &
            result(res) &
            bind(c, name = 'hipblasChbmvStridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        integer(c_int), value :: k
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int64_t), value :: stride_A
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        type(c_ptr), value :: beta
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int64_t), value :: stride_y
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasChbmvStridedBatched(handle, uplo, n, k, alpha, A, lda, stride_A,&
              x, incx, stride_x, beta, y, incy, stride_y, batch_count)
    end function hipblasChbmvStridedBatchedFortran

    function hipblasZhbmvStridedBatchedFortran(handle, uplo, n, k, alpha, A, lda, stride_A, &
            x, incx, stride_x, beta, y, incy, stride_y, batch_count) &
            result(res) &
            bind(c, name = 'hipblasZhbmvStridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        integer(c_int), value :: k
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int64_t), value :: stride_A
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        type(c_ptr), value :: beta
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int64_t), value :: stride_y
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasZhbmvStridedBatched(handle, uplo, n, k, alpha, A, lda, stride_A,&
              x, incx, stride_x, beta, y, incy, stride_y, batch_count)
    end function hipblasZhbmvStridedBatchedFortran

    ! hemv
    function hipblasChemvFortran(handle, uplo, n, alpha, A, lda, &
            x, incx, beta, y, incy) &
            result(res) &
            bind(c, name = 'hipblasChemvFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: beta
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int) :: res
        res = hipblasChemv(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy)
    end function hipblasChemvFortran

    function hipblasZhemvFortran(handle, uplo, n, alpha, A, lda, &
            x, incx, beta, y, incy) &
            result(res) &
            bind(c, name = 'hipblasZhemvFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: beta
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int) :: res
        res = hipblasZhemv(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy)
    end function hipblasZhemvFortran

    ! hemvBatched
    function hipblasChemvBatchedFortran(handle, uplo, n, alpha, A, lda, &
            x, incx, beta, y, incy, batch_count) &
            result(res) &
            bind(c, name = 'hipblasChemvBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: beta
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasChemvBatched(handle, uplo, n, alpha, A, lda,&
              x, incx, beta, y, incy, batch_count)
    end function hipblasChemvBatchedFortran

    function hipblasZhemvBatchedFortran(handle, uplo, n, alpha, A, lda, &
            x, incx, beta, y, incy, batch_count) &
            result(res) &
            bind(c, name = 'hipblasZhemvBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: beta
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasZhemvBatched(handle, uplo, n, alpha, A, lda,&
              x, incx, beta, y, incy, batch_count)
    end function hipblasZhemvBatchedFortran

    ! hemvStridedBatched
    function hipblasChemvStridedBatchedFortran(handle, uplo, n, alpha, A, lda, stride_A, &
            x, incx, stride_x, beta, y, incy, stride_y, batch_count) &
            result(res) &
            bind(c, name = 'hipblasChemvStridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int64_t), value :: stride_A
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        type(c_ptr), value :: beta
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int64_t), value :: stride_y
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasChemvStridedBatched(handle, uplo, n, alpha, A, lda, stride_A,&
              x, incx, stride_x, beta, y, incy, stride_y, batch_count)
    end function hipblasChemvStridedBatchedFortran

    function hipblasZhemvStridedBatchedFortran(handle, uplo, n, alpha, A, lda, stride_A, &
            x, incx, stride_x, beta, y, incy, stride_y, batch_count) &
            result(res) &
            bind(c, name = 'hipblasZhemvStridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int64_t), value :: stride_A
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        type(c_ptr), value :: beta
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int64_t), value :: stride_y
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasZhemvStridedBatched(handle, uplo, n, alpha, A, lda, stride_A,&
              x, incx, stride_x, beta, y, incy, stride_y, batch_count)
    end function hipblasZhemvStridedBatchedFortran

    ! her
    function hipblasCherFortran(handle, uplo, n, alpha, &
            x, incx, A, lda) &
            result(res) &
            bind(c, name = 'hipblasCherFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int) :: res
        res = hipblasCher(handle, uplo, n, alpha, x, incx, A, lda)
    end function hipblasCherFortran

    function hipblasZherFortran(handle, uplo, n, alpha, &
            x, incx, A, lda) &
            result(res) &
            bind(c, name = 'hipblasZherFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int) :: res
        res = hipblasZher(handle, uplo, n, alpha, x, incx, A, lda)
    end function hipblasZherFortran
    
    ! herBatched
    function hipblasCherBatchedFortran(handle, uplo, n, alpha, &
            x, incx, A, lda, batch_count) &
            result(res) &
            bind(c, name = 'hipblasCherBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasCherBatched(handle, uplo, n, alpha, x, incx, A, lda, batch_count)
    end function hipblasCherBatchedFortran

    function hipblasZherBatchedFortran(handle, uplo, n, alpha, &
            x, incx, A, lda, batch_count) &
            result(res) &
            bind(c, name = 'hipblasZherBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasZherBatched(handle, uplo, n, alpha, x, incx, A, lda, batch_count)
    end function hipblasZherBatchedFortran

    ! herStridedBatched
    function hipblasCherStridedBatchedFortran(handle, uplo, n, alpha, &
            x, incx, stride_x, A, lda, stride_A, batch_count) &
            result(res) &
            bind(c, name = 'hipblasCherStridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int64_t), value :: stride_A
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasCherStridedBatched(handle, uplo, n, alpha, x, incx, stride_x,&
              A, lda, stride_A, batch_count)
    end function hipblasCherStridedBatchedFortran

    function hipblasZherStridedBatchedFortran(handle, uplo, n, alpha, &
            x, incx, stride_x, A, lda, stride_A, batch_count) &
            result(res) &
            bind(c, name = 'hipblasZherStridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int64_t), value :: stride_A
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasZherStridedBatched(handle, uplo, n, alpha, x, incx, stride_x,&
              A, lda, stride_A, batch_count)
    end function hipblasZherStridedBatchedFortran

    ! her2
    function hipblasCher2Fortran(handle, uplo, n, alpha, &
            x, incx, y, incy, A, lda) &
            result(res) &
            bind(c, name = 'hipblasCher2Fortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int) :: res
        res = hipblasCher2(handle, uplo, n, alpha, x, incx,&
              y, incy, A, lda)
    end function hipblasCher2Fortran

    function hipblasZher2Fortran(handle, uplo, n, alpha, &
            x, incx, y, incy, A, lda) &
            result(res) &
            bind(c, name = 'hipblasZher2Fortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int) :: res
        res = hipblasZher2(handle, uplo, n, alpha, x, incx,&
              y, incy, A, lda)
    end function hipblasZher2Fortran

    ! her2Batched
    function hipblasCher2BatchedFortran(handle, uplo, n, alpha, &
            x, incx, y, incy, A, lda, batch_count) &
            result(res) &
            bind(c, name = 'hipblasCher2BatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasCher2Batched(handle, uplo, n, alpha, x, incx,&
              y, incy, A, lda, batch_count)
    end function hipblasCher2BatchedFortran

    function hipblasZher2BatchedFortran(handle, uplo, n, alpha, &
            x, incx, y, incy, A, lda, batch_count) &
            result(res) &
            bind(c, name = 'hipblasZher2BatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasZher2Batched(handle, uplo, n, alpha, x, incx,&
              y, incy, A, lda, batch_count)
    end function hipblasZher2BatchedFortran

    ! her2StridedBatched
    function hipblasCher2StridedBatchedFortran(handle, uplo, n, alpha, &
            x, incx, stride_x, y, incy, stride_y, A, lda, stride_A, batch_count) &
            result(res) &
            bind(c, name = 'hipblasCher2StridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int64_t), value :: stride_y
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int64_t), value :: stride_A
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasCher2StridedBatched(handle, uplo, n, alpha, x, incx, stride_x,&
              y, incy, stride_y, A, lda, stride_A, batch_count)
    end function hipblasCher2StridedBatchedFortran

    function hipblasZher2StridedBatchedFortran(handle, uplo, n, alpha, &
            x, incx, stride_x, y, incy, stride_y, A, lda, stride_A, batch_count) &
            result(res) &
            bind(c, name = 'hipblasZher2StridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int64_t), value :: stride_y
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int64_t), value :: stride_A
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasZher2StridedBatched(handle, uplo, n, alpha, x, incx, stride_x,&
              y, incy, stride_y, A, lda, stride_A, batch_count)
    end function hipblasZher2StridedBatchedFortran

    ! hpmv
    function hipblasChpmvFortran(handle, uplo, n, alpha, AP, &
            x, incx, beta, y, incy) &
            result(res) &
            bind(c, name = 'hipblasChpmvFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: AP
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: beta
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int) :: res
        res = hipblasChpmv(handle, uplo, n, alpha, AP,&
              x, incx, beta, y, incy)
    end function hipblasChpmvFortran

    function hipblasZhpmvFortran(handle, uplo, n, alpha, AP, &
            x, incx, beta, y, incy) &
            result(res) &
            bind(c, name = 'hipblasZhpmvFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: AP
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: beta
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int) :: res
        res = hipblasZhpmv(handle, uplo, n, alpha, AP,&
              x, incx, beta, y, incy)
    end function hipblasZhpmvFortran

    ! hpmvBatched
    function hipblasChpmvBatchedFortran(handle, uplo, n, alpha, AP, &
            x, incx, beta, y, incy, batch_count) &
            result(res) &
            bind(c, name = 'hipblasChpmvBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: AP
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: beta
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasChpmvBatched(handle, uplo, n, alpha, AP,&
              x, incx, beta, y, incy, batch_count)
    end function hipblasChpmvBatchedFortran

    function hipblasZhpmvBatchedFortran(handle, uplo, n, alpha, AP, &
            x, incx, beta, y, incy, batch_count) &
            result(res) &
            bind(c, name = 'hipblasZhpmvBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: AP
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: beta
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasZhpmvBatched(handle, uplo, n, alpha, AP,&
              x, incx, beta, y, incy, batch_count)
    end function hipblasZhpmvBatchedFortran

    ! hpmvStridedBatched
    function hipblasChpmvStridedBatchedFortran(handle, uplo, n, alpha, AP, stride_AP, &
            x, incx, stride_x, beta, y, incy, stride_y, batch_count) &
            result(res) &
            bind(c, name = 'hipblasChpmvStridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: AP
        integer(c_int64_t), value :: stride_AP
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        type(c_ptr), value :: beta
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int64_t), value :: stride_y
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasChpmvStridedBatched(handle, uplo, n, alpha, AP, stride_AP,&
              x, incx, stride_x, beta, y, incy, stride_y, batch_count)
    end function hipblasChpmvStridedBatchedFortran

    function hipblasZhpmvStridedBatchedFortran(handle, uplo, n, alpha, AP, stride_AP, &
            x, incx, stride_x, beta, y, incy, stride_y, batch_count) &
            result(res) &
            bind(c, name = 'hipblasZhpmvStridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: AP
        integer(c_int64_t), value :: stride_AP
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        type(c_ptr), value :: beta
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int64_t), value :: stride_y
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasZhpmvStridedBatched(handle, uplo, n, alpha, AP, stride_AP,&
              x, incx, stride_x, beta, y, incy, stride_y, batch_count)
    end function hipblasZhpmvStridedBatchedFortran

    ! hpr
    function hipblasChprFortran(handle, uplo, n, alpha, &
            x, incx, AP) &
            result(res) &
            bind(c, name = 'hipblasChprFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: AP
        integer(c_int) :: res
        res = hipblasChpr(handle, uplo, n, alpha, x, incx, AP)
    end function hipblasChprFortran

    function hipblasZhprFortran(handle, uplo, n, alpha, &
            x, incx, AP) &
            result(res) &
            bind(c, name = 'hipblasZhprFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: AP
        integer(c_int) :: res
        res = hipblasZhpr(handle, uplo, n, alpha, x, incx, AP)
    end function hipblasZhprFortran

    ! hprBatched
    function hipblasChprBatchedFortran(handle, uplo, n, alpha, &
            x, incx, AP, batch_count) &
            result(res) &
            bind(c, name = 'hipblasChprBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: AP
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasChprBatched(handle, uplo, n, alpha, x, incx, AP, batch_count)
    end function hipblasChprBatchedFortran

    function hipblasZhprBatchedFortran(handle, uplo, n, alpha, &
            x, incx, AP, batch_count) &
            result(res) &
            bind(c, name = 'hipblasZhprBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: AP
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasZhprBatched(handle, uplo, n, alpha, x, incx, AP, batch_count)
    end function hipblasZhprBatchedFortran

    ! hprStridedBatched
    function hipblasChprStridedBatchedFortran(handle, uplo, n, alpha, &
            x, incx, stride_x, AP, stride_AP, batch_count) &
            result(res) &
            bind(c, name = 'hipblasChprStridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        type(c_ptr), value :: AP
        integer(c_int64_t), value :: stride_AP
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasChprStridedBatched(handle, uplo, n, alpha, x, incx, stride_x,&
              AP, stride_AP, batch_count)
    end function hipblasChprStridedBatchedFortran

    function hipblasZhprStridedBatchedFortran(handle, uplo, n, alpha, &
            x, incx, stride_x, AP, stride_AP, batch_count) &
            result(res) &
            bind(c, name = 'hipblasZhprStridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        type(c_ptr), value :: AP
        integer(c_int64_t), value :: stride_AP
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasZhprStridedBatched(handle, uplo, n, alpha, x, incx, stride_x,&
              AP, stride_AP, batch_count)
    end function hipblasZhprStridedBatchedFortran

    ! hpr2
    function hipblasChpr2Fortran(handle, uplo, n, alpha, &
            x, incx, y, incy, AP) &
            result(res) &
            bind(c, name = 'hipblasChpr2Fortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        type(c_ptr), value :: AP
        integer(c_int) :: res
        res = hipblasChpr2(handle, uplo, n, alpha, x, incx,&
              y, incy, AP)
    end function hipblasChpr2Fortran

    function hipblasZhpr2Fortran(handle, uplo, n, alpha, &
            x, incx, y, incy, AP) &
            result(res) &
            bind(c, name = 'hipblasZhpr2Fortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        type(c_ptr), value :: AP
        integer(c_int) :: res
        res = hipblasZhpr2(handle, uplo, n, alpha, x, incx,&
              y, incy, AP)
    end function hipblasZhpr2Fortran

    ! hpr2Batched
    function hipblasChpr2BatchedFortran(handle, uplo, n, alpha, &
            x, incx, y, incy, AP, batch_count) &
            result(res) &
            bind(c, name = 'hipblasChpr2BatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        type(c_ptr), value :: AP
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasChpr2Batched(handle, uplo, n, alpha, x, incx,&
              y, incy, AP, batch_count)
    end function hipblasChpr2BatchedFortran

    function hipblasZhpr2BatchedFortran(handle, uplo, n, alpha, &
            x, incx, y, incy, AP, batch_count) &
            result(res) &
            bind(c, name = 'hipblasZhpr2BatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        type(c_ptr), value :: AP
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasZhpr2Batched(handle, uplo, n, alpha, x, incx,&
              y, incy, AP, batch_count)
    end function hipblasZhpr2BatchedFortran

    ! hpr2StridedBatched
    function hipblasChpr2StridedBatchedFortran(handle, uplo, n, alpha, &
            x, incx, stride_x, y, incy, stride_y, AP, stride_AP, batch_count) &
            result(res) &
            bind(c, name = 'hipblasChpr2StridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int64_t), value :: stride_y
        type(c_ptr), value :: AP
        integer(c_int64_t), value :: stride_AP
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasChpr2StridedBatched(handle, uplo, n, alpha, x, incx, stride_x,&
              y, incy, stride_y, AP, stride_AP, batch_count)
    end function hipblasChpr2StridedBatchedFortran

    function hipblasZhpr2StridedBatchedFortran(handle, uplo, n, alpha, &
            x, incx, stride_x, y, incy, stride_y, AP, stride_AP, batch_count) &
            result(res) &
            bind(c, name = 'hipblasZhpr2StridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int64_t), value :: stride_y
        type(c_ptr), value :: AP
        integer(c_int64_t), value :: stride_AP
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasZhpr2StridedBatched(handle, uplo, n, alpha, x, incx, stride_x,&
              y, incy, stride_y, AP, stride_AP, batch_count)
    end function hipblasZhpr2StridedBatchedFortran

    ! trmv
    function hipblasStrmvFortran(handle, uplo, transA, diag, m, &
            A, lda, x, incx) &
            result(res) &
            bind(c, name = 'hipblasStrmvFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
        integer(c_int), value :: m
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int) :: res
        res = hipblasStrmv(handle, uplo, transA, diag, m,&
              A, lda, x, incx)
    end function hipblasStrmvFortran

    function hipblasDtrmvFortran(handle, uplo, transA, diag, m, &
            A, lda, x, incx) &
            result(res) &
            bind(c, name = 'hipblasDtrmvFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
        integer(c_int), value :: m
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int) :: res
        res = hipblasDtrmv(handle, uplo, transA, diag, m,&
              A, lda, x, incx)
    end function hipblasDtrmvFortran

    function hipblasCtrmvFortran(handle, uplo, transA, diag, m, &
            A, lda, x, incx) &
            result(res) &
            bind(c, name = 'hipblasCtrmvFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
        integer(c_int), value :: m
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int) :: res
        res = hipblasCtrmv(handle, uplo, transA, diag, m,&
              A, lda, x, incx)
    end function hipblasCtrmvFortran

    function hipblasZtrmvFortran(handle, uplo, transA, diag, m, &
            A, lda, x, incx) &
            result(res) &
            bind(c, name = 'hipblasZtrmvFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
        integer(c_int), value :: m
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int) :: res
        res = hipblasZtrmv(handle, uplo, transA, diag, m,&
              A, lda, x, incx)
    end function hipblasZtrmvFortran

    ! trmvBatched
    function hipblasStrmvBatchedFortran(handle, uplo, transA, diag, m, &
            A, lda, x, incx, batch_count) &
            result(res) &
            bind(c, name = 'hipblasStrmvBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
        integer(c_int), value :: m
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasStrmvBatched(handle, uplo, transA, diag, m,&
              A, lda, x, incx, batch_count)
    end function hipblasStrmvBatchedFortran

    function hipblasDtrmvBatchedFortran(handle, uplo, transA, diag, m, &
            A, lda, x, incx, batch_count) &
            result(res) &
            bind(c, name = 'hipblasDtrmvBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
        integer(c_int), value :: m
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasDtrmvBatched(handle, uplo, transA, diag, m,&
              A, lda, x, incx, batch_count)
    end function hipblasDtrmvBatchedFortran

    function hipblasCtrmvBatchedFortran(handle, uplo, transA, diag, m, &
            A, lda, x, incx, batch_count) &
            result(res) &
            bind(c, name = 'hipblasCtrmvBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
        integer(c_int), value :: m
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasCtrmvBatched(handle, uplo, transA, diag, m,&
              A, lda, x, incx, batch_count)
    end function hipblasCtrmvBatchedFortran

    function hipblasZtrmvBatchedFortran(handle, uplo, transA, diag, m, &
            A, lda, x, incx, batch_count) &
            result(res) &
            bind(c, name = 'hipblasZtrmvBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
        integer(c_int), value :: m
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasZtrmvBatched(handle, uplo, transA, diag, m,&
              A, lda, x, incx, batch_count)
    end function hipblasZtrmvBatchedFortran

    ! trmvStridedBatched
    function hipblasStrmvStridedBatchedFortran(handle, uplo, transA, diag, m, &
            A, lda, stride_A, x, incx, stride_x, batch_count) &
            result(res) &
            bind(c, name = 'hipblasStrmvStridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
        integer(c_int), value :: m
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int64_t), value :: stride_A
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasStrmvStridedBatched(handle, uplo, transA, diag, m,&
              A, lda, stride_A, x, incx, stride_x, batch_count)
    end function hipblasStrmvStridedBatchedFortran

    function hipblasDtrmvStridedBatchedFortran(handle, uplo, transA, diag, m, &
            A, lda, stride_A, x, incx, stride_x, batch_count) &
            result(res) &
            bind(c, name = 'hipblasDtrmvStridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
        integer(c_int), value :: m
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int64_t), value :: stride_A
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasDtrmvStridedBatched(handle, uplo, transA, diag, m,&
              A, lda, stride_A, x, incx, stride_x, batch_count)
    end function hipblasDtrmvStridedBatchedFortran

    function hipblasCtrmvStridedBatchedFortran(handle, uplo, transA, diag, m, &
            A, lda, stride_A, x, incx, stride_x, batch_count) &
            result(res) &
            bind(c, name = 'hipblasCtrmvStridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
        integer(c_int), value :: m
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int64_t), value :: stride_A
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasCtrmvStridedBatched(handle, uplo, transA, diag, m,&
              A, lda, stride_A, x, incx, stride_x, batch_count)
    end function hipblasCtrmvStridedBatchedFortran

    function hipblasZtrmvStridedBatchedFortran(handle, uplo, transA, diag, m, &
            A, lda, stride_A, x, incx, stride_x, batch_count) &
            result(res) &
            bind(c, name = 'hipblasZtrmvStridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
        integer(c_int), value :: m
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int64_t), value :: stride_A
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasZtrmvStridedBatched(handle, uplo, transA, diag, m,&
              A, lda, stride_A, x, incx, stride_x, batch_count)
    end function hipblasZtrmvStridedBatchedFortran

    ! tpmv
    function hipblasStpmvFortran(handle, uplo, transA, diag, m, &
            AP, x, incx) &
            result(res) &
            bind(c, name = 'hipblasStpmvFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
        integer(c_int), value :: m
        type(c_ptr), value :: AP
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int) :: res
        res = hipblasStpmv(handle, uplo, transA, diag, m,&
              AP, x, incx)
    end function hipblasStpmvFortran

    function hipblasDtpmvFortran(handle, uplo, transA, diag, m, &
            AP, x, incx) &
            result(res) &
            bind(c, name = 'hipblasDtpmvFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
        integer(c_int), value :: m
        type(c_ptr), value :: AP
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int) :: res
        res = hipblasDtpmv(handle, uplo, transA, diag, m,&
              AP, x, incx)
    end function hipblasDtpmvFortran

    function hipblasCtpmvFortran(handle, uplo, transA, diag, m, &
            AP, x, incx) &
            result(res) &
            bind(c, name = 'hipblasCtpmvFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
        integer(c_int), value :: m
        type(c_ptr), value :: AP
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int) :: res
        res = hipblasCtpmv(handle, uplo, transA, diag, m,&
              AP, x, incx)
    end function hipblasCtpmvFortran

    function hipblasZtpmvFortran(handle, uplo, transA, diag, m, &
            AP, x, incx) &
            result(res) &
            bind(c, name = 'hipblasZtpmvFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
        integer(c_int), value :: m
        type(c_ptr), value :: AP
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int) :: res
        res = hipblasZtpmv(handle, uplo, transA, diag, m,&
              AP, x, incx)
    end function hipblasZtpmvFortran

    ! tpmvBatched
    function hipblasStpmvBatchedFortran(handle, uplo, transA, diag, m, &
            AP, x, incx, batch_count) &
            result(res) &
            bind(c, name = 'hipblasStpmvBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
        integer(c_int), value :: m
        type(c_ptr), value :: AP
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasStpmvBatched(handle, uplo, transA, diag, m,&
              AP, x, incx, batch_count)
    end function hipblasStpmvBatchedFortran

    function hipblasDtpmvBatchedFortran(handle, uplo, transA, diag, m, &
            AP, x, incx, batch_count) &
            result(res) &
            bind(c, name = 'hipblasDtpmvBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
        integer(c_int), value :: m
        type(c_ptr), value :: AP
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasDtpmvBatched(handle, uplo, transA, diag, m,&
              AP, x, incx, batch_count)
    end function hipblasDtpmvBatchedFortran

    function hipblasCtpmvBatchedFortran(handle, uplo, transA, diag, m, &
            AP, x, incx, batch_count) &
            result(res) &
            bind(c, name = 'hipblasCtpmvBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
        integer(c_int), value :: m
        type(c_ptr), value :: AP
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasCtpmvBatched(handle, uplo, transA, diag, m,&
              AP, x, incx, batch_count)
    end function hipblasCtpmvBatchedFortran

    function hipblasZtpmvBatchedFortran(handle, uplo, transA, diag, m, &
            AP, x, incx, batch_count) &
            result(res) &
            bind(c, name = 'hipblasZtpmvBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
        integer(c_int), value :: m
        type(c_ptr), value :: AP
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasZtpmvBatched(handle, uplo, transA, diag, m,&
              AP, x, incx, batch_count)
    end function hipblasZtpmvBatchedFortran

    ! tpmvStridedBatched
    function hipblasStpmvStridedBatchedFortran(handle, uplo, transA, diag, m, &
            AP, stride_AP, x, incx, stride_x, batch_count) &
            result(res) &
            bind(c, name = 'hipblasStpmvStridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
        integer(c_int), value :: m
        type(c_ptr), value :: AP
        integer(c_int64_t), value :: stride_AP
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasStpmvStridedBatched(handle, uplo, transA, diag, m,&
              AP, stride_AP, x, incx, stride_x, batch_count)
    end function hipblasStpmvStridedBatchedFortran

    function hipblasDtpmvStridedBatchedFortran(handle, uplo, transA, diag, m, &
            AP, stride_AP, x, incx, stride_x, batch_count) &
            result(res) &
            bind(c, name = 'hipblasDtpmvStridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
        integer(c_int), value :: m
        type(c_ptr), value :: AP
        integer(c_int64_t), value :: stride_AP
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasDtpmvStridedBatched(handle, uplo, transA, diag, m,&
              AP, stride_AP, x, incx, stride_x, batch_count)
    end function hipblasDtpmvStridedBatchedFortran

    function hipblasCtpmvStridedBatchedFortran(handle, uplo, transA, diag, m, &
            AP, stride_AP, x, incx, stride_x, batch_count) &
            result(res) &
            bind(c, name = 'hipblasCtpmvStridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
        integer(c_int), value :: m
        type(c_ptr), value :: AP
        integer(c_int64_t), value :: stride_AP
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasCtpmvStridedBatched(handle, uplo, transA, diag, m,&
              AP, stride_AP, x, incx, stride_x, batch_count)
    end function hipblasCtpmvStridedBatchedFortran

    function hipblasZtpmvStridedBatchedFortran(handle, uplo, transA, diag, m, &
            AP, stride_AP, x, incx, stride_x, batch_count) &
            result(res) &
            bind(c, name = 'hipblasZtpmvStridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
        integer(c_int), value :: m
        type(c_ptr), value :: AP
        integer(c_int64_t), value :: stride_AP
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasZtpmvStridedBatched(handle, uplo, transA, diag, m,&
              AP, stride_AP, x, incx, stride_x, batch_count)
    end function hipblasZtpmvStridedBatchedFortran

    ! tbmv
    function hipblasStbmvFortran(handle, uplo, transA, diag, m, k, &
            A, lda, x, incx) &
            result(res) &
            bind(c, name = 'hipblasStbmvFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
        integer(c_int), value :: m
        integer(c_int), value :: k
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int) :: res
        res = hipblasStbmv(handle, uplo, transA, diag, m, k,&
              A, lda, x, incx)
    end function hipblasStbmvFortran

    function hipblasDtbmvFortran(handle, uplo, transA, diag, m, k, &
            A, lda, x, incx) &
            result(res) &
            bind(c, name = 'hipblasDtbmvFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
        integer(c_int), value :: m
        integer(c_int), value :: k
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int) :: res
        res = hipblasDtbmv(handle, uplo, transA, diag, m, k,&
              A, lda, x, incx)
    end function hipblasDtbmvFortran

    function hipblasCtbmvFortran(handle, uplo, transA, diag, m, k, &
            A, lda, x, incx) &
            result(res) &
            bind(c, name = 'hipblasCtbmvFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
        integer(c_int), value :: m
        integer(c_int), value :: k
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int) :: res
        res = hipblasCtbmv(handle, uplo, transA, diag, m, k,&
              A, lda, x, incx)
    end function hipblasCtbmvFortran

    function hipblasZtbmvFortran(handle, uplo, transA, diag, m, k, &
            A, lda, x, incx) &
            result(res) &
            bind(c, name = 'hipblasZtbmvFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
        integer(c_int), value :: m
        integer(c_int), value :: k
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int) :: res
        res = hipblasZtbmv(handle, uplo, transA, diag, m, k,&
              A, lda, x, incx)
    end function hipblasZtbmvFortran

    ! tbmvBatched
    function hipblasStbmvBatchedFortran(handle, uplo, transA, diag, m, k, &
            A, lda, x, incx, batch_count) &
            result(res) &
            bind(c, name = 'hipblasStbmvBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
        integer(c_int), value :: m
        integer(c_int), value :: k
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasStbmvBatched(handle, uplo, transA, diag, m, k,&
              A, lda, x, incx, batch_count)
    end function hipblasStbmvBatchedFortran

    function hipblasDtbmvBatchedFortran(handle, uplo, transA, diag, m, k, &
            A, lda, x, incx, batch_count) &
            result(res) &
            bind(c, name = 'hipblasDtbmvBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
        integer(c_int), value :: m
        integer(c_int), value :: k
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasDtbmvBatched(handle, uplo, transA, diag, m, k,&
              A, lda, x, incx, batch_count)
    end function hipblasDtbmvBatchedFortran

    function hipblasCtbmvBatchedFortran(handle, uplo, transA, diag, m, k, &
            A, lda, x, incx, batch_count) &
            result(res) &
            bind(c, name = 'hipblasCtbmvBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
        integer(c_int), value :: m
        integer(c_int), value :: k
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasCtbmvBatched(handle, uplo, transA, diag, m, k,&
              A, lda, x, incx, batch_count)
    end function hipblasCtbmvBatchedFortran

    function hipblasZtbmvBatchedFortran(handle, uplo, transA, diag, m, k, &
            A, lda, x, incx, batch_count) &
            result(res) &
            bind(c, name = 'hipblasZtbmvBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
        integer(c_int), value :: m
        integer(c_int), value :: k
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasZtbmvBatched(handle, uplo, transA, diag, m, k,&
              A, lda, x, incx, batch_count)
    end function hipblasZtbmvBatchedFortran

    ! tbmvStridedBatched
    function hipblasStbmvStridedBatchedFortran(handle, uplo, transA, diag, m, k, &
            A, lda, stride_A, x, incx, stride_x, batch_count) &
            result(res) &
            bind(c, name = 'hipblasStbmvStridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
        integer(c_int), value :: m
        integer(c_int), value :: k
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int64_t), value :: stride_A
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasStbmvStridedBatched(handle, uplo, transA, diag, m, k,&
              A, lda, stride_A, x, incx, stride_x, batch_count)
    end function hipblasStbmvStridedBatchedFortran

    function hipblasDtbmvStridedBatchedFortran(handle, uplo, transA, diag, m, k, &
            A, lda, stride_A, x, incx, stride_x, batch_count) &
            result(res) &
            bind(c, name = 'hipblasDtbmvStridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
        integer(c_int), value :: m
        integer(c_int), value :: k
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int64_t), value :: stride_A
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasDtbmvStridedBatched(handle, uplo, transA, diag, m, k,&
              A, lda, stride_A, x, incx, stride_x, batch_count)
    end function hipblasDtbmvStridedBatchedFortran

    function hipblasCtbmvStridedBatchedFortran(handle, uplo, transA, diag, m, k, &
            A, lda, stride_A, x, incx, stride_x, batch_count) &
            result(res) &
            bind(c, name = 'hipblasCtbmvStridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
        integer(c_int), value :: m
        integer(c_int), value :: k
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int64_t), value :: stride_A
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasCtbmvStridedBatched(handle, uplo, transA, diag, m, k,&
              A, lda, stride_A, x, incx, stride_x, batch_count)
    end function hipblasCtbmvStridedBatchedFortran

    function hipblasZtbmvStridedBatchedFortran(handle, uplo, transA, diag, m, k, &
            A, lda, stride_A, x, incx, stride_x, batch_count) &
            result(res) &
            bind(c, name = 'hipblasZtbmvStridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
        integer(c_int), value :: m
        integer(c_int), value :: k
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int64_t), value :: stride_A
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasZtbmvStridedBatched(handle, uplo, transA, diag, m, k,&
              A, lda, stride_A, x, incx, stride_x, batch_count)
    end function hipblasZtbmvStridedBatchedFortran

    ! tbsv
    function hipblasStbsvFortran(handle, uplo, transA, diag, n, k, &
            A, lda, x, incx) &
            result(res) &
            bind(c, name = 'hipblasStbsvFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
        integer(c_int), value :: n
        integer(c_int), value :: k
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int) :: res
        res = hipblasStbsv(handle, uplo, transA, diag, n, k,&
              A, lda, x, incx)
    end function hipblasStbsvFortran

    function hipblasDtbsvFortran(handle, uplo, transA, diag, n, k, &
            A, lda, x, incx) &
            result(res) &
            bind(c, name = 'hipblasDtbsvFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
        integer(c_int), value :: n
        integer(c_int), value :: k
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int) :: res
        res = hipblasDtbsv(handle, uplo, transA, diag, n, k,&
              A, lda, x, incx)
    end function hipblasDtbsvFortran

    function hipblasCtbsvFortran(handle, uplo, transA, diag, n, k, &
            A, lda, x, incx) &
            result(res) &
            bind(c, name = 'hipblasCtbsvFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
        integer(c_int), value :: n
        integer(c_int), value :: k
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int) :: res
        res = hipblasCtbsv(handle, uplo, transA, diag, n, k,&
              A, lda, x, incx)
    end function hipblasCtbsvFortran

    function hipblasZtbsvFortran(handle, uplo, transA, diag, n, k, &
            A, lda, x, incx) &
            result(res) &
            bind(c, name = 'hipblasZtbsvFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
        integer(c_int), value :: n
        integer(c_int), value :: k
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int) :: res
        res = hipblasZtbsv(handle, uplo, transA, diag, n, k,&
              A, lda, x, incx)
    end function hipblasZtbsvFortran

    ! tbsvBatched
    function hipblasStbsvBatchedFortran(handle, uplo, transA, diag, n, k, &
            A, lda, x, incx, batch_count) &
            result(res) &
            bind(c, name = 'hipblasStbsvBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
        integer(c_int), value :: n
        integer(c_int), value :: k
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasStbsvBatched(handle, uplo, transA, diag, n, k,&
              A, lda, x, incx, batch_count)
    end function hipblasStbsvBatchedFortran

    function hipblasDtbsvBatchedFortran(handle, uplo, transA, diag, n, k, &
            A, lda, x, incx, batch_count) &
            result(res) &
            bind(c, name = 'hipblasDtbsvBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
        integer(c_int), value :: n
        integer(c_int), value :: k
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasDtbsvBatched(handle, uplo, transA, diag, n, k,&
              A, lda, x, incx, batch_count)
    end function hipblasDtbsvBatchedFortran

    function hipblasCtbsvBatchedFortran(handle, uplo, transA, diag, n, k, &
            A, lda, x, incx, batch_count) &
            result(res) &
            bind(c, name = 'hipblasCtbsvBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
        integer(c_int), value :: n
        integer(c_int), value :: k
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasCtbsvBatched(handle, uplo, transA, diag, n, k,&
              A, lda, x, incx, batch_count)
    end function hipblasCtbsvBatchedFortran

    function hipblasZtbsvBatchedFortran(handle, uplo, transA, diag, n, k, &
            A, lda, x, incx, batch_count) &
            result(res) &
            bind(c, name = 'hipblasZtbsvBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
        integer(c_int), value :: n
        integer(c_int), value :: k
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasZtbsvBatched(handle, uplo, transA, diag, n, k,&
              A, lda, x, incx, batch_count)
    end function hipblasZtbsvBatchedFortran

    ! tbsvStridedBatched
    function hipblasStbsvStridedBatchedFortran(handle, uplo, transA, diag, n, k, &
            A, lda, stride_A, x, incx, stride_x, batch_count) &
            result(res) &
            bind(c, name = 'hipblasStbsvStridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
        integer(c_int), value :: n
        integer(c_int), value :: k
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int64_t), value :: stride_A
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasStbsvStridedBatched(handle, uplo, transA, diag, n, k,&
              A, lda, stride_A, x, incx, stride_x, batch_count)
    end function hipblasStbsvStridedBatchedFortran

    function hipblasDtbsvStridedBatchedFortran(handle, uplo, transA, diag, n, k, &
            A, lda, stride_A, x, incx, stride_x, batch_count) &
            result(res) &
            bind(c, name = 'hipblasDtbsvStridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
        integer(c_int), value :: n
        integer(c_int), value :: k
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int64_t), value :: stride_A
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasDtbsvStridedBatched(handle, uplo, transA, diag, n, k,&
              A, lda, stride_A, x, incx, stride_x, batch_count)
    end function hipblasDtbsvStridedBatchedFortran

    function hipblasCtbsvStridedBatchedFortran(handle, uplo, transA, diag, n, k, &
            A, lda, stride_A, x, incx, stride_x, batch_count) &
            result(res) &
            bind(c, name = 'hipblasCtbsvStridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
        integer(c_int), value :: n
        integer(c_int), value :: k
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int64_t), value :: stride_A
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasCtbsvStridedBatched(handle, uplo, transA, diag, n, k,&
              A, lda, stride_A, x, incx, stride_x, batch_count)
    end function hipblasCtbsvStridedBatchedFortran

    function hipblasZtbsvStridedBatchedFortran(handle, uplo, transA, diag, n, k, &
            A, lda, stride_A, x, incx, stride_x, batch_count) &
            result(res) &
            bind(c, name = 'hipblasZtbsvStridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
        integer(c_int), value :: n
        integer(c_int), value :: k
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int64_t), value :: stride_A
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasZtbsvStridedBatched(handle, uplo, transA, diag, n, k,&
              A, lda, stride_A, x, incx, stride_x, batch_count)
    end function hipblasZtbsvStridedBatchedFortran

    ! tpsv
    function hipblasStpsvFortran(handle, uplo, transA, diag, n, &
            AP, x, incx) &
            result(res) &
            bind(c, name = 'hipblasStpsvFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
        integer(c_int), value :: n
        type(c_ptr), value :: AP
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int) :: res
        res = hipblasStpsv(handle, uplo, transA, diag, n,&
              AP, x, incx)
    end function hipblasStpsvFortran

    function hipblasDtpsvFortran(handle, uplo, transA, diag, n, &
            AP, x, incx) &
            result(res) &
            bind(c, name = 'hipblasDtpsvFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
        integer(c_int), value :: n
        type(c_ptr), value :: AP
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int) :: res
        res = hipblasDtpsv(handle, uplo, transA, diag, n,&
              AP, x, incx)
    end function hipblasDtpsvFortran

    function hipblasCtpsvFortran(handle, uplo, transA, diag, n, &
            AP, x, incx) &
            result(res) &
            bind(c, name = 'hipblasCtpsvFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
        integer(c_int), value :: n
        type(c_ptr), value :: AP
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int) :: res
        res = hipblasCtpsv(handle, uplo, transA, diag, n,&
              AP, x, incx)
    end function hipblasCtpsvFortran

    function hipblasZtpsvFortran(handle, uplo, transA, diag, n, &
            AP, x, incx) &
            result(res) &
            bind(c, name = 'hipblasZtpsvFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
        integer(c_int), value :: n
        type(c_ptr), value :: AP
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int) :: res
        res = hipblasZtpsv(handle, uplo, transA, diag, n,&
              AP, x, incx)
    end function hipblasZtpsvFortran

    ! tpsvBatched
    function hipblasStpsvBatchedFortran(handle, uplo, transA, diag, n, &
            AP, x, incx, batch_count) &
            result(res) &
            bind(c, name = 'hipblasStpsvBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
        integer(c_int), value :: n
        type(c_ptr), value :: AP
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasStpsvBatched(handle, uplo, transA, diag, n,&
              AP, x, incx, batch_count)
    end function hipblasStpsvBatchedFortran

    function hipblasDtpsvBatchedFortran(handle, uplo, transA, diag, n, &
            AP, x, incx, batch_count) &
            result(res) &
            bind(c, name = 'hipblasDtpsvBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
        integer(c_int), value :: n
        type(c_ptr), value :: AP
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasDtpsvBatched(handle, uplo, transA, diag, n,&
              AP, x, incx, batch_count)
    end function hipblasDtpsvBatchedFortran

    function hipblasCtpsvBatchedFortran(handle, uplo, transA, diag, n, &
            AP, x, incx, batch_count) &
            result(res) &
            bind(c, name = 'hipblasCtpsvBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
        integer(c_int), value :: n
        type(c_ptr), value :: AP
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasCtpsvBatched(handle, uplo, transA, diag, n,&
              AP, x, incx, batch_count)
    end function hipblasCtpsvBatchedFortran

    function hipblasZtpsvBatchedFortran(handle, uplo, transA, diag, n, &
            AP, x, incx, batch_count) &
            result(res) &
            bind(c, name = 'hipblasZtpsvBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
        integer(c_int), value :: n
        type(c_ptr), value :: AP
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasZtpsvBatched(handle, uplo, transA, diag, n,&
              AP, x, incx, batch_count)
    end function hipblasZtpsvBatchedFortran

    ! tpsvStridedBatched
    function hipblasStpsvStridedBatchedFortran(handle, uplo, transA, diag, n, &
            AP, stride_AP, x, incx, stride_x, batch_count) &
            result(res) &
            bind(c, name = 'hipblasStpsvStridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
        integer(c_int), value :: n
        type(c_ptr), value :: AP
        integer(c_int64_t), value :: stride_AP
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasStpsvStridedBatched(handle, uplo, transA, diag, n,&
              AP, stride_AP, x, incx, stride_x, batch_count)
    end function hipblasStpsvStridedBatchedFortran

    function hipblasDtpsvStridedBatchedFortran(handle, uplo, transA, diag, n, &
            AP, stride_AP, x, incx, stride_x, batch_count) &
            result(res) &
            bind(c, name = 'hipblasDtpsvStridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
        integer(c_int), value :: n
        type(c_ptr), value :: AP
        integer(c_int64_t), value :: stride_AP
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasDtpsvStridedBatched(handle, uplo, transA, diag, n,&
              AP, stride_AP, x, incx, stride_x, batch_count)
    end function hipblasDtpsvStridedBatchedFortran

    function hipblasCtpsvStridedBatchedFortran(handle, uplo, transA, diag, n, &
            AP, stride_AP, x, incx, stride_x, batch_count) &
            result(res) &
            bind(c, name = 'hipblasCtpsvStridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
        integer(c_int), value :: n
        type(c_ptr), value :: AP
        integer(c_int64_t), value :: stride_AP
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasCtpsvStridedBatched(handle, uplo, transA, diag, n,&
              AP, stride_AP, x, incx, stride_x, batch_count)
    end function hipblasCtpsvStridedBatchedFortran

    function hipblasZtpsvStridedBatchedFortran(handle, uplo, transA, diag, n, &
            AP, stride_AP, x, incx, stride_x, batch_count) &
            result(res) &
            bind(c, name = 'hipblasZtpsvStridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
        integer(c_int), value :: n
        type(c_ptr), value :: AP
        integer(c_int64_t), value :: stride_AP
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasZtpsvStridedBatched(handle, uplo, transA, diag, n,&
              AP, stride_AP, x, incx, stride_x, batch_count)
    end function hipblasZtpsvStridedBatchedFortran

    ! symv
    function hipblasSsymvFortran(handle, uplo, n, alpha, A, lda, &
            x, incx, beta, y, incy) &
            result(res) &
            bind(c, name = 'hipblasSsymvFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: beta
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int) :: res
        res = hipblasSsymv(handle, uplo, n, alpha,&
              A, lda, x, incx, beta, y, incy)
    end function hipblasSsymvFortran

    function hipblasDsymvFortran(handle, uplo, n, alpha, A, lda, &
            x, incx, beta, y, incy) &
            result(res) &
            bind(c, name = 'hipblasDsymvFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: beta
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int) :: res
        res = hipblasDsymv(handle, uplo, n, alpha,&
              A, lda, x, incx, beta, y, incy)
    end function hipblasDsymvFortran

    function hipblasCsymvFortran(handle, uplo, n, alpha, A, lda, &
            x, incx, beta, y, incy) &
            result(res) &
            bind(c, name = 'hipblasCsymvFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: beta
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int) :: res
        res = hipblasCsymv(handle, uplo, n, alpha,&
              A, lda, x, incx, beta, y, incy)
    end function hipblasCsymvFortran

    function hipblasZsymvFortran(handle, uplo, n, alpha, A, lda, &
            x, incx, beta, y, incy) &
            result(res) &
            bind(c, name = 'hipblasZsymvFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: beta
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int) :: res
        res = hipblasZsymv(handle, uplo, n, alpha,&
              A, lda, x, incx, beta, y, incy)
    end function hipblasZsymvFortran

    ! symvBatched
    function hipblasSsymvBatchedFortran(handle, uplo, n, alpha, A, lda, &
            x, incx, beta, y, incy, batch_count) &
            result(res) &
            bind(c, name = 'hipblasSsymvBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: beta
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasSsymvBatched(handle, uplo, n, alpha,&
              A, lda, x, incx, beta, y, incy, batch_count)
    end function hipblasSsymvBatchedFortran

    function hipblasDsymvBatchedFortran(handle, uplo, n, alpha, A, lda, &
            x, incx, beta, y, incy, batch_count) &
            result(res) &
            bind(c, name = 'hipblasDsymvBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: beta
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasDsymvBatched(handle, uplo, n, alpha,&
              A, lda, x, incx, beta, y, incy, batch_count)
    end function hipblasDsymvBatchedFortran

    function hipblasCsymvBatchedFortran(handle, uplo, n, alpha, A, lda, &
            x, incx, beta, y, incy, batch_count) &
            result(res) &
            bind(c, name = 'hipblasCsymvBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: beta
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasCsymvBatched(handle, uplo, n, alpha,&
              A, lda, x, incx, beta, y, incy, batch_count)
    end function hipblasCsymvBatchedFortran

    function hipblasZsymvBatchedFortran(handle, uplo, n, alpha, A, lda, &
            x, incx, beta, y, incy, batch_count) &
            result(res) &
            bind(c, name = 'hipblasZsymvBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: beta
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasZsymvBatched(handle, uplo, n, alpha,&
              A, lda, x, incx, beta, y, incy, batch_count)
    end function hipblasZsymvBatchedFortran

    ! symvStridedBatched
    function hipblasSsymvStridedBatchedFortran(handle, uplo, n, alpha, A, lda, stride_A, &
            x, incx, stride_x, beta, y, incy, stride_y, batch_count) &
            result(res) &
            bind(c, name = 'hipblasSsymvStridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int64_t), value :: stride_A
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        type(c_ptr), value :: beta
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int64_t), value :: stride_y
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasSsymvStridedBatched(handle, uplo, n, alpha,&
              A, lda, stride_A, x, incx, stride_x, beta, y, incy, stride_y, batch_count)
    end function hipblasSsymvStridedBatchedFortran

    function hipblasDsymvStridedBatchedFortran(handle, uplo, n, alpha, A, lda, stride_A, &
            x, incx, stride_x, beta, y, incy, stride_y, batch_count) &
            result(res) &
            bind(c, name = 'hipblasDsymvStridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int64_t), value :: stride_A
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        type(c_ptr), value :: beta
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int64_t), value :: stride_y
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasDsymvStridedBatched(handle, uplo, n, alpha,&
              A, lda, stride_A, x, incx, stride_x, beta, y, incy, stride_y, batch_count)
    end function hipblasDsymvStridedBatchedFortran

    function hipblasCsymvStridedBatchedFortran(handle, uplo, n, alpha, A, lda, stride_A, &
            x, incx, stride_x, beta, y, incy, stride_y, batch_count) &
            result(res) &
            bind(c, name = 'hipblasCsymvStridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int64_t), value :: stride_A
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        type(c_ptr), value :: beta
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int64_t), value :: stride_y
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasCsymvStridedBatched(handle, uplo, n, alpha,&
              A, lda, stride_A, x, incx, stride_x, beta, y, incy, stride_y, batch_count)
    end function hipblasCsymvStridedBatchedFortran

    function hipblasZsymvStridedBatchedFortran(handle, uplo, n, alpha, A, lda, stride_A, &
            x, incx, stride_x, beta, y, incy, stride_y, batch_count) &
            result(res) &
            bind(c, name = 'hipblasZsymvStridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int64_t), value :: stride_A
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        type(c_ptr), value :: beta
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int64_t), value :: stride_y
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasZsymvStridedBatched(handle, uplo, n, alpha,&
              A, lda, stride_A, x, incx, stride_x, beta, y, incy, stride_y, batch_count)
    end function hipblasZsymvStridedBatchedFortran

    ! spmv
    function hipblasSspmvFortran(handle, uplo, n, alpha, AP, &
            x, incx, beta, y, incy) &
            result(res) &
            bind(c, name = 'hipblasSspmvFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: AP
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: beta
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int) :: res
        res = hipblasSspmv(handle, uplo, n, alpha,&
              AP, x, incx, beta, y, incy)
    end function hipblasSspmvFortran

    function hipblasDspmvFortran(handle, uplo, n, alpha, AP, &
            x, incx, beta, y, incy) &
            result(res) &
            bind(c, name = 'hipblasDspmvFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: AP
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: beta
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int) :: res
        res = hipblasDspmv(handle, uplo, n, alpha,&
              AP, x, incx, beta, y, incy)
    end function hipblasDspmvFortran

    ! spmvBatched
    function hipblasSspmvBatchedFortran(handle, uplo, n, alpha, AP, &
            x, incx, beta, y, incy, batch_count) &
            result(res) &
            bind(c, name = 'hipblasSspmvBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: AP
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: beta
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasSspmvBatched(handle, uplo, n, alpha,&
              AP, x, incx, beta, y, incy, batch_count)
    end function hipblasSspmvBatchedFortran

    function hipblasDspmvBatchedFortran(handle, uplo, n, alpha, AP, &
            x, incx, beta, y, incy, batch_count) &
            result(res) &
            bind(c, name = 'hipblasDspmvBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: AP
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: beta
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasDspmvBatched(handle, uplo, n, alpha,&
              AP, x, incx, beta, y, incy, batch_count)
    end function hipblasDspmvBatchedFortran

    ! spmvStridedBatched
    function hipblasSspmvStridedBatchedFortran(handle, uplo, n, alpha, AP, stride_AP, &
            x, incx, stride_x, beta, y, incy, stride_y, batch_count) &
            result(res) &
            bind(c, name = 'hipblasSspmvStridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: AP
        integer(c_int64_t), value :: stride_AP
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        type(c_ptr), value :: beta
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int64_t), value :: stride_y
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasSspmvStridedBatched(handle, uplo, n, alpha,&
              AP, stride_AP, x, incx, stride_x, beta, y, incy, stride_y, batch_count)
    end function hipblasSspmvStridedBatchedFortran

    function hipblasDspmvStridedBatchedFortran(handle, uplo, n, alpha, AP, stride_AP, &
            x, incx, stride_x, beta, y, incy, stride_y, batch_count) &
            result(res) &
            bind(c, name = 'hipblasDspmvStridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: AP
        integer(c_int64_t), value :: stride_AP
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        type(c_ptr), value :: beta
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int64_t), value :: stride_y
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasDspmvStridedBatched(handle, uplo, n, alpha,&
              AP, stride_AP, x, incx, stride_x, beta, y, incy, stride_y, batch_count)
    end function hipblasDspmvStridedBatchedFortran

    ! sbmv
    function hipblasSsbmvFortran(handle, uplo, n, k, alpha, A, lda, &
            x, incx, beta, y, incy) &
            result(res) &
            bind(c, name = 'hipblasSsbmvFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        integer(c_int), value :: k
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: beta
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int) :: res
        res = hipblasSsbmv(handle, uplo, n, k, alpha,&
              A, lda, x, incx, beta, y, incy)
    end function hipblasSsbmvFortran

    function hipblasDsbmvFortran(handle, uplo, n, k, alpha, A, lda, &
            x, incx, beta, y, incy) &
            result(res) &
            bind(c, name = 'hipblasDsbmvFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        integer(c_int), value :: k
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: beta
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int) :: res
        res = hipblasDsbmv(handle, uplo, n, k, alpha,&
              A, lda, x, incx, beta, y, incy)
    end function hipblasDsbmvFortran

    ! sbmvBatched
    function hipblasSsbmvBatchedFortran(handle, uplo, n, k, alpha, A, lda, &
            x, incx, beta, y, incy, batch_count) &
            result(res) &
            bind(c, name = 'hipblasSsbmvBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        integer(c_int), value :: k
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: beta
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasSsbmvBatched(handle, uplo, n, k, alpha,&
              A, lda, x, incx, beta, y, incy, batch_count)
    end function hipblasSsbmvBatchedFortran

    function hipblasDsbmvBatchedFortran(handle, uplo, n, k, alpha, A, lda, &
            x, incx, beta, y, incy, batch_count) &
            result(res) &
            bind(c, name = 'hipblasDsbmvBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        integer(c_int), value :: k
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: beta
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasDsbmvBatched(handle, uplo, n, k, alpha,&
              A, lda, x, incx, beta, y, incy, batch_count)
    end function hipblasDsbmvBatchedFortran

    ! sbmvStridedBatched
    function hipblasSsbmvStridedBatchedFortran(handle, uplo, n, k, alpha, A, lda, stride_A, &
            x, incx, stride_x, beta, y, incy, stride_y, batch_count) &
            result(res) &
            bind(c, name = 'hipblasSsbmvStridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        integer(c_int), value :: k
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int64_t), value :: stride_A
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        type(c_ptr), value :: beta
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int64_t), value :: stride_y
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasSsbmvStridedBatched(handle, uplo, n, k, alpha,&
              A, lda, stride_A, x, incx, stride_x, beta, y, incy, stride_y, batch_count)
    end function hipblasSsbmvStridedBatchedFortran

    function hipblasDsbmvStridedBatchedFortran(handle, uplo, n, k, alpha, A, lda, stride_A, &
            x, incx, stride_x, beta, y, incy, stride_y, batch_count) &
            result(res) &
            bind(c, name = 'hipblasDsbmvStridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        integer(c_int), value :: k
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int64_t), value :: stride_A
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        type(c_ptr), value :: beta
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int64_t), value :: stride_y
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasDsbmvStridedBatched(handle, uplo, n, k, alpha,&
              A, lda, stride_A, x, incx, stride_x, beta, y, incy, stride_y, batch_count)
    end function hipblasDsbmvStridedBatchedFortran

    ! ger
    function hipblasSgerFortran(handle, m, n, alpha, x, incx, &
            y, incy, A, lda) &
            result(res) &
            bind(c, name = 'hipblasSgerFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: m
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int) :: res
        res = hipblasSger(handle, m, n, alpha,&
              x, incx, y, incy, A, lda)
    end function hipblasSgerFortran

    function hipblasDgerFortran(handle, m, n, alpha, x, incx, &
            y, incy, A, lda) &
            result(res) &
            bind(c, name = 'hipblasDgerFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: m
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int) :: res
        res = hipblasDger(handle, m, n, alpha,&
              x, incx, y, incy, A, lda)
    end function hipblasDgerFortran

    function hipblasCgeruFortran(handle, m, n, alpha, x, incx, &
            y, incy, A, lda) &
            result(res) &
            bind(c, name = 'hipblasCgeruFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: m
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int) :: res
        res = hipblasCgeru(handle, m, n, alpha,&
              x, incx, y, incy, A, lda)
    end function hipblasCgeruFortran

    function hipblasCgercFortran(handle, m, n, alpha, x, incx, &
            y, incy, A, lda) &
            result(res) &
            bind(c, name = 'hipblasCgercFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: m
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int) :: res
        res = hipblasCgerc(handle, m, n, alpha,&
              x, incx, y, incy, A, lda)
    end function hipblasCgercFortran

    function hipblasZgeruFortran(handle, m, n, alpha, x, incx, &
            y, incy, A, lda) &
            result(res) &
            bind(c, name = 'hipblasZgeruFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: m
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int) :: res
        res = hipblasZgeru(handle, m, n, alpha,&
              x, incx, y, incy, A, lda)
    end function hipblasZgeruFortran

    function hipblasZgercFortran(handle, m, n, alpha, x, incx, &
            y, incy, A, lda) &
            result(res) &
            bind(c, name = 'hipblasZgercFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: m
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int) :: res
        res = hipblasZgerc(handle, m, n, alpha,&
              x, incx, y, incy, A, lda)
    end function hipblasZgercFortran

    ! gerBatched
    function hipblasSgerBatchedFortran(handle, m, n, alpha, x, incx, &
            y, incy, A, lda, batch_count) &
            result(res) &
            bind(c, name = 'hipblasSgerBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: m
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasSgerBatched(handle, m, n, alpha,&
              x, incx, y, incy, A, lda, batch_count)
    end function hipblasSgerBatchedFortran

    function hipblasDgerBatchedFortran(handle, m, n, alpha, x, incx, &
            y, incy, A, lda, batch_count) &
            result(res) &
            bind(c, name = 'hipblasDgerBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: m
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasDgerBatched(handle, m, n, alpha,&
              x, incx, y, incy, A, lda, batch_count)
    end function hipblasDgerBatchedFortran

    function hipblasCgeruBatchedFortran(handle, m, n, alpha, x, incx, &
            y, incy, A, lda, batch_count) &
            result(res) &
            bind(c, name = 'hipblasCgeruBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: m
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasCgeruBatched(handle, m, n, alpha,&
              x, incx, y, incy, A, lda, batch_count)
    end function hipblasCgeruBatchedFortran

    function hipblasCgercBatchedFortran(handle, m, n, alpha, x, incx, &
            y, incy, A, lda, batch_count) &
            result(res) &
            bind(c, name = 'hipblasCgercBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: m
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasCgercBatched(handle, m, n, alpha,&
              x, incx, y, incy, A, lda, batch_count)
    end function hipblasCgercBatchedFortran

    function hipblasZgeruBatchedFortran(handle, m, n, alpha, x, incx, &
            y, incy, A, lda, batch_count) &
            result(res) &
            bind(c, name = 'hipblasZgeruBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: m
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasZgeruBatched(handle, m, n, alpha,&
              x, incx, y, incy, A, lda, batch_count)
    end function hipblasZgeruBatchedFortran

    function hipblasZgercBatchedFortran(handle, m, n, alpha, x, incx, &
            y, incy, A, lda, batch_count) &
            result(res) &
            bind(c, name = 'hipblasZgercBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: m
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasZgercBatched(handle, m, n, alpha,&
              x, incx, y, incy, A, lda, batch_count)
    end function hipblasZgercBatchedFortran

    ! gerStridedBatched
    function hipblasSgerStridedBatchedFortran(handle, m, n, alpha, x, incx, stride_x, &
            y, incy, stride_y, A, lda, stride_A, batch_count) &
            result(res) &
            bind(c, name = 'hipblasSgerStridedBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: m
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int64_t), value :: stride_y
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int64_t), value :: stride_A
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasSgerStridedBatched(handle, m, n, alpha,&
              x, incx, stride_x, y, incy, stride_y, A, lda, stride_A, batch_count)
    end function hipblasSgerStridedBatchedFortran

    function hipblasDgerStridedBatchedFortran(handle, m, n, alpha, x, incx, stride_x, &
            y, incy, stride_y, A, lda, stride_A, batch_count) &
            result(res) &
            bind(c, name = 'hipblasDgerStridedBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: m
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int64_t), value :: stride_y
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int64_t), value :: stride_A
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasDgerStridedBatched(handle, m, n, alpha,&
              x, incx, stride_x, y, incy, stride_y, A, lda, stride_A, batch_count)
    end function hipblasDgerStridedBatchedFortran

    function hipblasCgeruStridedBatchedFortran(handle, m, n, alpha, x, incx, stride_x, &
            y, incy, stride_y, A, lda, stride_A, batch_count) &
            result(res) &
            bind(c, name = 'hipblasCgeruStridedBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: m
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int64_t), value :: stride_y
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int64_t), value :: stride_A
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasCgeruStridedBatched(handle, m, n, alpha,&
              x, incx, stride_x, y, incy, stride_y, A, lda, stride_A, batch_count)
    end function hipblasCgeruStridedBatchedFortran

    function hipblasCgercStridedBatchedFortran(handle, m, n, alpha, x, incx, stride_x, &
            y, incy, stride_y, A, lda, stride_A, batch_count) &
            result(res) &
            bind(c, name = 'hipblasCgercStridedBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: m
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int64_t), value :: stride_y
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int64_t), value :: stride_A
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasCgercStridedBatched(handle, m, n, alpha,&
              x, incx, stride_x, y, incy, stride_y, A, lda, stride_A, batch_count)
    end function hipblasCgercStridedBatchedFortran

    function hipblasZgeruStridedBatchedFortran(handle, m, n, alpha, x, incx, stride_x, &
            y, incy, stride_y, A, lda, stride_A, batch_count) &
            result(res) &
            bind(c, name = 'hipblasZgeruStridedBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: m
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int64_t), value :: stride_y
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int64_t), value :: stride_A
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasZgeruStridedBatched(handle, m, n, alpha,&
              x, incx, stride_x, y, incy, stride_y, A, lda, stride_A, batch_count)
    end function hipblasZgeruStridedBatchedFortran

    function hipblasZgercStridedBatchedFortran(handle, m, n, alpha, x, incx, stride_x, &
            y, incy, stride_y, A, lda, stride_A, batch_count) &
            result(res) &
            bind(c, name = 'hipblasZgercStridedBatchedFortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: m
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int64_t), value :: stride_y
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int64_t), value :: stride_A
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasZgercStridedBatched(handle, m, n, alpha,&
              x, incx, stride_x, y, incy, stride_y, A, lda, stride_A, batch_count)
    end function hipblasZgercStridedBatchedFortran

    ! spr
    function hipblasSsprFortran(handle, uplo, n, alpha, x, incx, AP) &
            result(res) &
            bind(c, name = 'hipblasSsprFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: AP
        integer(c_int) :: res
        res = hipblasSspr(handle, uplo, n, alpha,&
              x, incx, AP)
    end function hipblasSsprFortran

    function hipblasDsprFortran(handle, uplo, n, alpha, x, incx, AP) &
            result(res) &
            bind(c, name = 'hipblasDsprFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: AP
        integer(c_int) :: res
        res = hipblasDspr(handle, uplo, n, alpha,&
              x, incx, AP)
    end function hipblasDsprFortran

    function hipblasCsprFortran(handle, uplo, n, alpha, x, incx, AP) &
            result(res) &
            bind(c, name = 'hipblasCsprFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: AP
        integer(c_int) :: res
        res = hipblasCspr(handle, uplo, n, alpha,&
              x, incx, AP)
    end function hipblasCsprFortran

    function hipblasZsprFortran(handle, uplo, n, alpha, x, incx, AP) &
            result(res) &
            bind(c, name = 'hipblasZsprFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: AP
        integer(c_int) :: res
        res = hipblasZspr(handle, uplo, n, alpha,&
              x, incx, AP)
    end function hipblasZsprFortran

    ! sprBatched
    function hipblasSsprBatchedFortran(handle, uplo, n, alpha, x, incx, AP, batch_count) &
            result(res) &
            bind(c, name = 'hipblasSsprBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: AP
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasSsprBatched(handle, uplo, n, alpha,&
              x, incx, AP, batch_count)
    end function hipblasSsprBatchedFortran

    function hipblasDsprBatchedFortran(handle, uplo, n, alpha, x, incx, AP, batch_count) &
            result(res) &
            bind(c, name = 'hipblasDsprBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: AP
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasDsprBatched(handle, uplo, n, alpha,&
              x, incx, AP, batch_count)
    end function hipblasDsprBatchedFortran

    function hipblasCsprBatchedFortran(handle, uplo, n, alpha, x, incx, AP, batch_count) &
            result(res) &
            bind(c, name = 'hipblasCsprBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: AP
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasCsprBatched(handle, uplo, n, alpha,&
              x, incx, AP, batch_count)
    end function hipblasCsprBatchedFortran

    function hipblasZsprBatchedFortran(handle, uplo, n, alpha, x, incx, AP, batch_count) &
            result(res) &
            bind(c, name = 'hipblasZsprBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: AP
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasZsprBatched(handle, uplo, n, alpha,&
              x, incx, AP, batch_count)
    end function hipblasZsprBatchedFortran

    ! sprStridedBatched
    function hipblasSsprStridedBatchedFortran(handle, uplo, n, alpha, x, incx, stride_x, &
            AP, stride_AP, batch_count) &
            result(res) &
            bind(c, name = 'hipblasSsprStridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        type(c_ptr), value :: AP
        integer(c_int64_t), value :: stride_AP
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasSsprStridedBatched(handle, uplo, n, alpha,&
              x, incx, stride_x, AP, stride_AP, batch_count)
    end function hipblasSsprStridedBatchedFortran

    function hipblasDsprStridedBatchedFortran(handle, uplo, n, alpha, x, incx, stride_x, &
            AP, stride_AP, batch_count) &
            result(res) &
            bind(c, name = 'hipblasDsprStridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        type(c_ptr), value :: AP
        integer(c_int64_t), value :: stride_AP
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasDsprStridedBatched(handle, uplo, n, alpha,&
              x, incx, stride_x, AP, stride_AP, batch_count)
    end function hipblasDsprStridedBatchedFortran

    function hipblasCsprStridedBatchedFortran(handle, uplo, n, alpha, x, incx, stride_x, &
            AP, stride_AP, batch_count) &
            result(res) &
            bind(c, name = 'hipblasCsprStridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        type(c_ptr), value :: AP
        integer(c_int64_t), value :: stride_AP
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasCsprStridedBatched(handle, uplo, n, alpha,&
              x, incx, stride_x, AP, stride_AP, batch_count)
    end function hipblasCsprStridedBatchedFortran

    function hipblasZsprStridedBatchedFortran(handle, uplo, n, alpha, x, incx, stride_x, &
            AP, stride_AP, batch_count) &
            result(res) &
            bind(c, name = 'hipblasZsprStridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        type(c_ptr), value :: AP
        integer(c_int64_t), value :: stride_AP
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasZsprStridedBatched(handle, uplo, n, alpha,&
              x, incx, stride_x, AP, stride_AP, batch_count)
    end function hipblasZsprStridedBatchedFortran

    ! spr2
    function hipblasSspr2Fortran(handle, uplo, n, alpha, x, incx, &
            y, incy, AP) &
            result(res) &
            bind(c, name = 'hipblasSspr2Fortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        type(c_ptr), value :: AP
        integer(c_int) :: res
        res = hipblasSspr2(handle, uplo, n, alpha,&
              x, incx, y, incy, AP)
    end function hipblasSspr2Fortran

    function hipblasDspr2Fortran(handle, uplo, n, alpha, x, incx, &
            y, incy, AP) &
            result(res) &
            bind(c, name = 'hipblasDspr2Fortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        type(c_ptr), value :: AP
        integer(c_int) :: res
        res = hipblasDspr2(handle, uplo, n, alpha,&
              x, incx, y, incy, AP)
    end function hipblasDspr2Fortran

    ! spr2Batched
    function hipblasSspr2BatchedFortran(handle, uplo, n, alpha, x, incx, &
            y, incy, AP, batch_count) &
            result(res) &
            bind(c, name = 'hipblasSspr2BatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        type(c_ptr), value :: AP
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasSspr2Batched(handle, uplo, n, alpha,&
              x, incx, y, incy, AP, batch_count)
    end function hipblasSspr2BatchedFortran

    function hipblasDspr2BatchedFortran(handle, uplo, n, alpha, x, incx, &
            y, incy, AP, batch_count) &
            result(res) &
            bind(c, name = 'hipblasDspr2BatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        type(c_ptr), value :: AP
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasDspr2Batched(handle, uplo, n, alpha,&
              x, incx, y, incy, AP, batch_count)
    end function hipblasDspr2BatchedFortran

    ! spr2StridedBatched
    function hipblasSspr2StridedBatchedFortran(handle, uplo, n, alpha, x, incx, stride_x, &
            y, incy, stride_y, AP, stride_AP, batch_count) &
            result(res) &
            bind(c, name = 'hipblasSspr2StridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int64_t), value :: stride_y
        type(c_ptr), value :: AP
        integer(c_int64_t), value :: stride_AP
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasSspr2StridedBatched(handle, uplo, n, alpha,&
              x, incx, stride_x, y, incy, stride_y, AP, stride_AP, batch_count)
    end function hipblasSspr2StridedBatchedFortran

    function hipblasDspr2StridedBatchedFortran(handle, uplo, n, alpha, x, incx, stride_x, &
            y, incy, stride_y, AP, stride_AP, batch_count) &
            result(res) &
            bind(c, name = 'hipblasDspr2StridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int64_t), value :: stride_y
        type(c_ptr), value :: AP
        integer(c_int64_t), value :: stride_AP
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasDspr2StridedBatched(handle, uplo, n, alpha,&
              x, incx, stride_x, y, incy, stride_y, AP, stride_AP, batch_count)
    end function hipblasDspr2StridedBatchedFortran

    ! syr
    function hipblasSsyrFortran(handle, uplo, n, alpha, x, incx, A, lda) &
            result(res) &
            bind(c, name = 'hipblasSsyrFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int) :: res
        res = hipblasSsyr(handle, uplo, n, alpha,&
              x, incx, A, lda)
    end function hipblasSsyrFortran

    function hipblasDsyrFortran(handle, uplo, n, alpha, x, incx, A, lda) &
            result(res) &
            bind(c, name = 'hipblasDsyrFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int) :: res
        res = hipblasDsyr(handle, uplo, n, alpha,&
              x, incx, A, lda)
    end function hipblasDsyrFortran

    function hipblasCsyrFortran(handle, uplo, n, alpha, x, incx, A, lda) &
            result(res) &
            bind(c, name = 'hipblasCsyrFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int) :: res
        res = hipblasCsyr(handle, uplo, n, alpha,&
              x, incx, A, lda)
    end function hipblasCsyrFortran

    function hipblasZsyrFortran(handle, uplo, n, alpha, x, incx, A, lda) &
            result(res) &
            bind(c, name = 'hipblasZsyrFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int) :: res
        res = hipblasZsyr(handle, uplo, n, alpha,&
              x, incx, A, lda)
    end function hipblasZsyrFortran

    ! syrBatched
    function hipblasSsyrBatchedFortran(handle, uplo, n, alpha, x, incx, A, lda, batch_count) &
            result(res) &
            bind(c, name = 'hipblasSsyrBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasSsyrBatched(handle, uplo, n, alpha,&
              x, incx, A, lda, batch_count)
    end function hipblasSsyrBatchedFortran

    function hipblasDsyrBatchedFortran(handle, uplo, n, alpha, x, incx, A, lda, batch_count) &
            result(res) &
            bind(c, name = 'hipblasDsyrBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasDsyrBatched(handle, uplo, n, alpha,&
              x, incx, A, lda, batch_count)
    end function hipblasDsyrBatchedFortran

    function hipblasCsyrBatchedFortran(handle, uplo, n, alpha, x, incx, A, lda, batch_count) &
            result(res) &
            bind(c, name = 'hipblasCsyrBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasCsyrBatched(handle, uplo, n, alpha,&
              x, incx, A, lda, batch_count)
    end function hipblasCsyrBatchedFortran

    function hipblasZsyrBatchedFortran(handle, uplo, n, alpha, x, incx, A, lda, batch_count) &
            result(res) &
            bind(c, name = 'hipblasZsyrBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasZsyrBatched(handle, uplo, n, alpha,&
              x, incx, A, lda, batch_count)
    end function hipblasZsyrBatchedFortran

    ! syrStridedBatched
    function hipblasSsyrStridedBatchedFortran(handle, uplo, n, alpha, x, incx, stride_x, &
            A, lda, stride_A, batch_count) &
            result(res) &
            bind(c, name = 'hipblasSsyrStridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int64_t), value :: stride_A
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasSsyrStridedBatched(handle, uplo, n, alpha,&
              x, incx, stride_x, A, lda, stride_A, batch_count)
    end function hipblasSsyrStridedBatchedFortran

    function hipblasDsyrStridedBatchedFortran(handle, uplo, n, alpha, x, incx, stride_x, &
            A, lda, stride_A, batch_count) &
            result(res) &
            bind(c, name = 'hipblasDsyrStridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int64_t), value :: stride_A
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasDsyrStridedBatched(handle, uplo, n, alpha,&
              x, incx, stride_x, A, lda, stride_A, batch_count)
    end function hipblasDsyrStridedBatchedFortran

    function hipblasCsyrStridedBatchedFortran(handle, uplo, n, alpha, x, incx, stride_x, &
            A, lda, stride_A, batch_count) &
            result(res) &
            bind(c, name = 'hipblasCsyrStridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int64_t), value :: stride_A
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasCsyrStridedBatched(handle, uplo, n, alpha,&
              x, incx, stride_x, A, lda, stride_A, batch_count)
    end function hipblasCsyrStridedBatchedFortran

    function hipblasZsyrStridedBatchedFortran(handle, uplo, n, alpha, x, incx, stride_x, &
            A, lda, stride_A, batch_count) &
            result(res) &
            bind(c, name = 'hipblasZsyrStridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int64_t), value :: stride_A
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasZsyrStridedBatched(handle, uplo, n, alpha,&
              x, incx, stride_x, A, lda, stride_A, batch_count)
    end function hipblasZsyrStridedBatchedFortran

    ! syr2
    function hipblasSsyr2Fortran(handle, uplo, n, alpha, x, incx, &
            y, incy, A, lda) &
            result(res) &
            bind(c, name = 'hipblasSsyr2Fortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int) :: res
        res = hipblasSsyr2(handle, uplo, n, alpha,&
              x, incx, y, incy, A, lda)
    end function hipblasSsyr2Fortran

    function hipblasDsyr2Fortran(handle, uplo, n, alpha, x, incx, &
            y, incy, A, lda) &
            result(res) &
            bind(c, name = 'hipblasDsyr2Fortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int) :: res
        res = hipblasDsyr2(handle, uplo, n, alpha,&
              x, incx, y, incy, A, lda)
    end function hipblasDsyr2Fortran

    function hipblasCsyr2Fortran(handle, uplo, n, alpha, x, incx, &
            y, incy, A, lda) &
            result(res) &
            bind(c, name = 'hipblasCsyr2Fortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int) :: res
        res = hipblasCsyr2(handle, uplo, n, alpha,&
              x, incx, y, incy, A, lda)
    end function hipblasCsyr2Fortran

    function hipblasZsyr2Fortran(handle, uplo, n, alpha, x, incx, &
            y, incy, A, lda) &
            result(res) &
            bind(c, name = 'hipblasZsyr2Fortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int) :: res
        res = hipblasZsyr2(handle, uplo, n, alpha,&
              x, incx, y, incy, A, lda)
    end function hipblasZsyr2Fortran

    ! syr2Batched
    function hipblasSsyr2BatchedFortran(handle, uplo, n, alpha, x, incx, &
            y, incy, A, lda, batch_count) &
            result(res) &
            bind(c, name = 'hipblasSsyr2BatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasSsyr2Batched(handle, uplo, n, alpha,&
              x, incx, y, incy, A, lda, batch_count)
    end function hipblasSsyr2BatchedFortran

    function hipblasDsyr2BatchedFortran(handle, uplo, n, alpha, x, incx, &
            y, incy, A, lda, batch_count) &
            result(res) &
            bind(c, name = 'hipblasDsyr2BatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasDsyr2Batched(handle, uplo, n, alpha,&
              x, incx, y, incy, A, lda, batch_count)
    end function hipblasDsyr2BatchedFortran

    function hipblasCsyr2BatchedFortran(handle, uplo, n, alpha, x, incx, &
            y, incy, A, lda, batch_count) &
            result(res) &
            bind(c, name = 'hipblasCsyr2BatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasCsyr2Batched(handle, uplo, n, alpha,&
              x, incx, y, incy, A, lda, batch_count)
    end function hipblasCsyr2BatchedFortran

    function hipblasZsyr2BatchedFortran(handle, uplo, n, alpha, x, incx, &
            y, incy, A, lda, batch_count) &
            result(res) &
            bind(c, name = 'hipblasZsyr2BatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasZsyr2Batched(handle, uplo, n, alpha,&
              x, incx, y, incy, A, lda, batch_count)
    end function hipblasZsyr2BatchedFortran

    ! syr2StridedBatched
    function hipblasSsyr2StridedBatchedFortran(handle, uplo, n, alpha, x, incx, stride_x, &
            y, incy, stride_y, A, lda, stride_A, batch_count) &
            result(res) &
            bind(c, name = 'hipblasSsyr2StridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int64_t), value :: stride_y
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int64_t), value :: stride_A
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasSsyr2StridedBatched(handle, uplo, n, alpha,&
              x, incx, stride_x, y, incy, stride_y, A, lda, stride_A, batch_count)
    end function hipblasSsyr2StridedBatchedFortran

    function hipblasDsyr2StridedBatchedFortran(handle, uplo, n, alpha, x, incx, stride_x, &
            y, incy, stride_y, A, lda, stride_A, batch_count) &
            result(res) &
            bind(c, name = 'hipblasDsyr2StridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int64_t), value :: stride_y
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int64_t), value :: stride_A
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasDsyr2StridedBatched(handle, uplo, n, alpha,&
              x, incx, stride_x, y, incy, stride_y, A, lda, stride_A, batch_count)
    end function hipblasDsyr2StridedBatchedFortran

    function hipblasCsyr2StridedBatchedFortran(handle, uplo, n, alpha, x, incx, stride_x, &
            y, incy, stride_y, A, lda, stride_A, batch_count) &
            result(res) &
            bind(c, name = 'hipblasCsyr2StridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int64_t), value :: stride_y
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int64_t), value :: stride_A
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasCsyr2StridedBatched(handle, uplo, n, alpha,&
              x, incx, stride_x, y, incy, stride_y, A, lda, stride_A, batch_count)
    end function hipblasCsyr2StridedBatchedFortran

    function hipblasZsyr2StridedBatchedFortran(handle, uplo, n, alpha, x, incx, stride_x, &
            y, incy, stride_y, A, lda, stride_A, batch_count) &
            result(res) &
            bind(c, name = 'hipblasZsyr2StridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int64_t), value :: stride_y
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int64_t), value :: stride_A
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasZsyr2StridedBatched(handle, uplo, n, alpha,&
              x, incx, stride_x, y, incy, stride_y, A, lda, stride_A, batch_count)
    end function hipblasZsyr2StridedBatchedFortran

    ! trsv
    function hipblasStrsvFortran(handle, uplo, transA, diag, m, &
            A, lda, x, incx) &
            result(res) &
            bind(c, name = 'hipblasStrsvFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
        integer(c_int), value :: m
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int) :: res
        res = hipblasStrsv(handle, uplo, transA, diag, m,&
              A, lda, x, incx)
    end function hipblasStrsvFortran

    function hipblasDtrsvFortran(handle, uplo, transA, diag, m, &
            A, lda, x, incx) &
            result(res) &
            bind(c, name = 'hipblasDtrsvFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
        integer(c_int), value :: m
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int) :: res
        res = hipblasDtrsv(handle, uplo, transA, diag, m,&
              A, lda, x, incx)
    end function hipblasDtrsvFortran

    function hipblasCtrsvFortran(handle, uplo, transA, diag, m, &
            A, lda, x, incx) &
            result(res) &
            bind(c, name = 'hipblasCtrsvFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
        integer(c_int), value :: m
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int) :: res
        res = hipblasCtrsv(handle, uplo, transA, diag, m,&
              A, lda, x, incx)
    end function hipblasCtrsvFortran

    function hipblasZtrsvFortran(handle, uplo, transA, diag, m, &
            A, lda, x, incx) &
            result(res) &
            bind(c, name = 'hipblasZtrsvFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
        integer(c_int), value :: m
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int) :: res
        res = hipblasZtrsv(handle, uplo, transA, diag, m,&
              A, lda, x, incx)
    end function hipblasZtrsvFortran

    ! trsvBatched
    function hipblasStrsvBatchedFortran(handle, uplo, transA, diag, m, &
            A, lda, x, incx, batch_count) &
            result(res) &
            bind(c, name = 'hipblasStrsvBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
        integer(c_int), value :: m
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasStrsvBatched(handle, uplo, transA, diag, m,&
              A, lda, x, incx, batch_count)
    end function hipblasStrsvBatchedFortran

    function hipblasDtrsvBatchedFortran(handle, uplo, transA, diag, m, &
            A, lda, x, incx, batch_count) &
            result(res) &
            bind(c, name = 'hipblasDtrsvBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
        integer(c_int), value :: m
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasDtrsvBatched(handle, uplo, transA, diag, m,&
              A, lda, x, incx, batch_count)
    end function hipblasDtrsvBatchedFortran

    function hipblasCtrsvBatchedFortran(handle, uplo, transA, diag, m, &
            A, lda, x, incx, batch_count) &
            result(res) &
            bind(c, name = 'hipblasCtrsvBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
        integer(c_int), value :: m
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasCtrsvBatched(handle, uplo, transA, diag, m,&
              A, lda, x, incx, batch_count)
    end function hipblasCtrsvBatchedFortran

    function hipblasZtrsvBatchedFortran(handle, uplo, transA, diag, m, &
            A, lda, x, incx, batch_count) &
            result(res) &
            bind(c, name = 'hipblasZtrsvBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
        integer(c_int), value :: m
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasZtrsvBatched(handle, uplo, transA, diag, m,&
              A, lda, x, incx, batch_count)
    end function hipblasZtrsvBatchedFortran

    ! trsvStridedBatched
    function hipblasStrsvStridedBatchedFortran(handle, uplo, transA, diag, m, &
            A, lda, stride_A, x, incx, stride_x, batch_count) &
            result(res) &
            bind(c, name = 'hipblasStrsvStridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
        integer(c_int), value :: m
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int64_t), value :: stride_A
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasStrsvStridedBatched(handle, uplo, transA, diag, m,&
              A, lda, stride_A, x, incx, stride_x, batch_count)
    end function hipblasStrsvStridedBatchedFortran

    function hipblasDtrsvStridedBatchedFortran(handle, uplo, transA, diag, m, &
            A, lda, stride_A, x, incx, stride_x, batch_count) &
            result(res) &
            bind(c, name = 'hipblasDtrsvStridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
        integer(c_int), value :: m
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int64_t), value :: stride_A
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasDtrsvStridedBatched(handle, uplo, transA, diag, m,&
              A, lda, stride_A, x, incx, stride_x, batch_count)
    end function hipblasDtrsvStridedBatchedFortran

    function hipblasCtrsvStridedBatchedFortran(handle, uplo, transA, diag, m, &
            A, lda, stride_A, x, incx, stride_x, batch_count) &
            result(res) &
            bind(c, name = 'hipblasCtrsvStridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
        integer(c_int), value :: m
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int64_t), value :: stride_A
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasCtrsvStridedBatched(handle, uplo, transA, diag, m,&
              A, lda, stride_A, x, incx, stride_x, batch_count)
    end function hipblasCtrsvStridedBatchedFortran

    function hipblasZtrsvStridedBatchedFortran(handle, uplo, transA, diag, m, &
            A, lda, stride_A, x, incx, stride_x, batch_count) &
            result(res) &
            bind(c, name = 'hipblasZtrsvStridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
        integer(c_int), value :: m
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int64_t), value :: stride_A
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasZtrsvStridedBatched(handle, uplo, transA, diag, m,&
              A, lda, stride_A, x, incx, stride_x, batch_count)
    end function hipblasZtrsvStridedBatchedFortran

    !--------!
    ! blas 3 !
    !--------!

    ! hemm
    function hipblasChemmFortran(handle, side, uplo, n, k, alpha, &
            A, lda, B, ldb, beta, C, ldc) &
            result(res) &
            bind(c, name = 'hipblasChemmFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        integer(c_int), value :: k
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: B
        integer(c_int), value :: ldb
        type(c_ptr), value :: beta
        type(c_ptr), value :: C
        integer(c_int), value :: ldc
        integer(c_int) :: res
        res = hipblasChemm(handle, side, uplo, n, k, alpha,&
                            A, lda, B, ldb, beta, C, ldc)
    end function hipblasChemmFortran

    function hipblasZhemmFortran(handle, side, uplo, n, k, alpha, &
            A, lda, B, ldb, beta, C, ldc) &
            result(res) &
            bind(c, name = 'hipblasZhemmFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        integer(c_int), value :: k
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: B
        integer(c_int), value :: ldb
        type(c_ptr), value :: beta
        type(c_ptr), value :: C
        integer(c_int), value :: ldc
        integer(c_int) :: res
        res = hipblasZhemm(handle, side, uplo, n, k, alpha,&
                            A, lda, B, ldb, beta, C, ldc)
    end function hipblasZhemmFortran

    ! hemmBatched
    function hipblasChemmBatchedFortran(handle, side, uplo, n, k, alpha, &
            A, lda, B, ldb, beta, C, ldc, batch_count) &
            result(res) &
            bind(c, name = 'hipblasChemmBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        integer(c_int), value :: k
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: B
        integer(c_int), value :: ldb
        type(c_ptr), value :: beta
        type(c_ptr), value :: C
        integer(c_int), value :: ldc
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasChemmBatched(handle, side, uplo, n, k, alpha,&
                            A, lda, B, ldb, beta, C, ldc, batch_count)
    end function hipblasChemmBatchedFortran

    function hipblasZhemmBatchedFortran(handle, side, uplo, n, k, alpha, &
            A, lda, B, ldb, beta, C, ldc, batch_count) &
            result(res) &
            bind(c, name = 'hipblasZhemmBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        integer(c_int), value :: k
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: B
        integer(c_int), value :: ldb
        type(c_ptr), value :: beta
        type(c_ptr), value :: C
        integer(c_int), value :: ldc
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasZhemmBatched(handle, side, uplo, n, k, alpha,&
                            A, lda, B, ldb, beta, C, ldc, batch_count)
    end function hipblasZhemmBatchedFortran

    ! hemmStridedBatched
    function hipblasChemmStridedBatchedFortran(handle, side, uplo, n, k, alpha, &
            A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count) &
            result(res) &
            bind(c, name = 'hipblasChemmStridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        integer(c_int), value :: k
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int64_t), value :: stride_A
        type(c_ptr), value :: B
        integer(c_int), value :: ldb
        integer(c_int64_t), value :: stride_B
        type(c_ptr), value :: beta
        type(c_ptr), value :: C
        integer(c_int), value :: ldc
        integer(c_int64_t), value :: stride_C
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasChemmStridedBatched(handle, side, uplo, n, k, alpha,&
                            A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count)
    end function hipblasChemmStridedBatchedFortran

    function hipblasZhemmStridedBatchedFortran(handle, side, uplo, n, k, alpha, &
            A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count) &
            result(res) &
            bind(c, name = 'hipblasZhemmStridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: n
        integer(c_int), value :: k
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int64_t), value :: stride_A
        type(c_ptr), value :: B
        integer(c_int), value :: ldb
        integer(c_int64_t), value :: stride_B
        type(c_ptr), value :: beta
        type(c_ptr), value :: C
        integer(c_int), value :: ldc
        integer(c_int64_t), value :: stride_C
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasZhemmStridedBatched(handle, side, uplo, n, k, alpha,&
                            A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count)
    end function hipblasZhemmStridedBatchedFortran

    ! herk
    function hipblasCherkFortran(handle, uplo, transA, n, k, alpha, &
            A, lda, beta, C, ldc) &
            result(res) &
            bind(c, name = 'hipblasCherkFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(c_int), value :: n
        integer(c_int), value :: k
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: beta
        type(c_ptr), value :: C
        integer(c_int), value :: ldc
        integer(c_int) :: res
        res = hipblasCherk(handle, uplo, transA, n, k, alpha,&
              A, lda, beta, C, ldc)
    end function hipblasCherkFortran

    function hipblasZherkFortran(handle, uplo, transA, n, k, alpha, &
            A, lda, beta, C, ldc) &
            result(res) &
            bind(c, name = 'hipblasZherkFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(c_int), value :: n
        integer(c_int), value :: k
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: beta
        type(c_ptr), value :: C
        integer(c_int), value :: ldc
        integer(c_int) :: res
        res = hipblasZherk(handle, uplo, transA, n, k, alpha,&
              A, lda, beta, C, ldc)
    end function hipblasZherkFortran

    ! herkBatched
    function hipblasCherkBatchedFortran(handle, uplo, transA, n, k, alpha, &
            A, lda, beta, C, ldc, batch_count) &
            result(res) &
            bind(c, name = 'hipblasCherkBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(c_int), value :: n
        integer(c_int), value :: k
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: beta
        type(c_ptr), value :: C
        integer(c_int), value :: ldc
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasCherkBatched(handle, uplo, transA, n, k, alpha,&
              A, lda, beta, C, ldc, batch_count)
    end function hipblasCherkBatchedFortran

    function hipblasZherkBatchedFortran(handle, uplo, transA, n, k, alpha, &
            A, lda, beta, C, ldc, batch_count) &
            result(res) &
            bind(c, name = 'hipblasZherkBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(c_int), value :: n
        integer(c_int), value :: k
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: beta
        type(c_ptr), value :: C
        integer(c_int), value :: ldc
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasZherkBatched(handle, uplo, transA, n, k, alpha,&
              A, lda, beta, C, ldc, batch_count)
    end function hipblasZherkBatchedFortran

    ! herkStridedBatched
    function hipblasCherkStridedBatchedFortran(handle, uplo, transA, n, k, alpha, &
            A, lda, stride_A, beta, C, ldc, stride_C, batch_count) &
            result(res) &
            bind(c, name = 'hipblasCherkStridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(c_int), value :: n
        integer(c_int), value :: k
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int64_t), value :: stride_A
        type(c_ptr), value :: beta
        type(c_ptr), value :: C
        integer(c_int), value :: ldc
        integer(c_int64_t), value :: stride_C
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasCherkStridedBatched(handle, uplo, transA, n, k, alpha,&
              A, lda, stride_A, beta, C, ldc, stride_C, batch_count)
    end function hipblasCherkStridedBatchedFortran

    function hipblasZherkStridedBatchedFortran(handle, uplo, transA, n, k, alpha, &
            A, lda, stride_A, beta, C, ldc, stride_C, batch_count) &
            result(res) &
            bind(c, name = 'hipblasZherkStridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(c_int), value :: n
        integer(c_int), value :: k
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int64_t), value :: stride_A
        type(c_ptr), value :: beta
        type(c_ptr), value :: C
        integer(c_int), value :: ldc
        integer(c_int64_t), value :: stride_C
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasZherkStridedBatched(handle, uplo, transA, n, k, alpha,&
              A, lda, stride_A, beta, C, ldc, stride_C, batch_count)
    end function hipblasZherkStridedBatchedFortran

    ! her2k
    function hipblasCher2kFortran(handle, uplo, transA, n, k, alpha, &
            A, lda, B, ldb, beta, C, ldc) &
            result(res) &
            bind(c, name = 'hipblasCher2kFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(c_int), value :: n
        integer(c_int), value :: k
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: B
        integer(c_int), value :: ldb
        type(c_ptr), value :: beta
        type(c_ptr), value :: C
        integer(c_int), value :: ldc
        integer(c_int) :: res
        res = hipblasCher2k(handle, uplo, transA, n, k, alpha,&
              A, lda, B, ldb, beta, C, ldc)
    end function hipblasCher2kFortran

    function hipblasZher2kFortran(handle, uplo, transA, n, k, alpha, &
            A, lda, B, ldb, beta, C, ldc) &
            result(res) &
            bind(c, name = 'hipblasZher2kFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(c_int), value :: n
        integer(c_int), value :: k
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: B
        integer(c_int), value :: ldb
        type(c_ptr), value :: beta
        type(c_ptr), value :: C
        integer(c_int), value :: ldc
        integer(c_int) :: res
        res = hipblasZher2k(handle, uplo, transA, n, k, alpha,&
              A, lda, B, ldb, beta, C, ldc)
    end function hipblasZher2kFortran

    ! her2kBatched
    function hipblasCher2kBatchedFortran(handle, uplo, transA, n, k, alpha, &
            A, lda, B, ldb, beta, C, ldc, batch_count) &
            result(res) &
            bind(c, name = 'hipblasCher2kBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(c_int), value :: n
        integer(c_int), value :: k
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: B
        integer(c_int), value :: ldb
        type(c_ptr), value :: beta
        type(c_ptr), value :: C
        integer(c_int), value :: ldc
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasCher2kBatched(handle, uplo, transA, n, k, alpha,&
              A, lda, B, ldb, beta, C, ldc, batch_count)
    end function hipblasCher2kBatchedFortran

    function hipblasZher2kBatchedFortran(handle, uplo, transA, n, k, alpha, &
            A, lda, B, ldb, beta, C, ldc, batch_count) &
            result(res) &
            bind(c, name = 'hipblasZher2kBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(c_int), value :: n
        integer(c_int), value :: k
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: B
        integer(c_int), value :: ldb
        type(c_ptr), value :: beta
        type(c_ptr), value :: C
        integer(c_int), value :: ldc
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasZher2kBatched(handle, uplo, transA, n, k, alpha,&
              A, lda, B, ldb, beta, C, ldc, batch_count)
    end function hipblasZher2kBatchedFortran

    ! her2kStridedBatched
    function hipblasCher2kStridedBatchedFortran(handle, uplo, transA, n, k, alpha, &
            A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count) &
            result(res) &
            bind(c, name = 'hipblasCher2kStridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(c_int), value :: n
        integer(c_int), value :: k
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int64_t), value :: stride_A
        type(c_ptr), value :: B
        integer(c_int), value :: ldb
        integer(c_int64_t), value :: stride_B
        type(c_ptr), value :: beta
        type(c_ptr), value :: C
        integer(c_int), value :: ldc
        integer(c_int64_t), value :: stride_C
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasCher2kStridedBatched(handle, uplo, transA, n, k, alpha,&
              A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count)
    end function hipblasCher2kStridedBatchedFortran

    function hipblasZher2kStridedBatchedFortran(handle, uplo, transA, n, k, alpha, &
            A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count) &
            result(res) &
            bind(c, name = 'hipblasZher2kStridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(c_int), value :: n
        integer(c_int), value :: k
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int64_t), value :: stride_A
        type(c_ptr), value :: B
        integer(c_int), value :: ldb
        integer(c_int64_t), value :: stride_B
        type(c_ptr), value :: beta
        type(c_ptr), value :: C
        integer(c_int), value :: ldc
        integer(c_int64_t), value :: stride_C
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasZher2kStridedBatched(handle, uplo, transA, n, k, alpha,&
              A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count)
    end function hipblasZher2kStridedBatchedFortran

    ! herkx
    function hipblasCherkxFortran(handle, uplo, transA, n, k, alpha, &
            A, lda, B, ldb, beta, C, ldc) &
            result(res) &
            bind(c, name = 'hipblasCherkxFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(c_int), value :: n
        integer(c_int), value :: k
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: B
        integer(c_int), value :: ldb
        type(c_ptr), value :: beta
        type(c_ptr), value :: C
        integer(c_int), value :: ldc
        integer(c_int) :: res
        res = hipblasCherkx(handle, uplo, transA, n, k, alpha,&
              A, lda, B, ldb, beta, C, ldc)
    end function hipblasCherkxFortran

    function hipblasZherkxFortran(handle, uplo, transA, n, k, alpha, &
            A, lda, B, ldb, beta, C, ldc) &
            result(res) &
            bind(c, name = 'hipblasZherkxFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(c_int), value :: n
        integer(c_int), value :: k
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: B
        integer(c_int), value :: ldb
        type(c_ptr), value :: beta
        type(c_ptr), value :: C
        integer(c_int), value :: ldc
        integer(c_int) :: res
        res = hipblasZherkx(handle, uplo, transA, n, k, alpha,&
              A, lda, B, ldb, beta, C, ldc)
    end function hipblasZherkxFortran

    ! herkxBatched
    function hipblasCherkxBatchedFortran(handle, uplo, transA, n, k, alpha, &
            A, lda, B, ldb, beta, C, ldc, batch_count) &
            result(res) &
            bind(c, name = 'hipblasCherkxBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(c_int), value :: n
        integer(c_int), value :: k
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: B
        integer(c_int), value :: ldb
        type(c_ptr), value :: beta
        type(c_ptr), value :: C
        integer(c_int), value :: ldc
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasCherkxBatched(handle, uplo, transA, n, k, alpha,&
              A, lda, B, ldb, beta, C, ldc, batch_count)
    end function hipblasCherkxBatchedFortran

    function hipblasZherkxBatchedFortran(handle, uplo, transA, n, k, alpha, &
            A, lda, B, ldb, beta, C, ldc, batch_count) &
            result(res) &
            bind(c, name = 'hipblasZherkxBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(c_int), value :: n
        integer(c_int), value :: k
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: B
        integer(c_int), value :: ldb
        type(c_ptr), value :: beta
        type(c_ptr), value :: C
        integer(c_int), value :: ldc
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasZherkxBatched(handle, uplo, transA, n, k, alpha,&
              A, lda, B, ldb, beta, C, ldc, batch_count)
    end function hipblasZherkxBatchedFortran

    ! herkxStridedBatched
    function hipblasCherkxStridedBatchedFortran(handle, uplo, transA, n, k, alpha, &
            A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count) &
            result(res) &
            bind(c, name = 'hipblasCherkxStridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(c_int), value :: n
        integer(c_int), value :: k
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int64_t), value :: stride_A
        type(c_ptr), value :: B
        integer(c_int), value :: ldb
        integer(c_int64_t), value :: stride_B
        type(c_ptr), value :: beta
        type(c_ptr), value :: C
        integer(c_int), value :: ldc
        integer(c_int64_t), value :: stride_C
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasCherkxStridedBatched(handle, uplo, transA, n, k, alpha,&
              A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count)
    end function hipblasCherkxStridedBatchedFortran

    function hipblasZherkxStridedBatchedFortran(handle, uplo, transA, n, k, alpha, &
            A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count) &
            result(res) &
            bind(c, name = 'hipblasZherkxStridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(c_int), value :: n
        integer(c_int), value :: k
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int64_t), value :: stride_A
        type(c_ptr), value :: B
        integer(c_int), value :: ldb
        integer(c_int64_t), value :: stride_B
        type(c_ptr), value :: beta
        type(c_ptr), value :: C
        integer(c_int), value :: ldc
        integer(c_int64_t), value :: stride_C
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasZherkxStridedBatched(handle, uplo, transA, n, k, alpha,&
              A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count)
    end function hipblasZherkxStridedBatchedFortran

    ! symm
    function hipblasSsymmFortran(handle, side, uplo, m, n, alpha, &
            A, lda, B, ldb, beta, C, ldc) &
            result(res) &
            bind(c, name = 'hipblasSsymmFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: m
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: B
        integer(c_int), value :: ldb
        type(c_ptr), value :: beta
        type(c_ptr), value :: C
        integer(c_int), value :: ldc
        integer(c_int) :: res
        res = hipblasSsymm(handle, side, uplo, m, n, alpha,&
              A, lda, B, ldb, beta, C, ldc)
    end function hipblasSsymmFortran

    function hipblasDsymmFortran(handle, side, uplo, m, n, alpha, &
            A, lda, B, ldb, beta, C, ldc) &
            result(res) &
            bind(c, name = 'hipblasDsymmFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: m
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: B
        integer(c_int), value :: ldb
        type(c_ptr), value :: beta
        type(c_ptr), value :: C
        integer(c_int), value :: ldc
        integer(c_int) :: res
        res = hipblasDsymm(handle, side, uplo, m, n, alpha,&
              A, lda, B, ldb, beta, C, ldc)
    end function hipblasDsymmFortran

    function hipblasCsymmFortran(handle, side, uplo, m, n, alpha, &
            A, lda, B, ldb, beta, C, ldc) &
            result(res) &
            bind(c, name = 'hipblasCsymmFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: m
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: B
        integer(c_int), value :: ldb
        type(c_ptr), value :: beta
        type(c_ptr), value :: C
        integer(c_int), value :: ldc
        integer(c_int) :: res
        res = hipblasCsymm(handle, side, uplo, m, n, alpha,&
              A, lda, B, ldb, beta, C, ldc)
    end function hipblasCsymmFortran

    function hipblasZsymmFortran(handle, side, uplo, m, n, alpha, &
            A, lda, B, ldb, beta, C, ldc) &
            result(res) &
            bind(c, name = 'hipblasZsymmFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: m
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: B
        integer(c_int), value :: ldb
        type(c_ptr), value :: beta
        type(c_ptr), value :: C
        integer(c_int), value :: ldc
        integer(c_int) :: res
        res = hipblasZsymm(handle, side, uplo, m, n, alpha,&
              A, lda, B, ldb, beta, C, ldc)
    end function hipblasZsymmFortran

    ! symmBatched
    function hipblasSsymmBatchedFortran(handle, side, uplo, m, n, alpha, &
            A, lda, B, ldb, beta, C, ldc, batch_count) &
            result(res) &
            bind(c, name = 'hipblasSsymmBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: m
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: B
        integer(c_int), value :: ldb
        type(c_ptr), value :: beta
        type(c_ptr), value :: C
        integer(c_int), value :: ldc
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasSsymmBatched(handle, side, uplo, m, n, alpha,&
              A, lda, B, ldb, beta, C, ldc, batch_count)
    end function hipblasSsymmBatchedFortran

    function hipblasDsymmBatchedFortran(handle, side, uplo, m, n, alpha, &
            A, lda, B, ldb, beta, C, ldc, batch_count) &
            result(res) &
            bind(c, name = 'hipblasDsymmBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: m
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: B
        integer(c_int), value :: ldb
        type(c_ptr), value :: beta
        type(c_ptr), value :: C
        integer(c_int), value :: ldc
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasDsymmBatched(handle, side, uplo, m, n, alpha,&
              A, lda, B, ldb, beta, C, ldc, batch_count)
    end function hipblasDsymmBatchedFortran

    function hipblasCsymmBatchedFortran(handle, side, uplo, m, n, alpha, &
            A, lda, B, ldb, beta, C, ldc, batch_count) &
            result(res) &
            bind(c, name = 'hipblasCsymmBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: m
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: B
        integer(c_int), value :: ldb
        type(c_ptr), value :: beta
        type(c_ptr), value :: C
        integer(c_int), value :: ldc
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasCsymmBatched(handle, side, uplo, m, n, alpha,&
              A, lda, B, ldb, beta, C, ldc, batch_count)
    end function hipblasCsymmBatchedFortran

    function hipblasZsymmBatchedFortran(handle, side, uplo, m, n, alpha, &
            A, lda, B, ldb, beta, C, ldc, batch_count) &
            result(res) &
            bind(c, name = 'hipblasZsymmBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: m
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: B
        integer(c_int), value :: ldb
        type(c_ptr), value :: beta
        type(c_ptr), value :: C
        integer(c_int), value :: ldc
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasZsymmBatched(handle, side, uplo, m, n, alpha,&
              A, lda, B, ldb, beta, C, ldc, batch_count)
    end function hipblasZsymmBatchedFortran

    ! symmStridedBatched
    function hipblasSsymmStridedBatchedFortran(handle, side, uplo, m, n, alpha, &
            A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count) &
            result(res) &
            bind(c, name = 'hipblasSsymmStridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: m
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int64_t), value :: stride_A
        type(c_ptr), value :: B
        integer(c_int), value :: ldb
        integer(c_int64_t), value :: stride_B
        type(c_ptr), value :: beta
        type(c_ptr), value :: C
        integer(c_int), value :: ldc
        integer(c_int64_t), value :: stride_C
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasSsymmStridedBatched(handle, side, uplo, m, n, alpha,&
              A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count)
    end function hipblasSsymmStridedBatchedFortran

    function hipblasDsymmStridedBatchedFortran(handle, side, uplo, m, n, alpha, &
            A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count) &
            result(res) &
            bind(c, name = 'hipblasDsymmStridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: m
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int64_t), value :: stride_A
        type(c_ptr), value :: B
        integer(c_int), value :: ldb
        integer(c_int64_t), value :: stride_B
        type(c_ptr), value :: beta
        type(c_ptr), value :: C
        integer(c_int), value :: ldc
        integer(c_int64_t), value :: stride_C
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasDsymmStridedBatched(handle, side, uplo, m, n, alpha,&
              A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count)
    end function hipblasDsymmStridedBatchedFortran

    function hipblasCsymmStridedBatchedFortran(handle, side, uplo, m, n, alpha, &
            A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count) &
            result(res) &
            bind(c, name = 'hipblasCsymmStridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: m
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int64_t), value :: stride_A
        type(c_ptr), value :: B
        integer(c_int), value :: ldb
        integer(c_int64_t), value :: stride_B
        type(c_ptr), value :: beta
        type(c_ptr), value :: C
        integer(c_int), value :: ldc
        integer(c_int64_t), value :: stride_C
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasCsymmStridedBatched(handle, side, uplo, m, n, alpha,&
              A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count)
    end function hipblasCsymmStridedBatchedFortran

    function hipblasZsymmStridedBatchedFortran(handle, side, uplo, m, n, alpha, &
            A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count) &
            result(res) &
            bind(c, name = 'hipblasZsymmStridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(c_int), value :: m
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int64_t), value :: stride_A
        type(c_ptr), value :: B
        integer(c_int), value :: ldb
        integer(c_int64_t), value :: stride_B
        type(c_ptr), value :: beta
        type(c_ptr), value :: C
        integer(c_int), value :: ldc
        integer(c_int64_t), value :: stride_C
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasZsymmStridedBatched(handle, side, uplo, m, n, alpha,&
              A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count)
    end function hipblasZsymmStridedBatchedFortran

    ! syrk
    function hipblasSsyrkFortran(handle, uplo, transA, n, k, alpha, &
            A, lda, beta, C, ldc) &
            result(res) &
            bind(c, name = 'hipblasSsyrkFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(c_int), value :: n
        integer(c_int), value :: k
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: beta
        type(c_ptr), value :: C
        integer(c_int), value :: ldc
        integer(c_int) :: res
        res = hipblasSsyrk(handle, uplo, transA, n, k, alpha,&
              A, lda, beta, C, ldc)
    end function hipblasSsyrkFortran

    function hipblasDsyrkFortran(handle, uplo, transA, n, k, alpha, &
            A, lda, beta, C, ldc) &
            result(res) &
            bind(c, name = 'hipblasDsyrkFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(c_int), value :: n
        integer(c_int), value :: k
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: beta
        type(c_ptr), value :: C
        integer(c_int), value :: ldc
        integer(c_int) :: res
        res = hipblasDsyrk(handle, uplo, transA, n, k, alpha,&
              A, lda, beta, C, ldc)
    end function hipblasDsyrkFortran

    function hipblasCsyrkFortran(handle, uplo, transA, n, k, alpha, &
            A, lda, beta, C, ldc) &
            result(res) &
            bind(c, name = 'hipblasCsyrkFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(c_int), value :: n
        integer(c_int), value :: k
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: beta
        type(c_ptr), value :: C
        integer(c_int), value :: ldc
        integer(c_int) :: res
        res = hipblasCsyrk(handle, uplo, transA, n, k, alpha,&
              A, lda, beta, C, ldc)
    end function hipblasCsyrkFortran

    function hipblasZsyrkFortran(handle, uplo, transA, n, k, alpha, &
            A, lda, beta, C, ldc) &
            result(res) &
            bind(c, name = 'hipblasZsyrkFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(c_int), value :: n
        integer(c_int), value :: k
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: beta
        type(c_ptr), value :: C
        integer(c_int), value :: ldc
        integer(c_int) :: res
        res = hipblasZsyrk(handle, uplo, transA, n, k, alpha,&
              A, lda, beta, C, ldc)
    end function hipblasZsyrkFortran

    ! syrkBatched
    function hipblasSsyrkBatchedFortran(handle, uplo, transA, n, k, alpha, &
            A, lda, beta, C, ldc, batch_count) &
            result(res) &
            bind(c, name = 'hipblasSsyrkBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(c_int), value :: n
        integer(c_int), value :: k
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: beta
        type(c_ptr), value :: C
        integer(c_int), value :: ldc
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasSsyrkBatched(handle, uplo, transA, n, k, alpha,&
              A, lda, beta, C, ldc, batch_count)
    end function hipblasSsyrkBatchedFortran

    function hipblasDsyrkBatchedFortran(handle, uplo, transA, n, k, alpha, &
            A, lda, beta, C, ldc, batch_count) &
            result(res) &
            bind(c, name = 'hipblasDsyrkBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(c_int), value :: n
        integer(c_int), value :: k
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: beta
        type(c_ptr), value :: C
        integer(c_int), value :: ldc
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasDsyrkBatched(handle, uplo, transA, n, k, alpha,&
              A, lda, beta, C, ldc, batch_count)
    end function hipblasDsyrkBatchedFortran

    function hipblasCsyrkBatchedFortran(handle, uplo, transA, n, k, alpha, &
            A, lda, beta, C, ldc, batch_count) &
            result(res) &
            bind(c, name = 'hipblasCsyrkBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(c_int), value :: n
        integer(c_int), value :: k
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: beta
        type(c_ptr), value :: C
        integer(c_int), value :: ldc
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasCsyrkBatched(handle, uplo, transA, n, k, alpha,&
              A, lda, beta, C, ldc, batch_count)
    end function hipblasCsyrkBatchedFortran

    function hipblasZsyrkBatchedFortran(handle, uplo, transA, n, k, alpha, &
            A, lda, beta, C, ldc, batch_count) &
            result(res) &
            bind(c, name = 'hipblasZsyrkBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(c_int), value :: n
        integer(c_int), value :: k
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: beta
        type(c_ptr), value :: C
        integer(c_int), value :: ldc
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasZsyrkBatched(handle, uplo, transA, n, k, alpha,&
              A, lda, beta, C, ldc, batch_count)
    end function hipblasZsyrkBatchedFortran

    ! syrkStridedBatched
    function hipblasSsyrkStridedBatchedFortran(handle, uplo, transA, n, k, alpha, &
            A, lda, stride_A, beta, C, ldc, stride_C, batch_count) &
            result(res) &
            bind(c, name = 'hipblasSsyrkStridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(c_int), value :: n
        integer(c_int), value :: k
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int64_t), value :: stride_A
        type(c_ptr), value :: beta
        type(c_ptr), value :: C
        integer(c_int), value :: ldc
        integer(c_int64_t), value :: stride_C
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasSsyrkStridedBatched(handle, uplo, transA, n, k, alpha,&
              A, lda, stride_A, beta, C, ldc, stride_C, batch_count)
    end function hipblasSsyrkStridedBatchedFortran

    function hipblasDsyrkStridedBatchedFortran(handle, uplo, transA, n, k, alpha, &
            A, lda, stride_A, beta, C, ldc, stride_C, batch_count) &
            result(res) &
            bind(c, name = 'hipblasDsyrkStridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(c_int), value :: n
        integer(c_int), value :: k
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int64_t), value :: stride_A
        type(c_ptr), value :: beta
        type(c_ptr), value :: C
        integer(c_int), value :: ldc
        integer(c_int64_t), value :: stride_C
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasDsyrkStridedBatched(handle, uplo, transA, n, k, alpha,&
              A, lda, stride_A, beta, C, ldc, stride_C, batch_count)
    end function hipblasDsyrkStridedBatchedFortran

    function hipblasCsyrkStridedBatchedFortran(handle, uplo, transA, n, k, alpha, &
            A, lda, stride_A, beta, C, ldc, stride_C, batch_count) &
            result(res) &
            bind(c, name = 'hipblasCsyrkStridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(c_int), value :: n
        integer(c_int), value :: k
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int64_t), value :: stride_A
        type(c_ptr), value :: beta
        type(c_ptr), value :: C
        integer(c_int), value :: ldc
        integer(c_int64_t), value :: stride_C
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasCsyrkStridedBatched(handle, uplo, transA, n, k, alpha,&
              A, lda, stride_A, beta, C, ldc, stride_C, batch_count)
    end function hipblasCsyrkStridedBatchedFortran

    function hipblasZsyrkStridedBatchedFortran(handle, uplo, transA, n, k, alpha, &
            A, lda, stride_A, beta, C, ldc, stride_C, batch_count) &
            result(res) &
            bind(c, name = 'hipblasZsyrkStridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(c_int), value :: n
        integer(c_int), value :: k
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int64_t), value :: stride_A
        type(c_ptr), value :: beta
        type(c_ptr), value :: C
        integer(c_int), value :: ldc
        integer(c_int64_t), value :: stride_C
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasZsyrkStridedBatched(handle, uplo, transA, n, k, alpha,&
              A, lda, stride_A, beta, C, ldc, stride_C, batch_count)
    end function hipblasZsyrkStridedBatchedFortran

    ! syr2k
    function hipblasSsyr2kFortran(handle, uplo, transA, n, k, alpha, &
            A, lda, B, ldb, beta, C, ldc) &
            result(res) &
            bind(c, name = 'hipblasSsyr2kFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(c_int), value :: n
        integer(c_int), value :: k
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: B
        integer(c_int), value :: ldb
        type(c_ptr), value :: beta
        type(c_ptr), value :: C
        integer(c_int), value :: ldc
        integer(c_int) :: res
        res = hipblasSsyr2k(handle, uplo, transA, n, k, alpha,&
              A, lda, B, ldb, beta, C, ldc)
    end function hipblasSsyr2kFortran

    function hipblasDsyr2kFortran(handle, uplo, transA, n, k, alpha, &
            A, lda, B, ldb, beta, C, ldc) &
            result(res) &
            bind(c, name = 'hipblasDsyr2kFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(c_int), value :: n
        integer(c_int), value :: k
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: B
        integer(c_int), value :: ldb
        type(c_ptr), value :: beta
        type(c_ptr), value :: C
        integer(c_int), value :: ldc
        integer(c_int) :: res
        res = hipblasDsyr2k(handle, uplo, transA, n, k, alpha,&
              A, lda, B, ldb, beta, C, ldc)
    end function hipblasDsyr2kFortran

    function hipblasCsyr2kFortran(handle, uplo, transA, n, k, alpha, &
            A, lda, B, ldb, beta, C, ldc) &
            result(res) &
            bind(c, name = 'hipblasCsyr2kFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(c_int), value :: n
        integer(c_int), value :: k
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: B
        integer(c_int), value :: ldb
        type(c_ptr), value :: beta
        type(c_ptr), value :: C
        integer(c_int), value :: ldc
        integer(c_int) :: res
        res = hipblasCsyr2k(handle, uplo, transA, n, k, alpha,&
              A, lda, B, ldb, beta, C, ldc)
    end function hipblasCsyr2kFortran

    function hipblasZsyr2kFortran(handle, uplo, transA, n, k, alpha, &
            A, lda, B, ldb, beta, C, ldc) &
            result(res) &
            bind(c, name = 'hipblasZsyr2kFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(c_int), value :: n
        integer(c_int), value :: k
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: B
        integer(c_int), value :: ldb
        type(c_ptr), value :: beta
        type(c_ptr), value :: C
        integer(c_int), value :: ldc
        integer(c_int) :: res
        res = hipblasZsyr2k(handle, uplo, transA, n, k, alpha,&
              A, lda, B, ldb, beta, C, ldc)
    end function hipblasZsyr2kFortran

    ! syr2kBatched
    function hipblasSsyr2kBatchedFortran(handle, uplo, transA, n, k, alpha, &
            A, lda, B, ldb, beta, C, ldc, batch_count) &
            result(res) &
            bind(c, name = 'hipblasSsyr2kBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(c_int), value :: n
        integer(c_int), value :: k
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: B
        integer(c_int), value :: ldb
        type(c_ptr), value :: beta
        type(c_ptr), value :: C
        integer(c_int), value :: ldc
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasSsyr2kBatched(handle, uplo, transA, n, k, alpha,&
              A, lda, B, ldb, beta, C, ldc, batch_count)
    end function hipblasSsyr2kBatchedFortran

    function hipblasDsyr2kBatchedFortran(handle, uplo, transA, n, k, alpha, &
            A, lda, B, ldb, beta, C, ldc, batch_count) &
            result(res) &
            bind(c, name = 'hipblasDsyr2kBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(c_int), value :: n
        integer(c_int), value :: k
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: B
        integer(c_int), value :: ldb
        type(c_ptr), value :: beta
        type(c_ptr), value :: C
        integer(c_int), value :: ldc
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasDsyr2kBatched(handle, uplo, transA, n, k, alpha,&
              A, lda, B, ldb, beta, C, ldc, batch_count)
    end function hipblasDsyr2kBatchedFortran

    function hipblasCsyr2kBatchedFortran(handle, uplo, transA, n, k, alpha, &
            A, lda, B, ldb, beta, C, ldc, batch_count) &
            result(res) &
            bind(c, name = 'hipblasCsyr2kBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(c_int), value :: n
        integer(c_int), value :: k
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: B
        integer(c_int), value :: ldb
        type(c_ptr), value :: beta
        type(c_ptr), value :: C
        integer(c_int), value :: ldc
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasCsyr2kBatched(handle, uplo, transA, n, k, alpha,&
              A, lda, B, ldb, beta, C, ldc, batch_count)
    end function hipblasCsyr2kBatchedFortran

    function hipblasZsyr2kBatchedFortran(handle, uplo, transA, n, k, alpha, &
            A, lda, B, ldb, beta, C, ldc, batch_count) &
            result(res) &
            bind(c, name = 'hipblasZsyr2kBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(c_int), value :: n
        integer(c_int), value :: k
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: B
        integer(c_int), value :: ldb
        type(c_ptr), value :: beta
        type(c_ptr), value :: C
        integer(c_int), value :: ldc
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasZsyr2kBatched(handle, uplo, transA, n, k, alpha,&
              A, lda, B, ldb, beta, C, ldc, batch_count)
    end function hipblasZsyr2kBatchedFortran

    ! syr2kStridedBatched
    function hipblasSsyr2kStridedBatchedFortran(handle, uplo, transA, n, k, alpha, &
            A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count) &
            result(res) &
            bind(c, name = 'hipblasSsyr2kStridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(c_int), value :: n
        integer(c_int), value :: k
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int64_t), value :: stride_A
        type(c_ptr), value :: B
        integer(c_int), value :: ldb
        integer(c_int64_t), value :: stride_B
        type(c_ptr), value :: beta
        type(c_ptr), value :: C
        integer(c_int), value :: ldc
        integer(c_int64_t), value :: stride_C
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasSsyr2kStridedBatched(handle, uplo, transA, n, k, alpha,&
              A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count)
    end function hipblasSsyr2kStridedBatchedFortran

    function hipblasDsyr2kStridedBatchedFortran(handle, uplo, transA, n, k, alpha, &
            A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count) &
            result(res) &
            bind(c, name = 'hipblasDsyr2kStridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(c_int), value :: n
        integer(c_int), value :: k
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int64_t), value :: stride_A
        type(c_ptr), value :: B
        integer(c_int), value :: ldb
        integer(c_int64_t), value :: stride_B
        type(c_ptr), value :: beta
        type(c_ptr), value :: C
        integer(c_int), value :: ldc
        integer(c_int64_t), value :: stride_C
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasDsyr2kStridedBatched(handle, uplo, transA, n, k, alpha,&
              A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count)
    end function hipblasDsyr2kStridedBatchedFortran

    function hipblasCsyr2kStridedBatchedFortran(handle, uplo, transA, n, k, alpha, &
            A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count) &
            result(res) &
            bind(c, name = 'hipblasCsyr2kStridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(c_int), value :: n
        integer(c_int), value :: k
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int64_t), value :: stride_A
        type(c_ptr), value :: B
        integer(c_int), value :: ldb
        integer(c_int64_t), value :: stride_B
        type(c_ptr), value :: beta
        type(c_ptr), value :: C
        integer(c_int), value :: ldc
        integer(c_int64_t), value :: stride_C
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasCsyr2kStridedBatched(handle, uplo, transA, n, k, alpha,&
              A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count)
    end function hipblasCsyr2kStridedBatchedFortran

    function hipblasZsyr2kStridedBatchedFortran(handle, uplo, transA, n, k, alpha, &
            A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count) &
            result(res) &
            bind(c, name = 'hipblasZsyr2kStridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(c_int), value :: n
        integer(c_int), value :: k
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int64_t), value :: stride_A
        type(c_ptr), value :: B
        integer(c_int), value :: ldb
        integer(c_int64_t), value :: stride_B
        type(c_ptr), value :: beta
        type(c_ptr), value :: C
        integer(c_int), value :: ldc
        integer(c_int64_t), value :: stride_C
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasZsyr2kStridedBatched(handle, uplo, transA, n, k, alpha,&
              A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count)
    end function hipblasZsyr2kStridedBatchedFortran

    ! syrkx
    function hipblasSsyrkxFortran(handle, uplo, transA, n, k, alpha, &
            A, lda, B, ldb, beta, C, ldc) &
            result(res) &
            bind(c, name = 'hipblasSsyrkxFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(c_int), value :: n
        integer(c_int), value :: k
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: B
        integer(c_int), value :: ldb
        type(c_ptr), value :: beta
        type(c_ptr), value :: C
        integer(c_int), value :: ldc
        integer(c_int) :: res
        res = hipblasSsyrkx(handle, uplo, transA, n, k, alpha,&
              A, lda, B, ldb, beta, C, ldc)
    end function hipblasSsyrkxFortran

    function hipblasDsyrkxFortran(handle, uplo, transA, n, k, alpha, &
            A, lda, B, ldb, beta, C, ldc) &
            result(res) &
            bind(c, name = 'hipblasDsyrkxFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(c_int), value :: n
        integer(c_int), value :: k
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: B
        integer(c_int), value :: ldb
        type(c_ptr), value :: beta
        type(c_ptr), value :: C
        integer(c_int), value :: ldc
        integer(c_int) :: res
        res = hipblasDsyrkx(handle, uplo, transA, n, k, alpha,&
              A, lda, B, ldb, beta, C, ldc)
    end function hipblasDsyrkxFortran

    function hipblasCsyrkxFortran(handle, uplo, transA, n, k, alpha, &
            A, lda, B, ldb, beta, C, ldc) &
            result(res) &
            bind(c, name = 'hipblasCsyrkxFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(c_int), value :: n
        integer(c_int), value :: k
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: B
        integer(c_int), value :: ldb
        type(c_ptr), value :: beta
        type(c_ptr), value :: C
        integer(c_int), value :: ldc
        integer(c_int) :: res
        res = hipblasCsyrkx(handle, uplo, transA, n, k, alpha,&
              A, lda, B, ldb, beta, C, ldc)
    end function hipblasCsyrkxFortran

    function hipblasZsyrkxFortran(handle, uplo, transA, n, k, alpha, &
            A, lda, B, ldb, beta, C, ldc) &
            result(res) &
            bind(c, name = 'hipblasZsyrkxFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(c_int), value :: n
        integer(c_int), value :: k
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: B
        integer(c_int), value :: ldb
        type(c_ptr), value :: beta
        type(c_ptr), value :: C
        integer(c_int), value :: ldc
        integer(c_int) :: res
        res = hipblasZsyrkx(handle, uplo, transA, n, k, alpha,&
              A, lda, B, ldb, beta, C, ldc)
    end function hipblasZsyrkxFortran

    ! syrkxBatched
    function hipblasSsyrkxBatchedFortran(handle, uplo, transA, n, k, alpha, &
            A, lda, B, ldb, beta, C, ldc, batch_count) &
            result(res) &
            bind(c, name = 'hipblasSsyrkxBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(c_int), value :: n
        integer(c_int), value :: k
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: B
        integer(c_int), value :: ldb
        type(c_ptr), value :: beta
        type(c_ptr), value :: C
        integer(c_int), value :: ldc
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasSsyrkxBatched(handle, uplo, transA, n, k, alpha,&
              A, lda, B, ldb, beta, C, ldc, batch_count)
    end function hipblasSsyrkxBatchedFortran

    function hipblasDsyrkxBatchedFortran(handle, uplo, transA, n, k, alpha, &
            A, lda, B, ldb, beta, C, ldc, batch_count) &
            result(res) &
            bind(c, name = 'hipblasDsyrkxBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(c_int), value :: n
        integer(c_int), value :: k
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: B
        integer(c_int), value :: ldb
        type(c_ptr), value :: beta
        type(c_ptr), value :: C
        integer(c_int), value :: ldc
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasDsyrkxBatched(handle, uplo, transA, n, k, alpha,&
              A, lda, B, ldb, beta, C, ldc, batch_count)
    end function hipblasDsyrkxBatchedFortran

    function hipblasCsyrkxBatchedFortran(handle, uplo, transA, n, k, alpha, &
            A, lda, B, ldb, beta, C, ldc, batch_count) &
            result(res) &
            bind(c, name = 'hipblasCsyrkxBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(c_int), value :: n
        integer(c_int), value :: k
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: B
        integer(c_int), value :: ldb
        type(c_ptr), value :: beta
        type(c_ptr), value :: C
        integer(c_int), value :: ldc
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasCsyrkxBatched(handle, uplo, transA, n, k, alpha,&
              A, lda, B, ldb, beta, C, ldc, batch_count)
    end function hipblasCsyrkxBatchedFortran

    function hipblasZsyrkxBatchedFortran(handle, uplo, transA, n, k, alpha, &
            A, lda, B, ldb, beta, C, ldc, batch_count) &
            result(res) &
            bind(c, name = 'hipblasZsyrkxBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(c_int), value :: n
        integer(c_int), value :: k
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: B
        integer(c_int), value :: ldb
        type(c_ptr), value :: beta
        type(c_ptr), value :: C
        integer(c_int), value :: ldc
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasZsyrkxBatched(handle, uplo, transA, n, k, alpha,&
              A, lda, B, ldb, beta, C, ldc, batch_count)
    end function hipblasZsyrkxBatchedFortran

    ! syrkxStridedBatched
    function hipblasSsyrkxStridedBatchedFortran(handle, uplo, transA, n, k, alpha, &
            A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count) &
            result(res) &
            bind(c, name = 'hipblasSsyrkxStridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(c_int), value :: n
        integer(c_int), value :: k
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int64_t), value :: stride_A
        type(c_ptr), value :: B
        integer(c_int), value :: ldb
        integer(c_int64_t), value :: stride_B
        type(c_ptr), value :: beta
        type(c_ptr), value :: C
        integer(c_int), value :: ldc
        integer(c_int64_t), value :: stride_C
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasSsyrkxStridedBatched(handle, uplo, transA, n, k, alpha,&
              A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count)
    end function hipblasSsyrkxStridedBatchedFortran

    function hipblasDsyrkxStridedBatchedFortran(handle, uplo, transA, n, k, alpha, &
            A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count) &
            result(res) &
            bind(c, name = 'hipblasDsyrkxStridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(c_int), value :: n
        integer(c_int), value :: k
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int64_t), value :: stride_A
        type(c_ptr), value :: B
        integer(c_int), value :: ldb
        integer(c_int64_t), value :: stride_B
        type(c_ptr), value :: beta
        type(c_ptr), value :: C
        integer(c_int), value :: ldc
        integer(c_int64_t), value :: stride_C
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasDsyrkxStridedBatched(handle, uplo, transA, n, k, alpha,&
              A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count)
    end function hipblasDsyrkxStridedBatchedFortran

    function hipblasCsyrkxStridedBatchedFortran(handle, uplo, transA, n, k, alpha, &
            A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count) &
            result(res) &
            bind(c, name = 'hipblasCsyrkxStridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(c_int), value :: n
        integer(c_int), value :: k
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int64_t), value :: stride_A
        type(c_ptr), value :: B
        integer(c_int), value :: ldb
        integer(c_int64_t), value :: stride_B
        type(c_ptr), value :: beta
        type(c_ptr), value :: C
        integer(c_int), value :: ldc
        integer(c_int64_t), value :: stride_C
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasCsyrkxStridedBatched(handle, uplo, transA, n, k, alpha,&
              A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count)
    end function hipblasCsyrkxStridedBatchedFortran

    function hipblasZsyrkxStridedBatchedFortran(handle, uplo, transA, n, k, alpha, &
            A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count) &
            result(res) &
            bind(c, name = 'hipblasZsyrkxStridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(c_int), value :: n
        integer(c_int), value :: k
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int64_t), value :: stride_A
        type(c_ptr), value :: B
        integer(c_int), value :: ldb
        integer(c_int64_t), value :: stride_B
        type(c_ptr), value :: beta
        type(c_ptr), value :: C
        integer(c_int), value :: ldc
        integer(c_int64_t), value :: stride_C
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasZsyrkxStridedBatched(handle, uplo, transA, n, k, alpha,&
              A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count)
    end function hipblasZsyrkxStridedBatchedFortran

    ! trmm
    function hipblasStrmmFortran(handle, side, uplo, transA, diag, m, n, alpha, &
            A, lda, B, ldb) &
            result(res) &
            bind(c, name = 'hipblasStrmmFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_DIAG_UNIT)), value :: diag
        integer(c_int), value :: m
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: B
        integer(c_int), value :: ldb
        integer(c_int) :: res
        res = hipblasStrmm(handle, side, uplo, transA, diag, m, n, alpha,&
              A, lda, B, ldb)
    end function hipblasStrmmFortran

    function hipblasDtrmmFortran(handle, side, uplo, transA, diag, m, n, alpha, &
            A, lda, B, ldb) &
            result(res) &
            bind(c, name = 'hipblasDtrmmFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_DIAG_UNIT)), value :: diag
        integer(c_int), value :: m
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: B
        integer(c_int), value :: ldb
        integer(c_int) :: res
        res = hipblasDtrmm(handle, side, uplo, transA, diag, m, n, alpha,&
              A, lda, B, ldb)
    end function hipblasDtrmmFortran

    function hipblasCtrmmFortran(handle, side, uplo, transA, diag, m, n, alpha, &
            A, lda, B, ldb) &
            result(res) &
            bind(c, name = 'hipblasCtrmmFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_DIAG_UNIT)), value :: diag
        integer(c_int), value :: m
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: B
        integer(c_int), value :: ldb
        integer(c_int) :: res
        res = hipblasCtrmm(handle, side, uplo, transA, diag, m, n, alpha,&
              A, lda, B, ldb)
    end function hipblasCtrmmFortran

    function hipblasZtrmmFortran(handle, side, uplo, transA, diag, m, n, alpha, &
            A, lda, B, ldb) &
            result(res) &
            bind(c, name = 'hipblasZtrmmFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_DIAG_UNIT)), value :: diag
        integer(c_int), value :: m
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: B
        integer(c_int), value :: ldb
        integer(c_int) :: res
        res = hipblasZtrmm(handle, side, uplo, transA, diag, m, n, alpha,&
              A, lda, B, ldb)
    end function hipblasZtrmmFortran

    ! trmmBatched
    function hipblasStrmmBatchedFortran(handle, side, uplo, transA, diag, m, n, alpha, &
            A, lda, B, ldb, batch_count) &
            result(res) &
            bind(c, name = 'hipblasStrmmBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_DIAG_UNIT)), value :: diag
        integer(c_int), value :: m
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: B
        integer(c_int), value :: ldb
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasStrmmBatched(handle, side, uplo, transA, diag, m, n, alpha,&
              A, lda, B, ldb, batch_count)
    end function hipblasStrmmBatchedFortran

    function hipblasDtrmmBatchedFortran(handle, side, uplo, transA, diag, m, n, alpha, &
            A, lda, B, ldb, batch_count) &
            result(res) &
            bind(c, name = 'hipblasDtrmmBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_DIAG_UNIT)), value :: diag
        integer(c_int), value :: m
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: B
        integer(c_int), value :: ldb
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasDtrmmBatched(handle, side, uplo, transA, diag, m, n, alpha,&
              A, lda, B, ldb, batch_count)
    end function hipblasDtrmmBatchedFortran

    function hipblasCtrmmBatchedFortran(handle, side, uplo, transA, diag, m, n, alpha, &
            A, lda, B, ldb, batch_count) &
            result(res) &
            bind(c, name = 'hipblasCtrmmBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_DIAG_UNIT)), value :: diag
        integer(c_int), value :: m
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: B
        integer(c_int), value :: ldb
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasCtrmmBatched(handle, side, uplo, transA, diag, m, n, alpha,&
              A, lda, B, ldb, batch_count)
    end function hipblasCtrmmBatchedFortran

    function hipblasZtrmmBatchedFortran(handle, side, uplo, transA, diag, m, n, alpha, &
            A, lda, B, ldb, batch_count) &
            result(res) &
            bind(c, name = 'hipblasZtrmmBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_DIAG_UNIT)), value :: diag
        integer(c_int), value :: m
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: B
        integer(c_int), value :: ldb
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasZtrmmBatched(handle, side, uplo, transA, diag, m, n, alpha,&
              A, lda, B, ldb, batch_count)
    end function hipblasZtrmmBatchedFortran

    ! trmmStridedBatched
    function hipblasStrmmStridedBatchedFortran(handle, side, uplo, transA, diag, m, n, alpha, &
            A, lda, stride_A, B, ldb, stride_B, batch_count) &
            result(res) &
            bind(c, name = 'hipblasStrmmStridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_DIAG_UNIT)), value :: diag
        integer(c_int), value :: m
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int64_t), value :: stride_A
        type(c_ptr), value :: B
        integer(c_int), value :: ldb
        integer(c_int64_t), value :: stride_B
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasStrmmStridedBatched(handle, side, uplo, transA, diag, m, n, alpha,&
              A, lda, stride_A, B, ldb, stride_B, batch_count)
    end function hipblasStrmmStridedBatchedFortran

    function hipblasDtrmmStridedBatchedFortran(handle, side, uplo, transA, diag, m, n, alpha, &
            A, lda, stride_A, B, ldb, stride_B, batch_count) &
            result(res) &
            bind(c, name = 'hipblasDtrmmStridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_DIAG_UNIT)), value :: diag
        integer(c_int), value :: m
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int64_t), value :: stride_A
        type(c_ptr), value :: B
        integer(c_int), value :: ldb
        integer(c_int64_t), value :: stride_B
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasDtrmmStridedBatched(handle, side, uplo, transA, diag, m, n, alpha,&
              A, lda, stride_A, B, ldb, stride_B, batch_count)
    end function hipblasDtrmmStridedBatchedFortran

    function hipblasCtrmmStridedBatchedFortran(handle, side, uplo, transA, diag, m, n, alpha, &
            A, lda, stride_A, B, ldb, stride_B, batch_count) &
            result(res) &
            bind(c, name = 'hipblasCtrmmStridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_DIAG_UNIT)), value :: diag
        integer(c_int), value :: m
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int64_t), value :: stride_A
        type(c_ptr), value :: B
        integer(c_int), value :: ldb
        integer(c_int64_t), value :: stride_B
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasCtrmmStridedBatched(handle, side, uplo, transA, diag, m, n, alpha,&
              A, lda, stride_A, B, ldb, stride_B, batch_count)
    end function hipblasCtrmmStridedBatchedFortran

    function hipblasZtrmmStridedBatchedFortran(handle, side, uplo, transA, diag, m, n, alpha, &
            A, lda, stride_A, B, ldb, stride_B, batch_count) &
            result(res) &
            bind(c, name = 'hipblasZtrmmStridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_DIAG_UNIT)), value :: diag
        integer(c_int), value :: m
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int64_t), value :: stride_A
        type(c_ptr), value :: B
        integer(c_int), value :: ldb
        integer(c_int64_t), value :: stride_B
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasZtrmmStridedBatched(handle, side, uplo, transA, diag, m, n, alpha,&
              A, lda, stride_A, B, ldb, stride_B, batch_count)
    end function hipblasZtrmmStridedBatchedFortran

    ! trtri
    function hipblasStrtriFortran(handle, uplo, diag, n, &
            A, lda, invA, ldinvA) &
            result(res) &
            bind(c, name = 'hipblasStrtriFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_DIAG_UNIT)), value :: diag
        integer(c_int), value :: n
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: invA
        integer(c_int), value :: ldinvA
        integer(c_int) :: res
        res = hipblasStrtri(handle, uplo, diag, n,&
              A, lda, invA, ldinvA)
    end function hipblasStrtriFortran

    function hipblasDtrtriFortran(handle, uplo, diag, n, &
            A, lda, invA, ldinvA) &
            result(res) &
            bind(c, name = 'hipblasDtrtriFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_DIAG_UNIT)), value :: diag
        integer(c_int), value :: n
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: invA
        integer(c_int), value :: ldinvA
        integer(c_int) :: res
        res = hipblasDtrtri(handle, uplo, diag, n,&
              A, lda, invA, ldinvA)
    end function hipblasDtrtriFortran

    function hipblasCtrtriFortran(handle, uplo, diag, n, &
            A, lda, invA, ldinvA) &
            result(res) &
            bind(c, name = 'hipblasCtrtriFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_DIAG_UNIT)), value :: diag
        integer(c_int), value :: n
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: invA
        integer(c_int), value :: ldinvA
        integer(c_int) :: res
        res = hipblasCtrtri(handle, uplo, diag, n,&
              A, lda, invA, ldinvA)
    end function hipblasCtrtriFortran

    function hipblasZtrtriFortran(handle, uplo, diag, n, &
            A, lda, invA, ldinvA) &
            result(res) &
            bind(c, name = 'hipblasZtrtriFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_DIAG_UNIT)), value :: diag
        integer(c_int), value :: n
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: invA
        integer(c_int), value :: ldinvA
        integer(c_int) :: res
        res = hipblasZtrtri(handle, uplo, diag, n,&
              A, lda, invA, ldinvA)
    end function hipblasZtrtriFortran

    ! trtriBatched
    function hipblasStrtriBatchedFortran(handle, uplo, diag, n, &
            A, lda, invA, ldinvA, batch_count) &
            result(res) &
            bind(c, name = 'hipblasStrtriBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_DIAG_UNIT)), value :: diag
        integer(c_int), value :: n
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: invA
        integer(c_int), value :: ldinvA
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasStrtriBatched(handle, uplo, diag, n,&
              A, lda, invA, ldinvA, batch_count)
    end function hipblasStrtriBatchedFortran

    function hipblasDtrtriBatchedFortran(handle, uplo, diag, n, &
            A, lda, invA, ldinvA, batch_count) &
            result(res) &
            bind(c, name = 'hipblasDtrtriBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_DIAG_UNIT)), value :: diag
        integer(c_int), value :: n
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: invA
        integer(c_int), value :: ldinvA
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasDtrtriBatched(handle, uplo, diag, n,&
              A, lda, invA, ldinvA, batch_count)
    end function hipblasDtrtriBatchedFortran

    function hipblasCtrtriBatchedFortran(handle, uplo, diag, n, &
            A, lda, invA, ldinvA, batch_count) &
            result(res) &
            bind(c, name = 'hipblasCtrtriBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_DIAG_UNIT)), value :: diag
        integer(c_int), value :: n
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: invA
        integer(c_int), value :: ldinvA
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasCtrtriBatched(handle, uplo, diag, n,&
              A, lda, invA, ldinvA, batch_count)
    end function hipblasCtrtriBatchedFortran

    function hipblasZtrtriBatchedFortran(handle, uplo, diag, n, &
            A, lda, invA, ldinvA, batch_count) &
            result(res) &
            bind(c, name = 'hipblasZtrtriBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_DIAG_UNIT)), value :: diag
        integer(c_int), value :: n
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: invA
        integer(c_int), value :: ldinvA
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasZtrtriBatched(handle, uplo, diag, n,&
              A, lda, invA, ldinvA, batch_count)
    end function hipblasZtrtriBatchedFortran

    ! trtriStridedBatched
    function hipblasStrtriStridedBatchedFortran(handle, uplo, diag, n, &
            A, lda, stride_A, invA, ldinvA, stride_invA, batch_count) &
            result(res) &
            bind(c, name = 'hipblasStrtriStridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_DIAG_UNIT)), value :: diag
        integer(c_int), value :: n
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int64_t), value :: stride_A
        type(c_ptr), value :: invA
        integer(c_int), value :: ldinvA
        integer(c_int64_t), value :: stride_invA
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasStrtriStridedBatched(handle, uplo, diag, n,&
              A, lda, stride_A, invA, ldinvA, stride_invA, batch_count)
    end function hipblasStrtriStridedBatchedFortran

    function hipblasDtrtriStridedBatchedFortran(handle, uplo, diag, n, &
            A, lda, stride_A, invA, ldinvA, stride_invA, batch_count) &
            result(res) &
            bind(c, name = 'hipblasDtrtriStridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_DIAG_UNIT)), value :: diag
        integer(c_int), value :: n
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int64_t), value :: stride_A
        type(c_ptr), value :: invA
        integer(c_int), value :: ldinvA
        integer(c_int64_t), value :: stride_invA
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasDtrtriStridedBatched(handle, uplo, diag, n,&
              A, lda, stride_A, invA, ldinvA, stride_invA, batch_count)
    end function hipblasDtrtriStridedBatchedFortran

    function hipblasCtrtriStridedBatchedFortran(handle, uplo, diag, n, &
            A, lda, stride_A, invA, ldinvA, stride_invA, batch_count) &
            result(res) &
            bind(c, name = 'hipblasCtrtriStridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_DIAG_UNIT)), value :: diag
        integer(c_int), value :: n
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int64_t), value :: stride_A
        type(c_ptr), value :: invA
        integer(c_int), value :: ldinvA
        integer(c_int64_t), value :: stride_invA
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasCtrtriStridedBatched(handle, uplo, diag, n,&
              A, lda, stride_A, invA, ldinvA, stride_invA, batch_count)
    end function hipblasCtrtriStridedBatchedFortran

    function hipblasZtrtriStridedBatchedFortran(handle, uplo, diag, n, &
            A, lda, stride_A, invA, ldinvA, stride_invA, batch_count) &
            result(res) &
            bind(c, name = 'hipblasZtrtriStridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_DIAG_UNIT)), value :: diag
        integer(c_int), value :: n
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int64_t), value :: stride_A
        type(c_ptr), value :: invA
        integer(c_int), value :: ldinvA
        integer(c_int64_t), value :: stride_invA
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasZtrtriStridedBatched(handle, uplo, diag, n,&
              A, lda, stride_A, invA, ldinvA, stride_invA, batch_count)
    end function hipblasZtrtriStridedBatchedFortran

    ! trsm
    function hipblasStrsmFortran(handle, side, uplo, transA, diag, m, n, alpha, &
            A, lda, B, ldb) &
            result(res) &
            bind(c, name = 'hipblasStrsmFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_DIAG_UNIT)), value :: diag
        integer(c_int), value :: m
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: B
        integer(c_int), value :: ldb
        integer(c_int) :: res
        res = hipblasStrsm(handle, side, uplo, transA, diag, m, n, alpha,&
              A, lda, B, ldb)
    end function hipblasStrsmFortran

    function hipblasDtrsmFortran(handle, side, uplo, transA, diag, m, n, alpha, &
            A, lda, B, ldb) &
            result(res) &
            bind(c, name = 'hipblasDtrsmFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_DIAG_UNIT)), value :: diag
        integer(c_int), value :: m
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: B
        integer(c_int), value :: ldb
        integer(c_int) :: res
        res = hipblasDtrsm(handle, side, uplo, transA, diag, m, n, alpha,&
              A, lda, B, ldb)
    end function hipblasDtrsmFortran

    function hipblasCtrsmFortran(handle, side, uplo, transA, diag, m, n, alpha, &
            A, lda, B, ldb) &
            result(res) &
            bind(c, name = 'hipblasCtrsmFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_DIAG_UNIT)), value :: diag
        integer(c_int), value :: m
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: B
        integer(c_int), value :: ldb
        integer(c_int) :: res
        res = hipblasCtrsm(handle, side, uplo, transA, diag, m, n, alpha,&
              A, lda, B, ldb)
    end function hipblasCtrsmFortran

    function hipblasZtrsmFortran(handle, side, uplo, transA, diag, m, n, alpha, &
            A, lda, B, ldb) &
            result(res) &
            bind(c, name = 'hipblasZtrsmFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_DIAG_UNIT)), value :: diag
        integer(c_int), value :: m
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: B
        integer(c_int), value :: ldb
        integer(c_int) :: res
        res = hipblasZtrsm(handle, side, uplo, transA, diag, m, n, alpha,&
              A, lda, B, ldb)
    end function hipblasZtrsmFortran

    ! trsmBatched
    function hipblasStrsmBatchedFortran(handle, side, uplo, transA, diag, m, n, alpha, &
            A, lda, B, ldb, batch_count) &
            result(res) &
            bind(c, name = 'hipblasStrsmBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_DIAG_UNIT)), value :: diag
        integer(c_int), value :: m
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: B
        integer(c_int), value :: ldb
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasStrsmBatched(handle, side, uplo, transA, diag, m, n, alpha,&
              A, lda, B, ldb, batch_count)
    end function hipblasStrsmBatchedFortran

    function hipblasDtrsmBatchedFortran(handle, side, uplo, transA, diag, m, n, alpha, &
            A, lda, B, ldb, batch_count) &
            result(res) &
            bind(c, name = 'hipblasDtrsmBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_DIAG_UNIT)), value :: diag
        integer(c_int), value :: m
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: B
        integer(c_int), value :: ldb
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasDtrsmBatched(handle, side, uplo, transA, diag, m, n, alpha,&
              A, lda, B, ldb, batch_count)
    end function hipblasDtrsmBatchedFortran

    function hipblasCtrsmBatchedFortran(handle, side, uplo, transA, diag, m, n, alpha, &
            A, lda, B, ldb, batch_count) &
            result(res) &
            bind(c, name = 'hipblasCtrsmBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_DIAG_UNIT)), value :: diag
        integer(c_int), value :: m
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: B
        integer(c_int), value :: ldb
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasCtrsmBatched(handle, side, uplo, transA, diag, m, n, alpha,&
              A, lda, B, ldb, batch_count)
    end function hipblasCtrsmBatchedFortran

    function hipblasZtrsmBatchedFortran(handle, side, uplo, transA, diag, m, n, alpha, &
            A, lda, B, ldb, batch_count) &
            result(res) &
            bind(c, name = 'hipblasZtrsmBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_DIAG_UNIT)), value :: diag
        integer(c_int), value :: m
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: B
        integer(c_int), value :: ldb
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasZtrsmBatched(handle, side, uplo, transA, diag, m, n, alpha,&
              A, lda, B, ldb, batch_count)
    end function hipblasZtrsmBatchedFortran

    ! trsmStridedBatched
    function hipblasStrsmStridedBatchedFortran(handle, side, uplo, transA, diag, m, n, alpha, &
            A, lda, stride_A, B, ldb, stride_B, batch_count) &
            result(res) &
            bind(c, name = 'hipblasStrsmStridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_DIAG_UNIT)), value :: diag
        integer(c_int), value :: m
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int64_t), value :: stride_A
        type(c_ptr), value :: B
        integer(c_int), value :: ldb
        integer(c_int64_t), value :: stride_B
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasStrsmStridedBatched(handle, side, uplo, transA, diag, m, n, alpha,&
              A, lda, stride_A, B, ldb, stride_B, batch_count)
    end function hipblasStrsmStridedBatchedFortran

    function hipblasDtrsmStridedBatchedFortran(handle, side, uplo, transA, diag, m, n, alpha, &
            A, lda, stride_A, B, ldb, stride_B, batch_count) &
            result(res) &
            bind(c, name = 'hipblasDtrsmStridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_DIAG_UNIT)), value :: diag
        integer(c_int), value :: m
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int64_t), value :: stride_A
        type(c_ptr), value :: B
        integer(c_int), value :: ldb
        integer(c_int64_t), value :: stride_B
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasDtrsmStridedBatched(handle, side, uplo, transA, diag, m, n, alpha,&
              A, lda, stride_A, B, ldb, stride_B, batch_count)
    end function hipblasDtrsmStridedBatchedFortran

    function hipblasCtrsmStridedBatchedFortran(handle, side, uplo, transA, diag, m, n, alpha, &
            A, lda, stride_A, B, ldb, stride_B, batch_count) &
            result(res) &
            bind(c, name = 'hipblasCtrsmStridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_DIAG_UNIT)), value :: diag
        integer(c_int), value :: m
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int64_t), value :: stride_A
        type(c_ptr), value :: B
        integer(c_int), value :: ldb
        integer(c_int64_t), value :: stride_B
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasCtrsmStridedBatched(handle, side, uplo, transA, diag, m, n, alpha,&
              A, lda, stride_A, B, ldb, stride_B, batch_count)
    end function hipblasCtrsmStridedBatchedFortran

    function hipblasZtrsmStridedBatchedFortran(handle, side, uplo, transA, diag, m, n, alpha, &
            A, lda, stride_A, B, ldb, stride_B, batch_count) &
            result(res) &
            bind(c, name = 'hipblasZtrsmStridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
        integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_DIAG_UNIT)), value :: diag
        integer(c_int), value :: m
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int64_t), value :: stride_A
        type(c_ptr), value :: B
        integer(c_int), value :: ldb
        integer(c_int64_t), value :: stride_B
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasZtrsmStridedBatched(handle, side, uplo, transA, diag, m, n, alpha,&
              A, lda, stride_A, B, ldb, stride_B, batch_count)
    end function hipblasZtrsmStridedBatchedFortran

    ! gemm
    function hipblasHgemmFortran(handle, transA, transB, m, n, k, alpha, &
            A, lda, B, ldb, beta, C, ldc) &
            result(res) &
            bind(c, name = 'hipblasHgemmFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_OP_N)), value :: transB
        integer(c_int), value :: m
        integer(c_int), value :: n
        integer(c_int), value :: k
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: B
        integer(c_int), value :: ldb
        type(c_ptr), value :: beta
        type(c_ptr), value :: C
        integer(c_int), value :: ldc
        integer(c_int) :: res
        res = hipblasHgemm(handle, transA, transB, m, n, k, alpha,&
              A, lda, B, ldb, beta, C, ldc)
    end function hipblasHgemmFortran

    function hipblasSgemmFortran(handle, transA, transB, m, n, k, alpha, &
            A, lda, B, ldb, beta, C, ldc) &
            result(res) &
            bind(c, name = 'hipblasSgemmFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_OP_N)), value :: transB
        integer(c_int), value :: m
        integer(c_int), value :: n
        integer(c_int), value :: k
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: B
        integer(c_int), value :: ldb
        type(c_ptr), value :: beta
        type(c_ptr), value :: C
        integer(c_int), value :: ldc
        integer(c_int) :: res
        res = hipblasSgemm(handle, transA, transB, m, n, k, alpha,&
              A, lda, B, ldb, beta, C, ldc)
    end function hipblasSgemmFortran

    function hipblasDgemmFortran(handle, transA, transB, m, n, k, alpha, &
            A, lda, B, ldb, beta, C, ldc) &
            result(res) &
            bind(c, name = 'hipblasDgemmFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_OP_N)), value :: transB
        integer(c_int), value :: m
        integer(c_int), value :: n
        integer(c_int), value :: k
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: B
        integer(c_int), value :: ldb
        type(c_ptr), value :: beta
        type(c_ptr), value :: C
        integer(c_int), value :: ldc
        integer(c_int) :: res
        res = hipblasDgemm(handle, transA, transB, m, n, k, alpha,&
              A, lda, B, ldb, beta, C, ldc)
    end function hipblasDgemmFortran

    function hipblasCgemmFortran(handle, transA, transB, m, n, k, alpha, &
            A, lda, B, ldb, beta, C, ldc) &
            result(res) &
            bind(c, name = 'hipblasCgemmFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_OP_N)), value :: transB
        integer(c_int), value :: m
        integer(c_int), value :: n
        integer(c_int), value :: k
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: B
        integer(c_int), value :: ldb
        type(c_ptr), value :: beta
        type(c_ptr), value :: C
        integer(c_int), value :: ldc
        integer(c_int) :: res
        res = hipblasCgemm(handle, transA, transB, m, n, k, alpha,&
              A, lda, B, ldb, beta, C, ldc)
    end function hipblasCgemmFortran

    function hipblasZgemmFortran(handle, transA, transB, m, n, k, alpha, &
            A, lda, B, ldb, beta, C, ldc) &
            result(res) &
            bind(c, name = 'hipblasZgemmFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_OP_N)), value :: transB
        integer(c_int), value :: m
        integer(c_int), value :: n
        integer(c_int), value :: k
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: B
        integer(c_int), value :: ldb
        type(c_ptr), value :: beta
        type(c_ptr), value :: C
        integer(c_int), value :: ldc
        integer(c_int) :: res
        res = hipblasZgemm(handle, transA, transB, m, n, k, alpha,&
              A, lda, B, ldb, beta, C, ldc)
    end function hipblasZgemmFortran

    ! gemmBatched
    function hipblasHgemmBatchedFortran(handle, transA, transB, m, n, k, alpha, &
            A, lda, B, ldb, beta, C, ldc, batch_count) &
            result(res) &
            bind(c, name = 'hipblasHgemmBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_OP_N)), value :: transB
        integer(c_int), value :: m
        integer(c_int), value :: n
        integer(c_int), value :: k
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: B
        integer(c_int), value :: ldb
        type(c_ptr), value :: beta
        type(c_ptr), value :: C
        integer(c_int), value :: ldc
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasHgemmBatched(handle, transA, transB, m, n, k, alpha,&
              A, lda, B, ldb, beta, C, ldc, batch_count)
    end function hipblasHgemmBatchedFortran

    function hipblasSgemmBatchedFortran(handle, transA, transB, m, n, k, alpha, &
            A, lda, B, ldb, beta, C, ldc, batch_count) &
            result(res) &
            bind(c, name = 'hipblasSgemmBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_OP_N)), value :: transB
        integer(c_int), value :: m
        integer(c_int), value :: n
        integer(c_int), value :: k
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: B
        integer(c_int), value :: ldb
        type(c_ptr), value :: beta
        type(c_ptr), value :: C
        integer(c_int), value :: ldc
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasSgemmBatched(handle, transA, transB, m, n, k, alpha,&
              A, lda, B, ldb, beta, C, ldc, batch_count)
    end function hipblasSgemmBatchedFortran

    function hipblasDgemmBatchedFortran(handle, transA, transB, m, n, k, alpha, &
            A, lda, B, ldb, beta, C, ldc, batch_count) &
            result(res) &
            bind(c, name = 'hipblasDgemmBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_OP_N)), value :: transB
        integer(c_int), value :: m
        integer(c_int), value :: n
        integer(c_int), value :: k
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: B
        integer(c_int), value :: ldb
        type(c_ptr), value :: beta
        type(c_ptr), value :: C
        integer(c_int), value :: ldc
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasDgemmBatched(handle, transA, transB, m, n, k, alpha,&
              A, lda, B, ldb, beta, C, ldc, batch_count)
    end function hipblasDgemmBatchedFortran

    function hipblasCgemmBatchedFortran(handle, transA, transB, m, n, k, alpha, &
            A, lda, B, ldb, beta, C, ldc, batch_count) &
            result(res) &
            bind(c, name = 'hipblasCgemmBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_OP_N)), value :: transB
        integer(c_int), value :: m
        integer(c_int), value :: n
        integer(c_int), value :: k
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: B
        integer(c_int), value :: ldb
        type(c_ptr), value :: beta
        type(c_ptr), value :: C
        integer(c_int), value :: ldc
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasCgemmBatched(handle, transA, transB, m, n, k, alpha,&
              A, lda, B, ldb, beta, C, ldc, batch_count)
    end function hipblasCgemmBatchedFortran

    function hipblasZgemmBatchedFortran(handle, transA, transB, m, n, k, alpha, &
            A, lda, B, ldb, beta, C, ldc, batch_count) &
            result(res) &
            bind(c, name = 'hipblasZgemmBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_OP_N)), value :: transB
        integer(c_int), value :: m
        integer(c_int), value :: n
        integer(c_int), value :: k
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: B
        integer(c_int), value :: ldb
        type(c_ptr), value :: beta
        type(c_ptr), value :: C
        integer(c_int), value :: ldc
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasZgemmBatched(handle, transA, transB, m, n, k, alpha,&
              A, lda, B, ldb, beta, C, ldc, batch_count)
    end function hipblasZgemmBatchedFortran

    ! gemmStridedBatched
    function hipblasHgemmStridedBatchedFortran(handle, transA, transB, m, n, k, alpha, &
            A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count) &
            result(res) &
            bind(c, name = 'hipblasHgemmStridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_OP_N)), value :: transB
        integer(c_int), value :: m
        integer(c_int), value :: n
        integer(c_int), value :: k
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int64_t), value :: stride_A
        type(c_ptr), value :: B
        integer(c_int), value :: ldb
        integer(c_int64_t), value :: stride_B
        type(c_ptr), value :: beta
        type(c_ptr), value :: C
        integer(c_int), value :: ldc
        integer(c_int64_t), value :: stride_C
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasHgemmStridedBatched(handle, transA, transB, m, n, k, alpha,&
              A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count)
    end function hipblasHgemmStridedBatchedFortran

    function hipblasSgemmStridedBatchedFortran(handle, transA, transB, m, n, k, alpha, &
            A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count) &
            result(res) &
            bind(c, name = 'hipblasSgemmStridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_OP_N)), value :: transB
        integer(c_int), value :: m
        integer(c_int), value :: n
        integer(c_int), value :: k
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int64_t), value :: stride_A
        type(c_ptr), value :: B
        integer(c_int), value :: ldb
        integer(c_int64_t), value :: stride_B
        type(c_ptr), value :: beta
        type(c_ptr), value :: C
        integer(c_int), value :: ldc
        integer(c_int64_t), value :: stride_C
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasSgemmStridedBatched(handle, transA, transB, m, n, k, alpha,&
              A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count)
    end function hipblasSgemmStridedBatchedFortran

    function hipblasDgemmStridedBatchedFortran(handle, transA, transB, m, n, k, alpha, &
            A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count) &
            result(res) &
            bind(c, name = 'hipblasDgemmStridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_OP_N)), value :: transB
        integer(c_int), value :: m
        integer(c_int), value :: n
        integer(c_int), value :: k
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int64_t), value :: stride_A
        type(c_ptr), value :: B
        integer(c_int), value :: ldb
        integer(c_int64_t), value :: stride_B
        type(c_ptr), value :: beta
        type(c_ptr), value :: C
        integer(c_int), value :: ldc
        integer(c_int64_t), value :: stride_C
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasDgemmStridedBatched(handle, transA, transB, m, n, k, alpha,&
              A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count)
    end function hipblasDgemmStridedBatchedFortran

    function hipblasCgemmStridedBatchedFortran(handle, transA, transB, m, n, k, alpha, &
            A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count) &
            result(res) &
            bind(c, name = 'hipblasCgemmStridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_OP_N)), value :: transB
        integer(c_int), value :: m
        integer(c_int), value :: n
        integer(c_int), value :: k
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int64_t), value :: stride_A
        type(c_ptr), value :: B
        integer(c_int), value :: ldb
        integer(c_int64_t), value :: stride_B
        type(c_ptr), value :: beta
        type(c_ptr), value :: C
        integer(c_int), value :: ldc
        integer(c_int64_t), value :: stride_C
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasCgemmStridedBatched(handle, transA, transB, m, n, k, alpha,&
              A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count)
    end function hipblasCgemmStridedBatchedFortran

    function hipblasZgemmStridedBatchedFortran(handle, transA, transB, m, n, k, alpha, &
            A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count) &
            result(res) &
            bind(c, name = 'hipblasZgemmStridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_OP_N)), value :: transB
        integer(c_int), value :: m
        integer(c_int), value :: n
        integer(c_int), value :: k
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int64_t), value :: stride_A
        type(c_ptr), value :: B
        integer(c_int), value :: ldb
        integer(c_int64_t), value :: stride_B
        type(c_ptr), value :: beta
        type(c_ptr), value :: C
        integer(c_int), value :: ldc
        integer(c_int64_t), value :: stride_C
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasZgemmStridedBatched(handle, transA, transB, m, n, k, alpha,&
              A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count)
    end function hipblasZgemmStridedBatchedFortran

    ! dgmm
    function hipblasSdgmmFortran(handle, side, m, n, &
            A, lda, x, incx, C, ldc) &
            result(res) &
            bind(c, name = 'hipblasSdgmmFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
        integer(c_int), value :: m
        integer(c_int), value :: n
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: C
        integer(c_int), value :: ldc
        integer(c_int) :: res
        res = hipblasSdgmm(handle, side, m, n,&
              A, lda, x, incx, C, ldc)
    end function hipblasSdgmmFortran

    function hipblasDdgmmFortran(handle, side, m, n, &
            A, lda, x, incx, C, ldc) &
            result(res) &
            bind(c, name = 'hipblasDdgmmFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
        integer(c_int), value :: m
        integer(c_int), value :: n
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: C
        integer(c_int), value :: ldc
        integer(c_int) :: res
        res = hipblasDdgmm(handle, side, m, n,&
              A, lda, x, incx, C, ldc)
    end function hipblasDdgmmFortran

    function hipblasCdgmmFortran(handle, side, m, n, &
            A, lda, x, incx, C, ldc) &
            result(res) &
            bind(c, name = 'hipblasCdgmmFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
        integer(c_int), value :: m
        integer(c_int), value :: n
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: C
        integer(c_int), value :: ldc
        integer(c_int) :: res
        res = hipblasCdgmm(handle, side, m, n,&
              A, lda, x, incx, C, ldc)
    end function hipblasCdgmmFortran

    function hipblasZdgmmFortran(handle, side, m, n, &
            A, lda, x, incx, C, ldc) &
            result(res) &
            bind(c, name = 'hipblasZdgmmFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
        integer(c_int), value :: m
        integer(c_int), value :: n
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: C
        integer(c_int), value :: ldc
        integer(c_int) :: res
        res = hipblasZdgmm(handle, side, m, n,&
              A, lda, x, incx, C, ldc)
    end function hipblasZdgmmFortran

    ! dgmmBatched
    function hipblasSdgmmBatchedFortran(handle, side, m, n, &
            A, lda, x, incx, C, ldc, batch_count) &
            result(res) &
            bind(c, name = 'hipblasSdgmmBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
        integer(c_int), value :: m
        integer(c_int), value :: n
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: C
        integer(c_int), value :: ldc
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasSdgmmBatched(handle, side, m, n,&
              A, lda, x, incx, C, ldc, batch_count)
    end function hipblasSdgmmBatchedFortran

    function hipblasDdgmmBatchedFortran(handle, side, m, n, &
            A, lda, x, incx, C, ldc, batch_count) &
            result(res) &
            bind(c, name = 'hipblasDdgmmBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
        integer(c_int), value :: m
        integer(c_int), value :: n
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: C
        integer(c_int), value :: ldc
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasDdgmmBatched(handle, side, m, n,&
              A, lda, x, incx, C, ldc, batch_count)
    end function hipblasDdgmmBatchedFortran

    function hipblasCdgmmBatchedFortran(handle, side, m, n, &
            A, lda, x, incx, C, ldc, batch_count) &
            result(res) &
            bind(c, name = 'hipblasCdgmmBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
        integer(c_int), value :: m
        integer(c_int), value :: n
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: C
        integer(c_int), value :: ldc
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasCdgmmBatched(handle, side, m, n,&
              A, lda, x, incx, C, ldc, batch_count)
    end function hipblasCdgmmBatchedFortran

    function hipblasZdgmmBatchedFortran(handle, side, m, n, &
            A, lda, x, incx, C, ldc, batch_count) &
            result(res) &
            bind(c, name = 'hipblasZdgmmBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
        integer(c_int), value :: m
        integer(c_int), value :: n
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: C
        integer(c_int), value :: ldc
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasZdgmmBatched(handle, side, m, n,&
              A, lda, x, incx, C, ldc, batch_count)
    end function hipblasZdgmmBatchedFortran

    ! dgmmStridedBatched
    function hipblasSdgmmStridedBatchedFortran(handle, side, m, n, &
            A, lda, stride_A, x, incx, stride_x, C, ldc, stride_C, batch_count) &
            result(res) &
            bind(c, name = 'hipblasSdgmmStridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
        integer(c_int), value :: m
        integer(c_int), value :: n
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int64_t), value :: stride_A
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        type(c_ptr), value :: C
        integer(c_int), value :: ldc
        integer(c_int64_t), value :: stride_C
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasSdgmmStridedBatched(handle, side, m, n,&
              A, lda, stride_A, x, incx, stride_x, C, ldc, stride_C, batch_count)
    end function hipblasSdgmmStridedBatchedFortran

    function hipblasDdgmmStridedBatchedFortran(handle, side, m, n, &
            A, lda, stride_A, x, incx, stride_x, C, ldc, stride_C, batch_count) &
            result(res) &
            bind(c, name = 'hipblasDdgmmStridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
        integer(c_int), value :: m
        integer(c_int), value :: n
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int64_t), value :: stride_A
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        type(c_ptr), value :: C
        integer(c_int), value :: ldc
        integer(c_int64_t), value :: stride_C
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasDdgmmStridedBatched(handle, side, m, n,&
              A, lda, stride_A, x, incx, stride_x, C, ldc, stride_C, batch_count)
    end function hipblasDdgmmStridedBatchedFortran

    function hipblasCdgmmStridedBatchedFortran(handle, side, m, n, &
            A, lda, stride_A, x, incx, stride_x, C, ldc, stride_C, batch_count) &
            result(res) &
            bind(c, name = 'hipblasCdgmmStridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
        integer(c_int), value :: m
        integer(c_int), value :: n
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int64_t), value :: stride_A
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        type(c_ptr), value :: C
        integer(c_int), value :: ldc
        integer(c_int64_t), value :: stride_C
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasCdgmmStridedBatched(handle, side, m, n,&
              A, lda, stride_A, x, incx, stride_x, C, ldc, stride_C, batch_count)
    end function hipblasCdgmmStridedBatchedFortran

    function hipblasZdgmmStridedBatchedFortran(handle, side, m, n, &
            A, lda, stride_A, x, incx, stride_x, C, ldc, stride_C, batch_count) &
            result(res) &
            bind(c, name = 'hipblasZdgmmStridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
        integer(c_int), value :: m
        integer(c_int), value :: n
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int64_t), value :: stride_A
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        type(c_ptr), value :: C
        integer(c_int), value :: ldc
        integer(c_int64_t), value :: stride_C
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasZdgmmStridedBatched(handle, side, m, n,&
              A, lda, stride_A, x, incx, stride_x, C, ldc, stride_C, batch_count)
    end function hipblasZdgmmStridedBatchedFortran

    ! geam
    function hipblasSgeamFortran(handle, transA, transB, m, n, alpha, &
            A, lda, beta, B, ldb, C, ldc) &
            result(res) &
            bind(c, name = 'hipblasSgeamFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_OP_N)), value :: transB
        integer(c_int), value :: m
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: beta
        type(c_ptr), value :: B
        integer(c_int), value :: ldb
        type(c_ptr), value :: C
        integer(c_int), value :: ldc
        integer(c_int) :: res
        res = hipblasSgeam(handle, transA, transB, m, n, alpha,&
              A, lda, beta, B, ldb, C, ldc)
    end function hipblasSgeamFortran

    function hipblasDgeamFortran(handle, transA, transB, m, n, alpha, &
            A, lda, beta, B, ldb, C, ldc) &
            result(res) &
            bind(c, name = 'hipblasDgeamFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_OP_N)), value :: transB
        integer(c_int), value :: m
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: beta
        type(c_ptr), value :: B
        integer(c_int), value :: ldb
        type(c_ptr), value :: C
        integer(c_int), value :: ldc
        integer(c_int) :: res
        res = hipblasDgeam(handle, transA, transB, m, n, alpha,&
              A, lda, beta, B, ldb, C, ldc)
    end function hipblasDgeamFortran

    function hipblasCgeamFortran(handle, transA, transB, m, n, alpha, &
            A, lda, beta, B, ldb, C, ldc) &
            result(res) &
            bind(c, name = 'hipblasCgeamFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_OP_N)), value :: transB
        integer(c_int), value :: m
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: beta
        type(c_ptr), value :: B
        integer(c_int), value :: ldb
        type(c_ptr), value :: C
        integer(c_int), value :: ldc
        integer(c_int) :: res
        res = hipblasCgeam(handle, transA, transB, m, n, alpha,&
              A, lda, beta, B, ldb, C, ldc)
    end function hipblasCgeamFortran

    function hipblasZgeamFortran(handle, transA, transB, m, n, alpha, &
            A, lda, beta, B, ldb, C, ldc) &
            result(res) &
            bind(c, name = 'hipblasZgeamFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_OP_N)), value :: transB
        integer(c_int), value :: m
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: beta
        type(c_ptr), value :: B
        integer(c_int), value :: ldb
        type(c_ptr), value :: C
        integer(c_int), value :: ldc
        integer(c_int) :: res
        res = hipblasZgeam(handle, transA, transB, m, n, alpha,&
              A, lda, beta, B, ldb, C, ldc)
    end function hipblasZgeamFortran

    ! geamBatched
    function hipblasSgeamBatchedFortran(handle, transA, transB, m, n, alpha, &
            A, lda, beta, B, ldb, C, ldc, batch_count) &
            result(res) &
            bind(c, name = 'hipblasSgeamBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_OP_N)), value :: transB
        integer(c_int), value :: m
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: beta
        type(c_ptr), value :: B
        integer(c_int), value :: ldb
        type(c_ptr), value :: C
        integer(c_int), value :: ldc
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasSgeamBatched(handle, transA, transB, m, n, alpha,&
              A, lda, beta, B, ldb, C, ldc, batch_count)
    end function hipblasSgeamBatchedFortran

    function hipblasDgeamBatchedFortran(handle, transA, transB, m, n, alpha, &
            A, lda, beta, B, ldb, C, ldc, batch_count) &
            result(res) &
            bind(c, name = 'hipblasDgeamBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_OP_N)), value :: transB
        integer(c_int), value :: m
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: beta
        type(c_ptr), value :: B
        integer(c_int), value :: ldb
        type(c_ptr), value :: C
        integer(c_int), value :: ldc
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasDgeamBatched(handle, transA, transB, m, n, alpha,&
              A, lda, beta, B, ldb, C, ldc, batch_count)
    end function hipblasDgeamBatchedFortran

    function hipblasCgeamBatchedFortran(handle, transA, transB, m, n, alpha, &
            A, lda, beta, B, ldb, C, ldc, batch_count) &
            result(res) &
            bind(c, name = 'hipblasCgeamBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_OP_N)), value :: transB
        integer(c_int), value :: m
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: beta
        type(c_ptr), value :: B
        integer(c_int), value :: ldb
        type(c_ptr), value :: C
        integer(c_int), value :: ldc
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasCgeamBatched(handle, transA, transB, m, n, alpha,&
              A, lda, beta, B, ldb, C, ldc, batch_count)
    end function hipblasCgeamBatchedFortran

    function hipblasZgeamBatchedFortran(handle, transA, transB, m, n, alpha, &
            A, lda, beta, B, ldb, C, ldc, batch_count) &
            result(res) &
            bind(c, name = 'hipblasZgeamBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_OP_N)), value :: transB
        integer(c_int), value :: m
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: beta
        type(c_ptr), value :: B
        integer(c_int), value :: ldb
        type(c_ptr), value :: C
        integer(c_int), value :: ldc
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasZgeamBatched(handle, transA, transB, m, n, alpha,&
              A, lda, beta, B, ldb, C, ldc, batch_count)
    end function hipblasZgeamBatchedFortran

    ! geamStridedBatched
    function hipblasSgeamStridedBatchedFortran(handle, transA, transB, m, n, alpha, &
            A, lda, stride_A, beta, B, ldb, stride_B, C, ldc, stride_C, batch_count) &
            result(res) &
            bind(c, name = 'hipblasSgeamStridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_OP_N)), value :: transB
        integer(c_int), value :: m
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int64_t), value :: stride_A
        type(c_ptr), value :: beta
        type(c_ptr), value :: B
        integer(c_int), value :: ldb
        integer(c_int64_t), value :: stride_B
        type(c_ptr), value :: C
        integer(c_int), value :: ldc
        integer(c_int64_t), value :: stride_C
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasSgeamStridedBatched(handle, transA, transB, m, n, alpha,&
              A, lda, stride_A, beta, B, ldb, stride_B, C, ldc, stride_C, batch_count)
    end function hipblasSgeamStridedBatchedFortran

    function hipblasDgeamStridedBatchedFortran(handle, transA, transB, m, n, alpha, &
            A, lda, stride_A, beta, B, ldb, stride_B, C, ldc, stride_C, batch_count) &
            result(res) &
            bind(c, name = 'hipblasDgeamStridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_OP_N)), value :: transB
        integer(c_int), value :: m
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int64_t), value :: stride_A
        type(c_ptr), value :: beta
        type(c_ptr), value :: B
        integer(c_int), value :: ldb
        integer(c_int64_t), value :: stride_B
        type(c_ptr), value :: C
        integer(c_int), value :: ldc
        integer(c_int64_t), value :: stride_C
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasDgeamStridedBatched(handle, transA, transB, m, n, alpha,&
              A, lda, stride_A, beta, B, ldb, stride_B, C, ldc, stride_C, batch_count)
    end function hipblasDgeamStridedBatchedFortran

    function hipblasCgeamStridedBatchedFortran(handle, transA, transB, m, n, alpha, &
            A, lda, stride_A, beta, B, ldb, stride_B, C, ldc, stride_C, batch_count) &
            result(res) &
            bind(c, name = 'hipblasCgeamStridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_OP_N)), value :: transB
        integer(c_int), value :: m
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int64_t), value :: stride_A
        type(c_ptr), value :: beta
        type(c_ptr), value :: B
        integer(c_int), value :: ldb
        integer(c_int64_t), value :: stride_B
        type(c_ptr), value :: C
        integer(c_int), value :: ldc
        integer(c_int64_t), value :: stride_C
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasCgeamStridedBatched(handle, transA, transB, m, n, alpha,&
              A, lda, stride_A, beta, B, ldb, stride_B, C, ldc, stride_C, batch_count)
    end function hipblasCgeamStridedBatchedFortran

    function hipblasZgeamStridedBatchedFortran(handle, transA, transB, m, n, alpha, &
            A, lda, stride_A, beta, B, ldb, stride_B, C, ldc, stride_C, batch_count) &
            result(res) &
            bind(c, name = 'hipblasZgeamStridedBatchedFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_OP_N)), value :: transB
        integer(c_int), value :: m
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int64_t), value :: stride_A
        type(c_ptr), value :: beta
        type(c_ptr), value :: B
        integer(c_int), value :: ldb
        integer(c_int64_t), value :: stride_B
        type(c_ptr), value :: C
        integer(c_int), value :: ldc
        integer(c_int64_t), value :: stride_C
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = hipblasZgeamStridedBatched(handle, transA, transB, m, n, alpha,&
              A, lda, stride_A, beta, B, ldb, stride_B, C, ldc, stride_C, batch_count)
    end function hipblasZgeamStridedBatchedFortran

    !-----------------!
    ! blas Extensions !
    !-----------------!

    ! gemmEx
    function hipblasGemmExFortran(handle, transA, transB, m, n, k, alpha, a, a_type, lda, &
            b, b_type, ldb, beta, c, c_type, ldc,&
            compute_type, algo, solution_index, flags) &
            result(res) &
            bind(c, name = 'hipblasGemmExFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_OP_N)), value :: transB
        integer(c_int), value :: m
        integer(c_int), value :: n
        integer(c_int), value :: k
        type(c_ptr), value :: alpha
        type(c_ptr), value :: a
        integer(kind(HIPBLAS_R_16F)), value :: a_type
        integer(c_int), value :: lda
        type(c_ptr), value :: b
        integer(kind(HIPBLAS_R_16F)), value :: b_type
        integer(c_int), value :: ldb
        type(c_ptr), value :: beta
        type(c_ptr), value :: c
        integer(kind(HIPBLAS_R_16F)), value :: c_type
        integer(c_int), value :: ldc
        integer(kind(HIPBLAS_R_16F)), value :: compute_type
        integer(kind(HIPBLAS_GEMM_DEFAULT)), value :: algo
        integer(c_int32_t), value :: solution_index
        ! No unsigned types in fortran. If larger values are needed
        ! we will need a workaround.
        integer(c_int32_t), value :: flags
        integer(c_int) :: res
        res = hipblasGemmEx(handle, transA, transB, m, n, k, alpha,&
              a, a_type, lda, b, b_type, ldb, beta, c, c_type, ldc,&
              compute_type, algo, solution_index, flags)
    end function hipblasGemmExFortran

    function hipblasGemmBatchedExFortran(handle, transA, transB, m, n, k, alpha, a, a_type, lda, &
            b, b_type, ldb, beta, c, c_type, ldc,&
            batch_count, compute_type, algo, solution_index, flags) &
            result(res) &
            bind(c, name = 'hipblasGemmBatchedExFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_OP_N)), value :: transB
        integer(c_int), value :: m
        integer(c_int), value :: n
        integer(c_int), value :: k
        type(c_ptr), value :: alpha
        type(c_ptr), value :: a
        integer(kind(HIPBLAS_R_16F)), value :: a_type
        integer(c_int), value :: lda
        type(c_ptr), value :: b
        integer(kind(HIPBLAS_R_16F)), value :: b_type
        integer(c_int), value :: ldb
        type(c_ptr), value :: beta
        type(c_ptr), value :: c
        integer(kind(HIPBLAS_R_16F)), value :: c_type
        integer(c_int), value :: ldc
        integer(c_int), value :: batch_count
        integer(kind(HIPBLAS_R_16F)), value :: compute_type
        integer(kind(HIPBLAS_GEMM_DEFAULT)), value :: algo
        integer(c_int32_t), value :: solution_index
        ! No unsigned types in fortran. If larger values are needed
        ! we will need a workaround.
        integer(c_int32_t), value :: flags
        integer(c_int) :: res
        res = hipblasGemmBatchedEx(handle, transA, transB, m, n, k, alpha,&
              a, a_type, lda, b, b_type, ldb, beta, c, c_type, ldc,&
              batch_count, compute_type, algo, solution_index, flags)
    end function hipblasGemmBatchedExFortran

    function hipblasGemmStridedBatchedExFortran(handle, transA, transB, m, n, k, alpha, a, a_type, lda, stride_a, &
            b, b_type, ldb, stride_b, beta, c, c_type, ldc, stride_c,&
            batch_count, compute_type, algo, solution_index, flags) &
            result(res) &
            bind(c, name = 'hipblasGemmStridedBatchedExFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_OP_N)), value :: transB
        integer(c_int), value :: m
        integer(c_int), value :: n
        integer(c_int), value :: k
        type(c_ptr), value :: alpha
        type(c_ptr), value :: a
        integer(kind(HIPBLAS_R_16F)), value :: a_type
        integer(c_int), value :: lda
        integer(c_int64_t), value :: stride_a
        type(c_ptr), value :: b
        integer(kind(HIPBLAS_R_16F)), value :: b_type
        integer(c_int), value :: ldb
        integer(c_int64_t), value :: stride_b
        type(c_ptr), value :: beta
        type(c_ptr), value :: c
        integer(kind(HIPBLAS_R_16F)), value :: c_type
        integer(c_int), value :: ldc
        integer(c_int64_t), value :: stride_c
        integer(c_int), value :: batch_count
        integer(kind(HIPBLAS_R_16F)), value :: compute_type
        integer(kind(HIPBLAS_GEMM_DEFAULT)), value :: algo
        integer(c_int32_t), value :: solution_index
        ! No unsigned types in fortran. If larger values are needed
        ! we will need a workaround.
        integer(c_int32_t), value :: flags
        integer(c_int) :: res
        res = hipblasGemmStridedBatchedEx(handle, transA, transB, m, n, k, alpha,&
              a, a_type, lda, stride_a, b, b_type, ldb, stride_b, beta, c, c_type, ldc, stride_c,&
              batch_count, compute_type, algo, solution_index, flags)
    end function hipblasGemmStridedBatchedExFortran

    ! trsmEx
    function hipblasTrsmExFortran(handle, side, uplo, transA, diag, m, n, alpha, A, lda, &
            B, ldb, invA, invA_size, compute_type) &
            result(res) &
            bind(c, name = 'hipblasTrsmExFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
        integer(kind(HIPBLAS_FILL_MODE_UPPER)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_DIAG_UNIT)), value :: diag
        integer(c_int), value :: m
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: B
        integer(c_int), value :: ldb
        type(c_ptr), value :: invA
        integer(c_int), value :: invA_size
        integer(kind(HIPBLAS_R_16F)), value :: compute_type
        integer(c_int) :: res
        res = hipblasTrsmEx(handle, side, uplo, transA, diag, m, n, alpha,&
              A, lda, B, ldb, invA, invA_size, compute_type)
    end function hipblasTrsmExFortran

    function hipblasTrsmBatchedExFortran(handle, side, uplo, transA, diag, m, n, alpha, A, lda, &
            B, ldb, batch_count, invA, invA_size, compute_type) &
            result(res) &
            bind(c, name = 'hipblasTrsmBatchedExFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
        integer(kind(HIPBLAS_FILL_MODE_UPPER)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_DIAG_UNIT)), value :: diag
        integer(c_int), value :: m
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: B
        integer(c_int), value :: ldb
        integer(c_int), value :: batch_count
        type(c_ptr), value :: invA
        integer(c_int), value :: invA_size
        integer(kind(HIPBLAS_R_16F)), value :: compute_type
        integer(c_int) :: res
        res = hipblasTrsmBatchedEx(handle, side, uplo, transA, diag, m, n, alpha,&
              A, lda, B, ldb, batch_count, invA, invA_size, compute_type)
    end function hipblasTrsmBatchedExFortran

    function hipblasTrsmStridedBatchedExFortran(handle, side, uplo, transA, diag, m, n, alpha, A, lda, stride_A, &
            B, ldb, stride_B, batch_count, invA, invA_size, stride_invA, compute_type) &
            result(res) &
            bind(c, name = 'hipblasTrsmStridedBatchedExFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
        integer(kind(HIPBLAS_FILL_MODE_UPPER)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(kind(HIPBLAS_DIAG_UNIT)), value :: diag
        integer(c_int), value :: m
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int64_t), value :: stride_A
        type(c_ptr), value :: B
        integer(c_int), value :: ldb
        integer(c_int64_t), value :: stride_B
        integer(c_int), value :: batch_count
        type(c_ptr), value :: invA
        integer(c_int), value :: invA_size
        integer(c_int64_t), value :: stride_invA
        integer(kind(HIPBLAS_R_16F)), value :: compute_type
        integer(c_int) :: res
        res = hipblasTrsmStridedBatchedEx(handle, side, uplo, transA, diag, m, n, alpha,&
              A, lda, stride_A, B, ldb, stride_B, batch_count, invA, invA_size, stride_invA, compute_type)
    end function hipblasTrsmStridedBatchedExFortran

end module hipblas_interface
