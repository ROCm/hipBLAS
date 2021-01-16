! ************************************************************************
!  Copyright 2016-2020 Advanced Micro Devices, Inc.
! ************************************************************************

module hipblas_enums
    use iso_c_binding

    !!!!!!!!!!!!!!!!!!!!!!!
    !    hipBLAS types    !
    !!!!!!!!!!!!!!!!!!!!!!!
    enum, bind(c)
        enumerator :: HIPBLAS_OP_N = 111
        enumerator :: HIPBLAS_OP_T = 112
        enumerator :: HIPBLAS_OP_C = 113
    end enum

    enum, bind(c)
        enumerator :: HIPBLAS_FILL_MODE_UPPER = 121
        enumerator :: HIPBLAS_FILL_MODE_LOWER = 122
        enumerator :: HIPBLAS_FILL_MODE_FULL  = 123
    end enum

    enum, bind(c)
        enumerator :: HIPBLAS_DIAG_NON_UNIT = 131
        enumerator :: HIPBLAS_DIAG_UNIT     = 132
    end enum

    enum, bind(c)
        enumerator :: HIPBLAS_SIDE_LEFT  = 141
        enumerator :: HIPBLAS_SIDE_RIGHT = 142
        enumerator :: HIPBLAS_SIDE_BOTH  = 143
    end enum

    enum, bind(c)
        enumerator :: HIPBLAS_STATUS_SUCCESS           = 0
        enumerator :: HIPBLAS_STATUS_NOT_INITIALIZED   = 1
        enumerator :: HIPBLAS_STATUS_ALLOC_FAILED      = 2
        enumerator :: HIPBLAS_STATUS_INVALID_VALUE     = 3
        enumerator :: HIPBLAS_STATUS_MAPPING_ERROR     = 4
        enumerator :: HIPBLAS_STATUS_EXECUTION_FAILED  = 5
        enumerator :: HIPBLAS_STATUS_INTERNAL_ERROR    = 6
        enumerator :: HIPBLAS_STATUS_NOT_SUPPORTED     = 7
        enumerator :: HIPBLAS_STATUS_ARCH_MISMATCH     = 8
        enumerator :: HIPBLAS_STATUS_HANDLE_IS_NULLPTR = 9
    end enum

    enum, bind(c)
        enumerator :: HIPBLAS_R_16F = 150
        enumerator :: HIPBLAS_R_32F = 151
        enumerator :: HIPBLAS_R_64F = 152
        enumerator :: HIPBLAS_C_16F = 153
        enumerator :: HIPBLAS_C_32F = 154
        enumerator :: HIPBLAS_C_64F = 155
        enumerator :: HIPBLAS_R_8I  = 160
        enumerator :: HIPBLAS_R_8U  = 161
        enumerator :: HIPBLAS_R_32I = 162
        enumerator :: HIPBLAS_R_32U = 163
        enumerator :: HIPBLAS_C_8I  = 164
        enumerator :: HIPBLAS_C_8U  = 165
        enumerator :: HIPBLAS_C_32I = 166
        enumerator :: HIPBLAS_C_32U = 167
        enumerator :: HIPBLAS_R_16B = 168
        enumerator :: HIPBLAS_C_16B = 169
    end enum

    enum, bind(c)
        enumerator :: HIPBLAS_POINTER_MODE_HOST = 0
        enumerator :: HIPBLAS_POINTER_MODE_DEVICE = 1
    end enum

    enum, bind(c)
        enumerator :: HIPBLAS_GEMM_DEFAULT = 100
    end enum

    enum, bind(c)
        enumerator :: HIPBLAS_ATOMICS_NOT_ALLOWED = 0
        enumerator :: HIPBLAS_ATOMICS_ALLOWED = 1
    end enum

end module hipblas_enums

module hipblas
    use iso_c_binding

    !--------!
    !   Aux  !
    !--------!
    interface
        function hipblasCreate(handle) &
                result(c_int) &
                bind(c, name = 'hipblasCreate')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
        end function hipblasCreate
    end interface

    interface
        function hipblasDestroy(handle) &
                result(c_int) &
                bind(c, name = 'hipblasDestroy')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
        end function hipblasDestroy
    end interface

    interface
        function hipblasSetStream(handle, streamId) &
                result(c_int) &
                bind(c, name = 'hipblasSetStream')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            type(c_ptr), value :: streamId
        end function hipblasSetStream
    end interface

    interface
        function hipblasGetStream(handle, streamId) &
                result(c_int) &
                bind(c, name = 'hipblasGetStream')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            type(c_ptr), value :: streamId
        end function hipblasGetStream
    end interface

    interface
        function hipblasSetPointerMode(handle, mode) &
                result(c_int) &
                bind(c, name = 'hipblasSetPointerMode')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            type(c_ptr), value :: mode
        end function hipblasSetPointerMode
    end interface

    interface
        function hipblasGetPointerMode(handle, mode) &
                result(c_int) &
                bind(c, name = 'hipblasGetPointerMode')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            type(c_ptr), value :: mode
        end function hipblasGetPointerMode
    end interface

    interface
        function hipblasSetVector(n, elemSize, x, incx, y, incy) &
                result(c_int) &
                bind(c, name = 'hipblasSetVector')
            use iso_c_binding
            implicit none
            integer(c_int), value :: n
            integer(c_int), value :: elemSize
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
        end function hipblasSetVector
    end interface

    interface
        function hipblasGetVector(n, elemSize, x, incx, y, incy) &
                result(c_int) &
                bind(c, name = 'hipblasGetVector')
            use iso_c_binding
            implicit none
            integer(c_int), value :: n
            integer(c_int), value :: elemSize
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
        end function hipblasGetVector
    end interface

    interface
        function hipblasSetMatrix(rows, cols, elemSize, A, lda, B, ldb) &
                result(c_int) &
                bind(c, name = 'hipblasSetMatrix')
            use iso_c_binding
            implicit none
            integer(c_int), value :: rows
            integer(c_int), value :: cols
            integer(c_int), value :: elemSize
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
        end function hipblasSetMatrix
    end interface

    interface
        function hipblasGetMatrix(rows, cols, elemSize, A, lda, B, ldb) &
                result(c_int) &
                bind(c, name = 'hipblasGetMatrix')
            use iso_c_binding
            implicit none
            integer(c_int), value :: rows
            integer(c_int), value :: cols
            integer(c_int), value :: elemSize
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
        end function hipblasGetMatrix
    end interface

    interface
        function hipblasSetVectorAsync(n, elemSize, x, incx, y, incy, stream) &
                result(c_int) &
                bind(c, name = 'hipblasSetVectorAsync')
            use iso_c_binding
            implicit none
            integer(c_int), value :: n
            integer(c_int), value :: elemSize
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            type(c_ptr), value :: stream
        end function hipblasSetVectorAsync
    end interface

    interface
        function hipblasGetVectorAsync(n, elemSize, x, incx, y, incy, stream) &
                result(c_int) &
                bind(c, name = 'hipblasGetVectorAsync')
            use iso_c_binding
            implicit none
            integer(c_int), value :: n
            integer(c_int), value :: elemSize
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            type(c_ptr), value :: stream
        end function hipblasGetVectorAsync
    end interface

    interface
        function hipblasSetMatrixAsync(rows, cols, elemSize, A, lda, B, ldb, stream) &
                result(c_int) &
                bind(c, name = 'hipblasSetMatrixAsync')
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
        end function hipblasSetMatrixAsync
    end interface

    interface
        function hipblasGetMatrixAsync(rows, cols, elemSize, A, lda, B, ldb, stream) &
                result(c_int) &
                bind(c, name = 'hipblasGetMatrixAsync')
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
        end function hipblasGetMatrixAsync
    end interface

    ! atomics mode
    interface
        function hipblasSetAtomicsMode(handle, atomics_mode) &
                result(c_int) &
                bind(c, name = 'hipblasSetAtomicsMode')
            use iso_c_binding
            use hipblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(HIPBLAS_ATOMICS_ALLOWED)), value :: atomics_mode
        end function hipblasSetAtomicsMode
    end interface

    interface
        function hipblasGetAtomicsMode(handle, atomics_mode) &
                result(c_int) &
                bind(c, name = 'hipblasGetAtomicsMode')
            use iso_c_binding
            use hipblas_enums
            implicit none
            type(c_ptr), value :: handle
            type(c_ptr), value :: atomics_mode
        end function hipblasGetAtomicsMode
    end interface

    !--------!
    ! blas 1 !
    !--------!

    ! scal
    interface
        function hipblasSscal(handle, n, alpha, x, incx) &
                result(c_int) &
                bind(c, name = 'hipblasSscal')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
        end function hipblasSscal
    end interface

    interface
        function hipblasDscal(handle, n, alpha, x, incx) &
                result(c_int) &
                bind(c, name = 'hipblasDscal')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
        end function hipblasDscal
    end interface

    interface
        function hipblasCscal(handle, n, alpha, x, incx) &
                result(c_int) &
                bind(c, name = 'hipblasCscal')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
        end function hipblasCscal
    end interface

    interface
        function hipblasZscal(handle, n, alpha, x, incx) &
                result(c_int) &
                bind(c, name = 'hipblasZscal')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
        end function hipblasZscal
    end interface

    interface
        function hipblasCsscal(handle, n, alpha, x, incx) &
                result(c_int) &
                bind(c, name = 'hipblasCsscal')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
        end function hipblasCsscal
    end interface

    interface
        function hipblasZdscal(handle, n, alpha, x, incx) &
                result(c_int) &
                bind(c, name = 'hipblasZdscal')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
        end function hipblasZdscal
    end interface

    ! scalBatched
    interface
        function hipblasSscalBatched(handle, n, alpha, x, incx, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasSscalBatched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: batch_count
        end function hipblasSscalBatched
    end interface

    interface
        function hipblasDscalBatched(handle, n, alpha, x, incx, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasDscalBatched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: batch_count
        end function hipblasDscalBatched
    end interface

    interface
        function hipblasCscalBatched(handle, n, alpha, x, incx, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasCscalBatched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: batch_count
        end function hipblasCscalBatched
    end interface

    interface
        function hipblasZscalBatched(handle, n, alpha, x, incx, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasZscalBatched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: batch_count
        end function hipblasZscalBatched
    end interface

    interface
        function hipblasCsscalBatched(handle, n, alpha, x, incx, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasCsscalBatched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: batch_count
        end function hipblasCsscalBatched
    end interface

    interface
        function hipblasZdscalBatched(handle, n, alpha, x, incx, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasZdscalBatched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: batch_count
        end function hipblasZdscalBatched
    end interface

    ! scalStridedBatched
    interface
        function hipblasSscalStridedBatched(handle, n, alpha, x, incx, stride_x, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasSscalStridedBatched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            integer(c_int), value :: batch_count
        end function hipblasSscalStridedBatched
    end interface

    interface
        function hipblasDscalStridedBatched(handle, n, alpha, x, incx, stride_x, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasDscalStridedBatched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            integer(c_int), value :: batch_count
        end function hipblasDscalStridedBatched
    end interface

    interface
        function hipblasCscalStridedBatched(handle, n, alpha, x, incx, stride_x, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasCscalStridedBatched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            integer(c_int), value :: batch_count
        end function hipblasCscalStridedBatched
    end interface

    interface
        function hipblasZscalStridedBatched(handle, n, alpha, x, incx, stride_x, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasZscalStridedBatched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            integer(c_int), value :: batch_count
        end function hipblasZscalStridedBatched
    end interface

    interface
        function hipblasCsscalStridedBatched(handle, n, alpha, x, incx, stride_x, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasCsscalStridedBatched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            integer(c_int), value :: batch_count
        end function hipblasCsscalStridedBatched
    end interface

    interface
        function hipblasZdscalStridedBatched(handle, n, alpha, x, incx, stride_x, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasZdscalStridedBatched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            integer(c_int), value :: batch_count
        end function hipblasZdscalStridedBatched
    end interface

    ! copy
    interface
        function hipblasScopy(handle, n, x, incx, y, incy) &
                result(c_int) &
                bind(c, name = 'hipblasScopy')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
        end function hipblasScopy
    end interface

    interface
        function hipblasDcopy(handle, n, x, incx, y, incy) &
                result(c_int) &
                bind(c, name = 'hipblasDcopy')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
        end function hipblasDcopy
    end interface

    interface
        function hipblasCcopy(handle, n, x, incx, y, incy) &
                result(c_int) &
                bind(c, name = 'hipblasCcopy')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
        end function hipblasCcopy
    end interface

    interface
        function hipblasZcopy(handle, n, x, incx, y, incy) &
                result(c_int) &
                bind(c, name = 'hipblasZcopy')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
        end function hipblasZcopy
    end interface

    ! copyBatched
    interface
        function hipblasScopyBatched(handle, n, x, incx, y, incy, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasScopyBatched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int), value :: batch_count
        end function hipblasScopyBatched
    end interface

    interface
        function hipblasDcopyBatched(handle, n, x, incx, y, incy, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasDcopyBatched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int), value :: batch_count
        end function hipblasDcopyBatched
    end interface

    interface
        function hipblasCcopyBatched(handle, n, x, incx, y, incy, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasCcopyBatched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int), value :: batch_count
        end function hipblasCcopyBatched
    end interface

    interface
        function hipblasZcopyBatched(handle, n, x, incx, y, incy, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasZcopyBatched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int), value :: batch_count
        end function hipblasZcopyBatched
    end interface

    ! copyStridedBatched
    interface
        function hipblasScopyStridedBatched(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasScopyStridedBatched')
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
        end function hipblasScopyStridedBatched
    end interface

    interface
        function hipblasDcopyStridedBatched(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasDcopyStridedBatched')
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
        end function hipblasDcopyStridedBatched
    end interface

    interface
        function hipblasCcopyStridedBatched(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasCcopyStridedBatched')
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
        end function hipblasCcopyStridedBatched
    end interface

    interface
        function hipblasZcopyStridedBatched(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasZcopyStridedBatched')
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
        end function hipblasZcopyStridedBatched
    end interface

    ! dot
    interface
        function hipblasSdot(handle, n, x, incx, y, incy, result) &
                result(c_int) &
                bind(c, name = 'hipblasSdot')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            type(c_ptr), value :: result
        end function hipblasSdot
    end interface

    interface
        function hipblasDdot(handle, n, x, incx, y, incy, result) &
                result(c_int) &
                bind(c, name = 'hipblasDdot')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            type(c_ptr), value :: result
        end function hipblasDdot
    end interface

    interface
        function hipblasHdot(handle, n, x, incx, y, incy, result) &
                result(c_int) &
                bind(c, name = 'hipblasHdot')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            type(c_ptr), value :: result
        end function hipblasHdot
    end interface

    interface
        function hipblasBfdot(handle, n, x, incx, y, incy, result) &
                result(c_int) &
                bind(c, name = 'hipblasBfdot')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            type(c_ptr), value :: result
        end function hipblasBfdot
    end interface

    interface
        function hipblasCdotu(handle, n, x, incx, y, incy, result) &
                result(c_int) &
                bind(c, name = 'hipblasCdotu')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            type(c_ptr), value :: result
        end function hipblasCdotu
    end interface

    interface
        function hipblasCdotc(handle, n, x, incx, y, incy, result) &
                result(c_int) &
                bind(c, name = 'hipblasCdotc')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            type(c_ptr), value :: result
        end function hipblasCdotc
    end interface

    interface
        function hipblasZdotu(handle, n, x, incx, y, incy, result) &
                result(c_int) &
                bind(c, name = 'hipblasZdotu')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            type(c_ptr), value :: result
        end function hipblasZdotu
    end interface

    interface
        function hipblasZdotc(handle, n, x, incx, y, incy, result) &
                result(c_int) &
                bind(c, name = 'hipblasZdotc')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            type(c_ptr), value :: result
        end function hipblasZdotc
    end interface

    ! dotBatched
    interface
        function hipblasSdotBatched(handle, n, x, incx, y, incy, batch_count, result) &
                result(c_int) &
                bind(c, name = 'hipblasSdotBatched')
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
        end function hipblasSdotBatched
    end interface

    interface
        function hipblasDdotBatched(handle, n, x, incx, y, incy, batch_count, result) &
                result(c_int) &
                bind(c, name = 'hipblasDdotBatched')
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
        end function hipblasDdotBatched
    end interface

    interface
        function hipblasHdotBatched(handle, n, x, incx, y, incy, batch_count, result) &
                result(c_int) &
                bind(c, name = 'hipblasHdotBatched')
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
        end function hipblasHdotBatched
    end interface

    interface
        function hipblasBfdotBatched(handle, n, x, incx, y, incy, batch_count, result) &
                result(c_int) &
                bind(c, name = 'hipblasBfdotBatched')
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
        end function hipblasBfdotBatched
    end interface

    interface
        function hipblasCdotuBatched(handle, n, x, incx, y, incy, batch_count, result) &
                result(c_int) &
                bind(c, name = 'hipblasCdotuBatched')
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
        end function hipblasCdotuBatched
    end interface

    interface
        function hipblasCdotcBatched(handle, n, x, incx, y, incy, batch_count, result) &
                result(c_int) &
                bind(c, name = 'hipblasCdotcBatched')
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
        end function hipblasCdotcBatched
    end interface

    interface
        function hipblasZdotuBatched(handle, n, x, incx, y, incy, batch_count, result) &
                result(c_int) &
                bind(c, name = 'hipblasZdotuBatched')
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
        end function hipblasZdotuBatched
    end interface

    interface
        function hipblasZdotcBatched(handle, n, x, incx, y, incy, batch_count, result) &
                result(c_int) &
                bind(c, name = 'hipblasZdotcBatched')
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
        end function hipblasZdotcBatched
    end interface

    ! dotStridedBatched
    interface
        function hipblasSdotStridedBatched(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count, result) &
                result(c_int) &
                bind(c, name = 'hipblasSdotStridedBatched')
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
        end function hipblasSdotStridedBatched
    end interface

    interface
        function hipblasDdotStridedBatched(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count, result) &
                result(c_int) &
                bind(c, name = 'hipblasDdotStridedBatched')
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
        end function hipblasDdotStridedBatched
    end interface

    interface
        function hipblasHdotStridedBatched(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count, result) &
                result(c_int) &
                bind(c, name = 'hipblasHdotStridedBatched')
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
        end function hipblasHdotStridedBatched
    end interface

    interface
        function hipblasBfdotStridedBatched(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count, result) &
                result(c_int) &
                bind(c, name = 'hipblasBfdotStridedBatched')
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
        end function hipblasBfdotStridedBatched
    end interface

    interface
        function hipblasCdotuStridedBatched(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count, result) &
                result(c_int) &
                bind(c, name = 'hipblasCdotuStridedBatched')
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
        end function hipblasCdotuStridedBatched
    end interface

    interface
        function hipblasCdotcStridedBatched(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count, result) &
                result(c_int) &
                bind(c, name = 'hipblasCdotcStridedBatched')
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
        end function hipblasCdotcStridedBatched
    end interface

    interface
        function hipblasZdotuStridedBatched(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count, result) &
                result(c_int) &
                bind(c, name = 'hipblasZdotuStridedBatched')
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
        end function hipblasZdotuStridedBatched
    end interface

    interface
        function hipblasZdotcStridedBatched(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count, result) &
                result(c_int) &
                bind(c, name = 'hipblasZdotcStridedBatched')
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
        end function hipblasZdotcStridedBatched
    end interface

    ! swap
    interface
        function hipblasSswap(handle, n, x, incx, y, incy) &
                result(c_int) &
                bind(c, name = 'hipblasSswap')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
        end function hipblasSswap
    end interface

    interface
        function hipblasDswap(handle, n, x, incx, y, incy) &
                result(c_int) &
                bind(c, name = 'hipblasDswap')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
        end function hipblasDswap
    end interface

    interface
        function hipblasCswap(handle, n, x, incx, y, incy) &
                result(c_int) &
                bind(c, name = 'hipblasCswap')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
        end function hipblasCswap
    end interface

    interface
        function hipblasZswap(handle, n, x, incx, y, incy) &
                result(c_int) &
                bind(c, name = 'hipblasZswap')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
        end function hipblasZswap
    end interface

    ! swapBatched
    interface
        function hipblasSswapBatched(handle, n, x, incx, y, incy, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasSswapBatched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int), value :: batch_count
        end function hipblasSswapBatched
    end interface

    interface
        function hipblasDswapBatched(handle, n, x, incx, y, incy, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasDswapBatched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int), value :: batch_count
        end function hipblasDswapBatched
    end interface

    interface
        function hipblasCswapBatched(handle, n, x, incx, y, incy, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasCswapBatched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int), value :: batch_count
        end function hipblasCswapBatched
    end interface

    interface
        function hipblasZswapBatched(handle, n, x, incx, y, incy, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasZswapBatched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int), value :: batch_count
        end function hipblasZswapBatched
    end interface

    ! swapStridedBatched
    interface
        function hipblasSswapStridedBatched(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasSswapStridedBatched')
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
        end function hipblasSswapStridedBatched
    end interface

    interface
        function hipblasDswapStridedBatched(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasDswapStridedBatched')
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
        end function hipblasDswapStridedBatched
    end interface

    interface
        function hipblasCswapStridedBatched(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasCswapStridedBatched')
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
        end function hipblasCswapStridedBatched
    end interface

    interface
        function hipblasZswapStridedBatched(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasZswapStridedBatched')
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
        end function hipblasZswapStridedBatched
    end interface

    ! axpy
    interface
        function hipblasHaxpy(handle, n, alpha, x, incx, y, incy) &
                result(c_int) &
                bind(c, name = 'hipblasHaxpy')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
        end function hipblasHaxpy
    end interface

    interface
        function hipblasSaxpy(handle, n, alpha, x, incx, y, incy) &
                result(c_int) &
                bind(c, name = 'hipblasSaxpy')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
        end function hipblasSaxpy
    end interface

    interface
        function hipblasDaxpy(handle, n, alpha, x, incx, y, incy) &
                result(c_int) &
                bind(c, name = 'hipblasDaxpy')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
        end function hipblasDaxpy
    end interface

    interface
        function hipblasCaxpy(handle, n, alpha, x, incx, y, incy) &
                result(c_int) &
                bind(c, name = 'hipblasCaxpy')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
        end function hipblasCaxpy
    end interface

    interface
        function hipblasZaxpy(handle, n, alpha, x, incx, y, incy) &
                result(c_int) &
                bind(c, name = 'hipblasZaxpy')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
        end function hipblasZaxpy
    end interface

    ! axpyBatched
    interface
        function hipblasHaxpyBatched(handle, n, alpha, x, incx, y, incy, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasHaxpyBatched')
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
        end function hipblasHaxpyBatched
    end interface

    interface
        function hipblasSaxpyBatched(handle, n, alpha, x, incx, y, incy, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasSaxpyBatched')
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
        end function hipblasSaxpyBatched
    end interface

    interface
        function hipblasDaxpyBatched(handle, n, alpha, x, incx, y, incy, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasDaxpyBatched')
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
        end function hipblasDaxpyBatched
    end interface

    interface
        function hipblasCaxpyBatched(handle, n, alpha, x, incx, y, incy, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasCaxpyBatched')
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
        end function hipblasCaxpyBatched
    end interface

    interface
        function hipblasZaxpyBatched(handle, n, alpha, x, incx, y, incy, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasZaxpyBatched')
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
        end function hipblasZaxpyBatched
    end interface

    ! axpyStridedBatched
    interface
        function hipblasHaxpyStridedBatched(handle, n, alpha, x, incx, stride_x, y, incy, stride_y, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasHaxpyStridedBatched')
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
        end function hipblasHaxpyStridedBatched
    end interface

    interface
        function hipblasSaxpyStridedBatched(handle, n, alpha, x, incx, stride_x, y, incy, stride_y, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasSaxpyStridedBatched')
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
        end function hipblasSaxpyStridedBatched
    end interface

    interface
        function hipblasDaxpyStridedBatched(handle, n, alpha, x, incx, stride_x, y, incy, stride_y, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasDaxpyStridedBatched')
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
        end function hipblasDaxpyStridedBatched
    end interface

    interface
        function hipblasCaxpyStridedBatched(handle, n, alpha, x, incx, stride_x, y, incy, stride_y, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasCaxpyStridedBatched')
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
        end function hipblasCaxpyStridedBatched
    end interface

    interface
        function hipblasZaxpyStridedBatched(handle, n, alpha, x, incx, stride_x, y, incy, stride_y, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasZaxpyStridedBatched')
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
        end function hipblasZaxpyStridedBatched
    end interface

    ! asum
    interface
        function hipblasSasum(handle, n, x, incx, result) &
                result(c_int) &
                bind(c, name = 'hipblasSasum')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: result
        end function hipblasSasum
    end interface

    interface
        function hipblasDasum(handle, n, x, incx, result) &
                result(c_int) &
                bind(c, name = 'hipblasDasum')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: result
        end function hipblasDasum
    end interface

    interface
        function hipblasScasum(handle, n, x, incx, result) &
                result(c_int) &
                bind(c, name = 'hipblasScasum')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: result
        end function hipblasScasum
    end interface

    interface
        function hipblasDzasum(handle, n, x, incx, result) &
                result(c_int) &
                bind(c, name = 'hipblasDzasum')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: result
        end function hipblasDzasum
    end interface

    ! asumBatched
    interface
        function hipblasSasumBatched(handle, n, x, incx, batch_count, result) &
                result(c_int) &
                bind(c, name = 'hipblasSasumBatched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function hipblasSasumBatched
    end interface

    interface
        function hipblasDasumBatched(handle, n, x, incx, batch_count, result) &
                result(c_int) &
                bind(c, name = 'hipblasDasumBatched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function hipblasDasumBatched
    end interface

    interface
        function hipblasScasumBatched(handle, n, x, incx, batch_count, result) &
                result(c_int) &
                bind(c, name = 'hipblasScasumBatched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function hipblasScasumBatched
    end interface

    interface
        function hipblasDzasumBatched(handle, n, x, incx, batch_count, result) &
                result(c_int) &
                bind(c, name = 'hipblasDzasumBatched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function hipblasDzasumBatched
    end interface

    ! asumStridedBatched
    interface
        function hipblasSasumStridedBatched(handle, n, x, incx, stride_x, batch_count, result) &
                result(c_int) &
                bind(c, name = 'hipblasSasumStridedBatched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function hipblasSasumStridedBatched
    end interface

    interface
        function hipblasDasumStridedBatched(handle, n, x, incx, stride_x, batch_count, result) &
                result(c_int) &
                bind(c, name = 'hipblasDasumStridedBatched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function hipblasDasumStridedBatched
    end interface

    interface
        function hipblasScasumStridedBatched(handle, n, x, incx, stride_x, batch_count, result) &
                result(c_int) &
                bind(c, name = 'hipblasScasumStridedBatched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function hipblasScasumStridedBatched
    end interface

    interface
        function hipblasDzasumStridedBatched(handle, n, x, incx, stride_x, batch_count, result) &
                result(c_int) &
                bind(c, name = 'hipblasDzasumStridedBatched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function hipblasDzasumStridedBatched
    end interface

    ! nrm2
    interface
        function hipblasSnrm2(handle, n, x, incx, result) &
                result(c_int) &
                bind(c, name = 'hipblasSnrm2')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: result
        end function hipblasSnrm2
    end interface

    interface
        function hipblasDnrm2(handle, n, x, incx, result) &
                result(c_int) &
                bind(c, name = 'hipblasDnrm2')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: result
        end function hipblasDnrm2
    end interface

    interface
        function hipblasScnrm2(handle, n, x, incx, result) &
                result(c_int) &
                bind(c, name = 'hipblasScnrm2')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: result
        end function hipblasScnrm2
    end interface

    interface
        function hipblasDznrm2(handle, n, x, incx, result) &
                result(c_int) &
                bind(c, name = 'hipblasDznrm2')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: result
        end function hipblasDznrm2
    end interface

    ! nrm2Batched
    interface
        function hipblasSnrm2Batched(handle, n, x, incx, batch_count, result) &
                result(c_int) &
                bind(c, name = 'hipblasSnrm2Batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function hipblasSnrm2Batched
    end interface

    interface
        function hipblasDnrm2Batched(handle, n, x, incx, batch_count, result) &
                result(c_int) &
                bind(c, name = 'hipblasDnrm2Batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function hipblasDnrm2Batched
    end interface

    interface
        function hipblasScnrm2Batched(handle, n, x, incx, batch_count, result) &
                result(c_int) &
                bind(c, name = 'hipblasScnrm2Batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function hipblasScnrm2Batched
    end interface

    interface
        function hipblasDznrm2Batched(handle, n, x, incx, batch_count, result) &
                result(c_int) &
                bind(c, name = 'hipblasDznrm2Batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function hipblasDznrm2Batched
    end interface

    ! nrm2StridedBatched
    interface
        function hipblasSnrm2StridedBatched(handle, n, x, incx, stride_x, batch_count, result) &
                result(c_int) &
                bind(c, name = 'hipblasSnrm2StridedBatched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function hipblasSnrm2StridedBatched
    end interface

    interface
        function hipblasDnrm2StridedBatched(handle, n, x, incx, stride_x, batch_count, result) &
                result(c_int) &
                bind(c, name = 'hipblasDnrm2StridedBatched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function hipblasDnrm2StridedBatched
    end interface

    interface
        function hipblasScnrm2StridedBatched(handle, n, x, incx, stride_x, batch_count, result) &
                result(c_int) &
                bind(c, name = 'hipblasScnrm2StridedBatched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function hipblasScnrm2StridedBatched
    end interface

    interface
        function hipblasDznrm2StridedBatched(handle, n, x, incx, stride_x, batch_count, result) &
                result(c_int) &
                bind(c, name = 'hipblasDznrm2StridedBatched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function hipblasDznrm2StridedBatched
    end interface

    ! amax
    interface
        function hipblasIsamax(handle, n, x, incx, result) &
                result(c_int) &
                bind(c, name = 'hipblasIsamax')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: result
        end function hipblasIsamax
    end interface

    interface
        function hipblasIdamax(handle, n, x, incx, result) &
                result(c_int) &
                bind(c, name = 'hipblasIdamax')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: result
        end function hipblasIdamax
    end interface

    interface
        function hipblasIcamax(handle, n, x, incx, result) &
                result(c_int) &
                bind(c, name = 'hipblasIcamax')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: result
        end function hipblasIcamax
    end interface

    interface
        function hipblasIzamax(handle, n, x, incx, result) &
                result(c_int) &
                bind(c, name = 'hipblasIzamax')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: result
        end function hipblasIzamax
    end interface

    ! amaxBatched
    interface
        function hipblasIsamaxBatched(handle, n, x, incx, batch_count, result) &
                result(c_int) &
                bind(c, name = 'hipblasIsamaxBatched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function hipblasIsamaxBatched
    end interface

    interface
        function hipblasIdamaxBatched(handle, n, x, incx, batch_count, result) &
                result(c_int) &
                bind(c, name = 'hipblasIdamaxBatched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function hipblasIdamaxBatched
    end interface

    interface
        function hipblasIcamaxBatched(handle, n, x, incx, batch_count, result) &
                result(c_int) &
                bind(c, name = 'hipblasIcamaxBatched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function hipblasIcamaxBatched
    end interface

    interface
        function hipblasIzamaxBatched(handle, n, x, incx, batch_count, result) &
                result(c_int) &
                bind(c, name = 'hipblasIzamaxBatched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function hipblasIzamaxBatched
    end interface

    ! amaxStridedBatched
    interface
        function hipblasIsamaxStridedBatched(handle, n, x, incx, stride_x, batch_count, result) &
                result(c_int) &
                bind(c, name = 'hipblasIsamaxStridedBatched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function hipblasIsamaxStridedBatched
    end interface

    interface
        function hipblasIdamaxStridedBatched(handle, n, x, incx, stride_x, batch_count, result) &
                result(c_int) &
                bind(c, name = 'hipblasIdamaxStridedBatched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function hipblasIdamaxStridedBatched
    end interface

    interface
        function hipblasIcamaxStridedBatched(handle, n, x, incx, stride_x, batch_count, result) &
                result(c_int) &
                bind(c, name = 'hipblasIcamaxStridedBatched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function hipblasIcamaxStridedBatched
    end interface

    interface
        function hipblasIzamaxStridedBatched(handle, n, x, incx, stride_x, batch_count, result) &
                result(c_int) &
                bind(c, name = 'hipblasIzamaxStridedBatched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function hipblasIzamaxStridedBatched
    end interface

    ! amin
    interface
        function hipblasIsamin(handle, n, x, incx, result) &
                result(c_int) &
                bind(c, name = 'hipblasIsamin')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: result
        end function hipblasIsamin
    end interface

    interface
        function hipblasIdamin(handle, n, x, incx, result) &
                result(c_int) &
                bind(c, name = 'hipblasIdamin')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: result
        end function hipblasIdamin
    end interface

    interface
        function hipblasIcamin(handle, n, x, incx, result) &
                result(c_int) &
                bind(c, name = 'hipblasIcamin')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: result
        end function hipblasIcamin
    end interface

    interface
        function hipblasIzamin(handle, n, x, incx, result) &
                result(c_int) &
                bind(c, name = 'hipblasIzamin')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: result
        end function hipblasIzamin
    end interface

    ! aminBatched
    interface
        function hipblasIsaminBatched(handle, n, x, incx, batch_count, result) &
                result(c_int) &
                bind(c, name = 'hipblasIsaminBatched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function hipblasIsaminBatched
    end interface

    interface
        function hipblasIdaminBatched(handle, n, x, incx, batch_count, result) &
                result(c_int) &
                bind(c, name = 'hipblasIdaminBatched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function hipblasIdaminBatched
    end interface

    interface
        function hipblasIcaminBatched(handle, n, x, incx, batch_count, result) &
                result(c_int) &
                bind(c, name = 'hipblasIcaminBatched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function hipblasIcaminBatched
    end interface

    interface
        function hipblasIzaminBatched(handle, n, x, incx, batch_count, result) &
                result(c_int) &
                bind(c, name = 'hipblasIzaminBatched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function hipblasIzaminBatched
    end interface

    ! aminStridedBatched
    interface
        function hipblasIsaminStridedBatched(handle, n, x, incx, stride_x, batch_count, result) &
                result(c_int) &
                bind(c, name = 'hipblasIsaminStridedBatched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function hipblasIsaminStridedBatched
    end interface

    interface
        function hipblasIdaminStridedBatched(handle, n, x, incx, stride_x, batch_count, result) &
                result(c_int) &
                bind(c, name = 'hipblasIdaminStridedBatched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function hipblasIdaminStridedBatched
    end interface

    interface
        function hipblasIcaminStridedBatched(handle, n, x, incx, stride_x, batch_count, result) &
                result(c_int) &
                bind(c, name = 'hipblasIcaminStridedBatched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function hipblasIcaminStridedBatched
    end interface

    interface
        function hipblasIzaminStridedBatched(handle, n, x, incx, stride_x, batch_count, result) &
                result(c_int) &
                bind(c, name = 'hipblasIzaminStridedBatched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function hipblasIzaminStridedBatched
    end interface

    ! rot
    interface
        function hipblasSrot(handle, n, x, incx, y, incy, c, s) &
                result(c_int) &
                bind(c, name = 'hipblasSrot')
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
        end function hipblasSrot
    end interface

    interface
        function hipblasDrot(handle, n, x, incx, y, incy, c, s) &
                result(c_int) &
                bind(c, name = 'hipblasDrot')
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
        end function hipblasDrot
    end interface

    interface
        function hipblasCrot(handle, n, x, incx, y, incy, c, s) &
                result(c_int) &
                bind(c, name = 'hipblasCrot')
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
        end function hipblasCrot
    end interface

    interface
        function hipblasCsrot(handle, n, x, incx, y, incy, c, s) &
                result(c_int) &
                bind(c, name = 'hipblasCsrot')
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
        end function hipblasCsrot
    end interface

    interface
        function hipblasZrot(handle, n, x, incx, y, incy, c, s) &
                result(c_int) &
                bind(c, name = 'hipblasZrot')
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
        end function hipblasZrot
    end interface

    interface
        function hipblasZdrot(handle, n, x, incx, y, incy, c, s) &
                result(c_int) &
                bind(c, name = 'hipblasZdrot')
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
        end function hipblasZdrot
    end interface

    ! rotBatched
    interface
        function hipblasSrotBatched(handle, n, x, incx, y, incy, c, s, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasSrotBatched')
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
        end function hipblasSrotBatched
    end interface

    interface
        function hipblasDrotBatched(handle, n, x, incx, y, incy, c, s, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasDrotBatched')
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
        end function hipblasDrotBatched
    end interface

    interface
        function hipblasCrotBatched(handle, n, x, incx, y, incy, c, s, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasCrotBatched')
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
        end function hipblasCrotBatched
    end interface

    interface
        function hipblasCsrotBatched(handle, n, x, incx, y, incy, c, s, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasCsrotBatched')
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
        end function hipblasCsrotBatched
    end interface

    interface
        function hipblasZrotBatched(handle, n, x, incx, y, incy, c, s, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasZrotBatched')
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
        end function hipblasZrotBatched
    end interface

    interface
        function hipblasZdrotBatched(handle, n, x, incx, y, incy, c, s, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasZdrotBatched')
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
        end function hipblasZdrotBatched
    end interface

    ! rotStridedBatched
    interface
        function hipblasSrotStridedBatched(handle, n, x, incx, stride_x, y, incy, stride_y, c, s, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasSrotStridedBatched')
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
        end function hipblasSrotStridedBatched
    end interface

    interface
        function hipblasDrotStridedBatched(handle, n, x, incx, stride_x, y, incy, stride_y, c, s, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasDrotStridedBatched')
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
        end function hipblasDrotStridedBatched
    end interface

    interface
        function hipblasCrotStridedBatched(handle, n, x, incx, stride_x, y, incy, stride_y, c, s, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasCrotStridedBatched')
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
        end function hipblasCrotStridedBatched
    end interface

    interface
        function hipblasCsrotStridedBatched(handle, n, x, incx, stride_x, y, incy, stride_y, c, s, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasCsrotStridedBatched')
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
        end function hipblasCsrotStridedBatched
    end interface

    interface
        function hipblasZrotStridedBatched(handle, n, x, incx, stride_x, y, incy, stride_y, c, s, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasZrotStridedBatched')
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
        end function hipblasZrotStridedBatched
    end interface

    interface
        function hipblasZdrotStridedBatched(handle, n, x, incx, stride_x, y, incy, stride_y, c, s, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasZdrotStridedBatched')
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
        end function hipblasZdrotStridedBatched
    end interface

    ! rotg
    interface
        function hipblasSrotg(handle, a, b, c, s) &
                result(c_int) &
                bind(c, name = 'hipblasSrotg')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            type(c_ptr), value :: a
            type(c_ptr), value :: b
            type(c_ptr), value :: c
            type(c_ptr), value :: s
        end function hipblasSrotg
    end interface

    interface
        function hipblasDrotg(handle, a, b, c, s) &
                result(c_int) &
                bind(c, name = 'hipblasDrotg')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            type(c_ptr), value :: a
            type(c_ptr), value :: b
            type(c_ptr), value :: c
            type(c_ptr), value :: s
        end function hipblasDrotg
    end interface

    interface
        function hipblasCrotg(handle, a, b, c, s) &
                result(c_int) &
                bind(c, name = 'hipblasCrotg')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            type(c_ptr), value :: a
            type(c_ptr), value :: b
            type(c_ptr), value :: c
            type(c_ptr), value :: s
        end function hipblasCrotg
    end interface

    interface
        function hipblasZrotg(handle, a, b, c, s) &
                result(c_int) &
                bind(c, name = 'hipblasZrotg')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            type(c_ptr), value :: a
            type(c_ptr), value :: b
            type(c_ptr), value :: c
            type(c_ptr), value :: s
        end function hipblasZrotg
    end interface

    ! rotgBatched
    interface
        function hipblasSrotgBatched(handle, a, b, c, s, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasSrotgBatched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            type(c_ptr), value :: a
            type(c_ptr), value :: b
            type(c_ptr), value :: c
            type(c_ptr), value :: s
            integer(c_int), value :: batch_count
        end function hipblasSrotgBatched
    end interface

    interface
        function hipblasDrotgBatched(handle, a, b, c, s, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasDrotgBatched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            type(c_ptr), value :: a
            type(c_ptr), value :: b
            type(c_ptr), value :: c
            type(c_ptr), value :: s
            integer(c_int), value :: batch_count
        end function hipblasDrotgBatched
    end interface

    interface
        function hipblasCrotgBatched(handle, a, b, c, s, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasCrotgBatched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            type(c_ptr), value :: a
            type(c_ptr), value :: b
            type(c_ptr), value :: c
            type(c_ptr), value :: s
            integer(c_int), value :: batch_count
        end function hipblasCrotgBatched
    end interface

    interface
        function hipblasZrotgBatched(handle, a, b, c, s, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasZrotgBatched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            type(c_ptr), value :: a
            type(c_ptr), value :: b
            type(c_ptr), value :: c
            type(c_ptr), value :: s
            integer(c_int), value :: batch_count
        end function hipblasZrotgBatched
    end interface

    ! rotgStridedBatched
    interface
        function hipblasSrotgStridedBatched(handle, a, stride_a, b, stride_b, c, stride_c, s, stride_s, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasSrotgStridedBatched')
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
        end function hipblasSrotgStridedBatched
    end interface

    interface
        function hipblasDrotgStridedBatched(handle, a, stride_a, b, stride_b, c, stride_c, s, stride_s, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasDrotgStridedBatched')
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
        end function hipblasDrotgStridedBatched
    end interface

    interface
        function hipblasCrotgStridedBatched(handle, a, stride_a, b, stride_b, c, stride_c, s, stride_s, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasCrotgStridedBatched')
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
        end function hipblasCrotgStridedBatched
    end interface

    interface
        function hipblasZrotgStridedBatched(handle, a, stride_a, b, stride_b, c, stride_c, s, stride_s, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasZrotgStridedBatched')
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
        end function hipblasZrotgStridedBatched
    end interface

    ! rotm
    interface
        function hipblasSrotm(handle, n, x, incx, y, incy, param) &
                result(c_int) &
                bind(c, name = 'hipblasSrotm')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            type(c_ptr), value :: param
        end function hipblasSrotm
    end interface

    interface
        function hipblasDrotm(handle, n, x, incx, y, incy, param) &
                result(c_int) &
                bind(c, name = 'hipblasDrotm')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            type(c_ptr), value :: param
        end function hipblasDrotm
    end interface

    ! rotmBatched
    interface
        function hipblasSrotmBatched(handle, n, x, incx, y, incy, param, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasSrotmBatched')
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
        end function hipblasSrotmBatched
    end interface

    interface
        function hipblasDrotmBatched(handle, n, x, incx, y, incy, param, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasDrotmBatched')
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
        end function hipblasDrotmBatched
    end interface

    ! rotmStridedBatched
    interface
        function hipblasSrotmStridedBatched(handle, n, x, incx, stride_x, y, incy, stride_y, param, stride_param, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasSrotmStridedBatched')
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
        end function hipblasSrotmStridedBatched
    end interface

    interface
        function hipblasDrotmStridedBatched(handle, n, x, incx, stride_x, y, incy, stride_y, param, stride_param, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasDrotmStridedBatched')
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
        end function hipblasDrotmStridedBatched
    end interface

    ! rotmg
    interface
        function hipblasSrotmg(handle, d1, d2, x1, y1, param) &
                result(c_int) &
                bind(c, name = 'hipblasSrotmg')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            type(c_ptr), value :: d1
            type(c_ptr), value :: d2
            type(c_ptr), value :: x1
            type(c_ptr), value :: y1
            type(c_ptr), value :: param
        end function hipblasSrotmg
    end interface

    interface
        function hipblasDrotmg(handle, d1, d2, x1, y1, param) &
                result(c_int) &
                bind(c, name = 'hipblasDrotmg')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            type(c_ptr), value :: d1
            type(c_ptr), value :: d2
            type(c_ptr), value :: x1
            type(c_ptr), value :: y1
            type(c_ptr), value :: param
        end function hipblasDrotmg
    end interface

    ! rotmgBatched
    interface
        function hipblasSrotmgBatched(handle, d1, d2, x1, y1, param, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasSrotmgBatched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            type(c_ptr), value :: d1
            type(c_ptr), value :: d2
            type(c_ptr), value :: x1
            type(c_ptr), value :: y1
            type(c_ptr), value :: param
            integer(c_int), value :: batch_count
        end function hipblasSrotmgBatched
    end interface

    interface
        function hipblasDrotmgBatched(handle, d1, d2, x1, y1, param, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasDrotmgBatched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            type(c_ptr), value :: d1
            type(c_ptr), value :: d2
            type(c_ptr), value :: x1
            type(c_ptr), value :: y1
            type(c_ptr), value :: param
            integer(c_int), value :: batch_count
        end function hipblasDrotmgBatched
    end interface

    ! rotmgStridedBatched
    interface
        function hipblasSrotmgStridedBatched(handle, d1, stride_d1, d2, stride_d2, x1, stride_x1, &
             y1, stride_y1, param, stride_param, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasSrotmgStridedBatched')
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
        end function hipblasSrotmgStridedBatched
    end interface

    interface
        function hipblasDrotmgStridedBatched(handle, d1, stride_d1, d2, stride_d2, x1, stride_x1, &
             y1, stride_y1, param, stride_param, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasDrotmgStridedBatched')
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
        end function hipblasDrotmgStridedBatched
    end interface

    !--------!
    ! blas 2 !
    !--------!

    ! gbmv
    interface
        function hipblasSgbmv(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy) &
                result(c_int) &
                bind(c, name = 'hipblasSgbmv')
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
        end function hipblasSgbmv
    end interface

    interface
        function hipblasDgbmv(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy) &
                result(c_int) &
                bind(c, name = 'hipblasDgbmv')
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
        end function hipblasDgbmv
    end interface

    interface
        function hipblasCgbmv(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy) &
                result(c_int) &
                bind(c, name = 'hipblasCgbmv')
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
        end function hipblasCgbmv
    end interface

    interface
        function hipblasZgbmv(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy) &
                result(c_int) &
                bind(c, name = 'hipblasZgbmv')
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
        end function hipblasZgbmv
    end interface

    ! gbmvBatched
    interface
        function hipblasSgbmvBatched(handle, trans, m, n, kl, ku, alpha, A, lda, &
                x, incx, beta, y, incy, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasSgbmvBatched')
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
        end function hipblasSgbmvBatched
    end interface

    interface
        function hipblasDgbmvBatched(handle, trans, m, n, kl, ku, alpha, A, lda, &
                x, incx, beta, y, incy, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasDgbmvBatched')
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
        end function hipblasDgbmvBatched
    end interface

    interface
        function hipblasCgbmvBatched(handle, trans, m, n, kl, ku, alpha, A, lda, &
                x, incx, beta, y, incy, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasCgbmvBatched')
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
        end function hipblasCgbmvBatched
    end interface

    interface
        function hipblasZgbmvBatched(handle, trans, m, n, kl, ku, alpha, A, lda, &
                x, incx, beta, y, incy, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasZgbmvBatched')
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
        end function hipblasZgbmvBatched
    end interface

    ! gbmvStridedBatched
    interface
        function hipblasSgbmvStridedBatched(handle, trans, m, n, kl, ku, alpha, A, lda, stride_A, &
                x, incx, stride_x, beta, y, incy, stride_y, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasSgbmvStridedBatched')
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
        end function hipblasSgbmvStridedBatched
    end interface

    interface
        function hipblasDgbmvStridedBatched(handle, trans, m, n, kl, ku, alpha, A, lda, stride_A, &
                x, incx, stride_x, beta, y, incy, stride_y, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasDgbmvStridedBatched')
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
        end function hipblasDgbmvStridedBatched
    end interface

    interface
        function hipblasCgbmvStridedBatched(handle, trans, m, n, kl, ku, alpha, A, lda, stride_A, &
                x, incx, stride_x, beta, y, incy, stride_y, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasCgbmvStridedBatched')
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
        end function hipblasCgbmvStridedBatched
    end interface

    interface
        function hipblasZgbmvStridedBatched(handle, trans, m, n, kl, ku, alpha, A, lda, stride_A, &
                x, incx, stride_x, beta, y, incy, stride_y, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasZgbmvStridedBatched')
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
        end function hipblasZgbmvStridedBatched
    end interface

    ! gemv
    interface
        function hipblasSgemv(handle, trans, m, n, alpha, A, lda, &
                x, incx, beta, y, incy) &
                result(c_int) &
                bind(c, name = 'hipblasSgemv')
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
        end function hipblasSgemv
    end interface

    interface
        function hipblasDgemv(handle, trans, m, n, alpha, A, lda, &
                x, incx, beta, y, incy) &
                result(c_int) &
                bind(c, name = 'hipblasDgemv')
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
        end function hipblasDgemv
    end interface

    interface
        function hipblasCgemv(handle, trans, m, n, alpha, A, lda, &
                x, incx, beta, y, incy) &
                result(c_int) &
                bind(c, name = 'hipblasCgemv')
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
        end function hipblasCgemv
    end interface

    interface
        function hipblasZgemv(handle, trans, m, n, alpha, A, lda, &
                x, incx, beta, y, incy) &
                result(c_int) &
                bind(c, name = 'hipblasZgemv')
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
        end function hipblasZgemv
    end interface

    ! gemvBatched
    interface
        function hipblasSgemvBatched(handle, trans, m, n, alpha, A, lda, &
                x, incx, beta, y, incy, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasSgemvBatched')
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
        end function hipblasSgemvBatched
    end interface

    interface
        function hipblasDgemvBatched(handle, trans, m, n, alpha, A, lda, &
                x, incx, beta, y, incy, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasDgemvBatched')
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
        end function hipblasDgemvBatched
    end interface

    interface
        function hipblasCgemvBatched(handle, trans, m, n, alpha, A, lda, &
                x, incx, beta, y, incy, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasCgemvBatched')
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
        end function hipblasCgemvBatched
    end interface

    interface
        function hipblasZgemvBatched(handle, trans, m, n, alpha, A, lda, &
                x, incx, beta, y, incy, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasZgemvBatched')
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
        end function hipblasZgemvBatched
    end interface

    ! gemvStridedBatched
    interface
        function hipblasSgemvStridedBatched(handle, trans, m, n, alpha, A, lda, stride_A, &
                x, incx, stride_x, beta, y, incy, stride_y, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasSgemvStridedBatched')
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
        end function hipblasSgemvStridedBatched
    end interface

    interface
        function hipblasDgemvStridedBatched(handle, trans, m, n, alpha, A, lda, stride_A, &
                x, incx, stride_x, beta, y, incy, stride_y, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasDgemvStridedBatched')
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
        end function hipblasDgemvStridedBatched
    end interface

    interface
        function hipblasCgemvStridedBatched(handle, trans, m, n, alpha, A, lda, stride_A, &
                x, incx, stride_x, beta, y, incy, stride_y, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasCgemvStridedBatched')
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
        end function hipblasCgemvStridedBatched
    end interface

    interface
        function hipblasZgemvStridedBatched(handle, trans, m, n, alpha, A, lda, stride_A, &
                x, incx, stride_x, beta, y, incy, stride_y, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasZgemvStridedBatched')
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
        end function hipblasZgemvStridedBatched
    end interface

    ! hbmv
    interface
        function hipblasChbmv(handle, uplo, n, k, alpha, A, lda, &
                x, incx, beta, y, incy) &
                result(c_int) &
                bind(c, name = 'hipblasChbmv')
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
        end function hipblasChbmv
    end interface

    interface
        function hipblasZhbmv(handle, uplo, n, k, alpha, A, lda, &
                x, incx, beta, y, incy) &
                result(c_int) &
                bind(c, name = 'hipblasZhbmv')
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
        end function hipblasZhbmv
    end interface

    ! hbmvBatched
    interface
        function hipblasChbmvBatched(handle, uplo, n, k, alpha, A, lda, &
                x, incx, beta, y, incy, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasChbmvBatched')
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
        end function hipblasChbmvBatched
    end interface

    interface
        function hipblasZhbmvBatched(handle, uplo, n, k, alpha, A, lda, &
                x, incx, beta, y, incy, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasZhbmvBatched')
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
        end function hipblasZhbmvBatched
    end interface

    ! hbmvStridedBatched
    interface
        function hipblasChbmvStridedBatched(handle, uplo, n, k, alpha, A, lda, stride_A, &
                x, incx, stride_x, beta, y, incy, stride_y, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasChbmvStridedBatched')
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
        end function hipblasChbmvStridedBatched
    end interface

    interface
        function hipblasZhbmvStridedBatched(handle, uplo, n, k, alpha, A, lda, stride_A, &
                x, incx, stride_x, beta, y, incy, stride_y, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasZhbmvStridedBatched')
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
        end function hipblasZhbmvStridedBatched
    end interface

    ! hemv
    interface
        function hipblasChemv(handle, uplo, n, alpha, A, lda, &
                x, incx, beta, y, incy) &
                result(c_int) &
                bind(c, name = 'hipblasChemv')
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
        end function hipblasChemv
    end interface

    interface
        function hipblasZhemv(handle, uplo, n, alpha, A, lda, &
                x, incx, beta, y, incy) &
                result(c_int) &
                bind(c, name = 'hipblasZhemv')
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
        end function hipblasZhemv
    end interface

    ! hemvBatched
    interface
        function hipblasChemvBatched(handle, uplo, n, alpha, A, lda, &
                x, incx, beta, y, incy, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasChemvBatched')
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
        end function hipblasChemvBatched
    end interface

    interface
        function hipblasZhemvBatched(handle, uplo, n, alpha, A, lda, &
                x, incx, beta, y, incy, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasZhemvBatched')
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
        end function hipblasZhemvBatched
    end interface

    ! hemvStridedBatched
    interface
        function hipblasChemvStridedBatched(handle, uplo, n, alpha, A, lda, stride_A, &
                x, incx, stride_x, beta, y, incy, stride_y, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasChemvStridedBatched')
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
        end function hipblasChemvStridedBatched
    end interface

    interface
        function hipblasZhemvStridedBatched(handle, uplo, n, alpha, A, lda, stride_A, &
                x, incx, stride_x, beta, y, incy, stride_y, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasZhemvStridedBatched')
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
        end function hipblasZhemvStridedBatched
    end interface

    ! her
    interface
        function hipblasCher(handle, uplo, n, alpha, &
                x, incx, A, lda) &
                result(c_int) &
                bind(c, name = 'hipblasCher')
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
        end function hipblasCher
    end interface

    interface
        function hipblasZher(handle, uplo, n, alpha, &
                x, incx, A, lda) &
                result(c_int) &
                bind(c, name = 'hipblasZher')
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
        end function hipblasZher
    end interface

    ! herBatched
    interface
        function hipblasCherBatched(handle, uplo, n, alpha, &
                x, incx, A, lda, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasCherBatched')
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
        end function hipblasCherBatched
    end interface

    interface
        function hipblasZherBatched(handle, uplo, n, alpha, &
                x, incx, A, lda, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasZherBatched')
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
        end function hipblasZherBatched
    end interface

    ! herStridedBatched
    interface
        function hipblasCherStridedBatched(handle, uplo, n, alpha, &
                x, incx, stride_x, A, lda, stride_A, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasCherStridedBatched')
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
        end function hipblasCherStridedBatched
    end interface

    interface
        function hipblasZherStridedBatched(handle, uplo, n, alpha, &
                x, incx, stride_x, A, lda, stride_A, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasZherStridedBatched')
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
        end function hipblasZherStridedBatched
    end interface

    ! her2
    interface
        function hipblasCher2(handle, uplo, n, alpha, &
                x, incx, y, incy, A, lda) &
                result(c_int) &
                bind(c, name = 'hipblasCher2')
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
        end function hipblasCher2
    end interface

    interface
        function hipblasZher2(handle, uplo, n, alpha, &
                x, incx, y, incy, A, lda) &
                result(c_int) &
                bind(c, name = 'hipblasZher2')
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
        end function hipblasZher2
    end interface

    ! her2Batched
    interface
        function hipblasCher2Batched(handle, uplo, n, alpha, &
                x, incx, y, incy, A, lda, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasCher2Batched')
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
        end function hipblasCher2Batched
    end interface

    interface
        function hipblasZher2Batched(handle, uplo, n, alpha, &
                x, incx, y, incy, A, lda, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasZher2Batched')
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
        end function hipblasZher2Batched
    end interface

    ! her2StridedBatched
    interface
        function hipblasCher2StridedBatched(handle, uplo, n, alpha, &
                x, incx, stride_x, y, incy, stride_y, A, lda, stride_A, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasCher2StridedBatched')
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
        end function hipblasCher2StridedBatched
    end interface

    interface
        function hipblasZher2StridedBatched(handle, uplo, n, alpha, &
                x, incx, stride_x, y, incy, stride_y, A, lda, stride_A, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasZher2StridedBatched')
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
        end function hipblasZher2StridedBatched
    end interface

    ! hpmv
    interface
        function hipblasChpmv(handle, uplo, n, alpha, AP, &
                x, incx, beta, y, incy) &
                result(c_int) &
                bind(c, name = 'hipblasChpmv')
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
        end function hipblasChpmv
    end interface

    interface
        function hipblasZhpmv(handle, uplo, n, alpha, AP, &
                x, incx, beta, y, incy) &
                result(c_int) &
                bind(c, name = 'hipblasZhpmv')
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
        end function hipblasZhpmv
    end interface

    ! hpmvBatched
    interface
        function hipblasChpmvBatched(handle, uplo, n, alpha, AP, &
                x, incx, beta, y, incy, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasChpmvBatched')
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
        end function hipblasChpmvBatched
    end interface

    interface
        function hipblasZhpmvBatched(handle, uplo, n, alpha, AP, &
                x, incx, beta, y, incy, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasZhpmvBatched')
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
        end function hipblasZhpmvBatched
    end interface

    ! hpmvStridedBatched
    interface
        function hipblasChpmvStridedBatched(handle, uplo, n, alpha, AP, stride_AP, &
                x, incx, stride_x, beta, y, incy, stride_y, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasChpmvStridedBatched')
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
        end function hipblasChpmvStridedBatched
    end interface

    interface
        function hipblasZhpmvStridedBatched(handle, uplo, n, alpha, AP, stride_AP, &
                x, incx, stride_x, beta, y, incy, stride_y, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasZhpmvStridedBatched')
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
        end function hipblasZhpmvStridedBatched
    end interface

    ! hpr
    interface
        function hipblasChpr(handle, uplo, n, alpha, &
                x, incx, AP) &
                result(c_int) &
                bind(c, name = 'hipblasChpr')
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
        end function hipblasChpr
    end interface

    interface
        function hipblasZhpr(handle, uplo, n, alpha, &
                x, incx, AP) &
                result(c_int) &
                bind(c, name = 'hipblasZhpr')
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
        end function hipblasZhpr
    end interface

    ! hprBatched
    interface
        function hipblasChprBatched(handle, uplo, n, alpha, &
                x, incx, AP, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasChprBatched')
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
        end function hipblasChprBatched
    end interface

    interface
        function hipblasZhprBatched(handle, uplo, n, alpha, &
                x, incx, AP, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasZhprBatched')
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
        end function hipblasZhprBatched
    end interface

    ! hprStridedBatched
    interface
        function hipblasChprStridedBatched(handle, uplo, n, alpha, &
                x, incx, stride_x, AP, stride_AP, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasChprStridedBatched')
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
        end function hipblasChprStridedBatched
    end interface

    interface
        function hipblasZhprStridedBatched(handle, uplo, n, alpha, &
                x, incx, stride_x, AP, stride_AP, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasZhprStridedBatched')
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
        end function hipblasZhprStridedBatched
    end interface

    ! hpr2
    interface
        function hipblasChpr2(handle, uplo, n, alpha, &
                x, incx, y, incy, AP) &
                result(c_int) &
                bind(c, name = 'hipblasChpr2')
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
        end function hipblasChpr2
    end interface

    interface
        function hipblasZhpr2(handle, uplo, n, alpha, &
                x, incx, y, incy, AP) &
                result(c_int) &
                bind(c, name = 'hipblasZhpr2')
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
        end function hipblasZhpr2
    end interface

    ! hpr2Batched
    interface
        function hipblasChpr2Batched(handle, uplo, n, alpha, &
                x, incx, y, incy, AP, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasChpr2Batched')
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
        end function hipblasChpr2Batched
    end interface

    interface
        function hipblasZhpr2Batched(handle, uplo, n, alpha, &
                x, incx, y, incy, AP, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasZhpr2Batched')
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
        end function hipblasZhpr2Batched
    end interface

    ! hpr2StridedBatched
    interface
        function hipblasChpr2StridedBatched(handle, uplo, n, alpha, &
                x, incx, stride_x, y, incy, stride_y, AP, stride_AP, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasChpr2StridedBatched')
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
        end function hipblasChpr2StridedBatched
    end interface

    interface
        function hipblasZhpr2StridedBatched(handle, uplo, n, alpha, &
                x, incx, stride_x, y, incy, stride_y, AP, stride_AP, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasZhpr2StridedBatched')
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
        end function hipblasZhpr2StridedBatched
    end interface

    ! trmv
    interface
        function hipblasStrmv(handle, uplo, transA, diag, m, &
                A, lda, x, incx) &
                result(c_int) &
                bind(c, name = 'hipblasStrmv')
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
        end function hipblasStrmv
    end interface

    interface
        function hipblasDtrmv(handle, uplo, transA, diag, m, &
                A, lda, x, incx) &
                result(c_int) &
                bind(c, name = 'hipblasDtrmv')
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
        end function hipblasDtrmv
    end interface

    interface
        function hipblasCtrmv(handle, uplo, transA, diag, m, &
                A, lda, x, incx) &
                result(c_int) &
                bind(c, name = 'hipblasCtrmv')
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
        end function hipblasCtrmv
    end interface

    interface
        function hipblasZtrmv(handle, uplo, transA, diag, m, &
                A, lda, x, incx) &
                result(c_int) &
                bind(c, name = 'hipblasZtrmv')
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
        end function hipblasZtrmv
    end interface

    ! trmvBatched
    interface
        function hipblasStrmvBatched(handle, uplo, transA, diag, m, &
                A, lda, x, incx, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasStrmvBatched')
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
        end function hipblasStrmvBatched
    end interface

    interface
        function hipblasDtrmvBatched(handle, uplo, transA, diag, m, &
                A, lda, x, incx, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasDtrmvBatched')
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
        end function hipblasDtrmvBatched
    end interface

    interface
        function hipblasCtrmvBatched(handle, uplo, transA, diag, m, &
                A, lda, x, incx, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasCtrmvBatched')
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
        end function hipblasCtrmvBatched
    end interface

    interface
        function hipblasZtrmvBatched(handle, uplo, transA, diag, m, &
                A, lda, x, incx, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasZtrmvBatched')
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
        end function hipblasZtrmvBatched
    end interface

    ! trmvStridedBatched
    interface
        function hipblasStrmvStridedBatched(handle, uplo, transA, diag, m, &
                A, lda, stride_A, x, incx, stride_x, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasStrmvStridedBatched')
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
        end function hipblasStrmvStridedBatched
    end interface

    interface
        function hipblasDtrmvStridedBatched(handle, uplo, transA, diag, m, &
                A, lda, stride_A, x, incx, stride_x, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasDtrmvStridedBatched')
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
        end function hipblasDtrmvStridedBatched
    end interface

    interface
        function hipblasCtrmvStridedBatched(handle, uplo, transA, diag, m, &
                A, lda, stride_A, x, incx, stride_x, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasCtrmvStridedBatched')
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
        end function hipblasCtrmvStridedBatched
    end interface

    interface
        function hipblasZtrmvStridedBatched(handle, uplo, transA, diag, m, &
                A, lda, stride_A, x, incx, stride_x, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasZtrmvStridedBatched')
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
        end function hipblasZtrmvStridedBatched
    end interface

    ! tpmv
    interface
        function hipblasStpmv(handle, uplo, transA, diag, m, &
                AP, x, incx) &
                result(c_int) &
                bind(c, name = 'hipblasStpmv')
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
        end function hipblasStpmv
    end interface

    interface
        function hipblasDtpmv(handle, uplo, transA, diag, m, &
                AP, x, incx) &
                result(c_int) &
                bind(c, name = 'hipblasDtpmv')
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
        end function hipblasDtpmv
    end interface

    interface
        function hipblasCtpmv(handle, uplo, transA, diag, m, &
                AP, x, incx) &
                result(c_int) &
                bind(c, name = 'hipblasCtpmv')
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
        end function hipblasCtpmv
    end interface

    interface
        function hipblasZtpmv(handle, uplo, transA, diag, m, &
                AP, x, incx) &
                result(c_int) &
                bind(c, name = 'hipblasZtpmv')
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
        end function hipblasZtpmv
    end interface

    ! tpmvBatched
    interface
        function hipblasStpmvBatched(handle, uplo, transA, diag, m, &
                AP, x, incx, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasStpmvBatched')
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
        end function hipblasStpmvBatched
    end interface

    interface
        function hipblasDtpmvBatched(handle, uplo, transA, diag, m, &
                AP, x, incx, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasDtpmvBatched')
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
        end function hipblasDtpmvBatched
    end interface

    interface
        function hipblasCtpmvBatched(handle, uplo, transA, diag, m, &
                AP, x, incx, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasCtpmvBatched')
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
        end function hipblasCtpmvBatched
    end interface

    interface
        function hipblasZtpmvBatched(handle, uplo, transA, diag, m, &
                AP, x, incx, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasZtpmvBatched')
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
        end function hipblasZtpmvBatched
    end interface

    ! tpmvStridedBatched
    interface
        function hipblasStpmvStridedBatched(handle, uplo, transA, diag, m, &
                AP, stride_AP, x, incx, stride_x, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasStpmvStridedBatched')
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
        end function hipblasStpmvStridedBatched
    end interface

    interface
        function hipblasDtpmvStridedBatched(handle, uplo, transA, diag, m, &
                AP, stride_AP, x, incx, stride_x, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasDtpmvStridedBatched')
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
        end function hipblasDtpmvStridedBatched
    end interface

    interface
        function hipblasCtpmvStridedBatched(handle, uplo, transA, diag, m, &
                AP, stride_AP, x, incx, stride_x, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasCtpmvStridedBatched')
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
        end function hipblasCtpmvStridedBatched
    end interface

    interface
        function hipblasZtpmvStridedBatched(handle, uplo, transA, diag, m, &
                AP, stride_AP, x, incx, stride_x, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasZtpmvStridedBatched')
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
        end function hipblasZtpmvStridedBatched
    end interface

    ! tbmv
    interface
        function hipblasStbmv(handle, uplo, transA, diag, m, k, &
                A, lda, x, incx) &
                result(c_int) &
                bind(c, name = 'hipblasStbmv')
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
        end function hipblasStbmv
    end interface

    interface
        function hipblasDtbmv(handle, uplo, transA, diag, m, k, &
                A, lda, x, incx) &
                result(c_int) &
                bind(c, name = 'hipblasDtbmv')
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
        end function hipblasDtbmv
    end interface

    interface
        function hipblasCtbmv(handle, uplo, transA, diag, m, k, &
                A, lda, x, incx) &
                result(c_int) &
                bind(c, name = 'hipblasCtbmv')
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
        end function hipblasCtbmv
    end interface

    interface
        function hipblasZtbmv(handle, uplo, transA, diag, m, k, &
                A, lda, x, incx) &
                result(c_int) &
                bind(c, name = 'hipblasZtbmv')
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
        end function hipblasZtbmv
    end interface

    ! tbmvBatched
    interface
        function hipblasStbmvBatched(handle, uplo, transA, diag, m, k, &
                A, lda, x, incx, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasStbmvBatched')
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
        end function hipblasStbmvBatched
    end interface

    interface
        function hipblasDtbmvBatched(handle, uplo, transA, diag, m, k, &
                A, lda, x, incx, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasDtbmvBatched')
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
        end function hipblasDtbmvBatched
    end interface

    interface
        function hipblasCtbmvBatched(handle, uplo, transA, diag, m, k, &
                A, lda, x, incx, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasCtbmvBatched')
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
        end function hipblasCtbmvBatched
    end interface

    interface
        function hipblasZtbmvBatched(handle, uplo, transA, diag, m, k, &
                A, lda, x, incx, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasZtbmvBatched')
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
        end function hipblasZtbmvBatched
    end interface

    ! tbmvStridedBatched
    interface
        function hipblasStbmvStridedBatched(handle, uplo, transA, diag, m, k, &
                A, lda, stride_A, x, incx, stride_x, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasStbmvStridedBatched')
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
        end function hipblasStbmvStridedBatched
    end interface

    interface
        function hipblasDtbmvStridedBatched(handle, uplo, transA, diag, m, k, &
                A, lda, stride_A, x, incx, stride_x, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasDtbmvStridedBatched')
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
        end function hipblasDtbmvStridedBatched
    end interface

    interface
        function hipblasCtbmvStridedBatched(handle, uplo, transA, diag, m, k, &
                A, lda, stride_A, x, incx, stride_x, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasCtbmvStridedBatched')
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
        end function hipblasCtbmvStridedBatched
    end interface

    interface
        function hipblasZtbmvStridedBatched(handle, uplo, transA, diag, m, k, &
                A, lda, stride_A, x, incx, stride_x, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasZtbmvStridedBatched')
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
        end function hipblasZtbmvStridedBatched
    end interface

    ! tbsv
    interface
        function hipblasStbsv(handle, uplo, transA, diag, n, k, &
                A, lda, x, incx) &
                result(c_int) &
                bind(c, name = 'hipblasStbsv')
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
        end function hipblasStbsv
    end interface

    interface
        function hipblasDtbsv(handle, uplo, transA, diag, n, k, &
                A, lda, x, incx) &
                result(c_int) &
                bind(c, name = 'hipblasDtbsv')
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
        end function hipblasDtbsv
    end interface

    interface
        function hipblasCtbsv(handle, uplo, transA, diag, n, k, &
                A, lda, x, incx) &
                result(c_int) &
                bind(c, name = 'hipblasCtbsv')
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
        end function hipblasCtbsv
    end interface

    interface
        function hipblasZtbsv(handle, uplo, transA, diag, n, k, &
                A, lda, x, incx) &
                result(c_int) &
                bind(c, name = 'hipblasZtbsv')
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
        end function hipblasZtbsv
    end interface

    ! tbsvBatched
    interface
        function hipblasStbsvBatched(handle, uplo, transA, diag, n, k, &
                A, lda, x, incx, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasStbsvBatched')
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
        end function hipblasStbsvBatched
    end interface

    interface
        function hipblasDtbsvBatched(handle, uplo, transA, diag, n, k, &
                A, lda, x, incx, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasDtbsvBatched')
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
        end function hipblasDtbsvBatched
    end interface

    interface
        function hipblasCtbsvBatched(handle, uplo, transA, diag, n, k, &
                A, lda, x, incx, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasCtbsvBatched')
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
        end function hipblasCtbsvBatched
    end interface

    interface
        function hipblasZtbsvBatched(handle, uplo, transA, diag, n, k, &
                A, lda, x, incx, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasZtbsvBatched')
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
        end function hipblasZtbsvBatched
    end interface

    ! tbsvStridedBatched
    interface
        function hipblasStbsvStridedBatched(handle, uplo, transA, diag, n, k, &
                A, lda, stride_A, x, incx, stride_x, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasStbsvStridedBatched')
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
        end function hipblasStbsvStridedBatched
    end interface

    interface
        function hipblasDtbsvStridedBatched(handle, uplo, transA, diag, n, k, &
                A, lda, stride_A, x, incx, stride_x, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasDtbsvStridedBatched')
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
        end function hipblasDtbsvStridedBatched
    end interface

    interface
        function hipblasCtbsvStridedBatched(handle, uplo, transA, diag, n, k, &
                A, lda, stride_A, x, incx, stride_x, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasCtbsvStridedBatched')
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
        end function hipblasCtbsvStridedBatched
    end interface

    interface
        function hipblasZtbsvStridedBatched(handle, uplo, transA, diag, n, k, &
                A, lda, stride_A, x, incx, stride_x, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasZtbsvStridedBatched')
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
        end function hipblasZtbsvStridedBatched
    end interface

    ! trsv
    interface
        function hipblasStrsv(handle, uplo, transA, diag, m, &
                A, lda, x, incx) &
                result(c_int) &
                bind(c, name = 'hipblasStrsv')
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
        end function hipblasStrsv
    end interface

    interface
        function hipblasDtrsv(handle, uplo, transA, diag, m, &
                A, lda, x, incx) &
                result(c_int) &
                bind(c, name = 'hipblasDtrsv')
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
        end function hipblasDtrsv
    end interface

    interface
        function hipblasCtrsv(handle, uplo, transA, diag, m, &
                A, lda, x, incx) &
                result(c_int) &
                bind(c, name = 'hipblasCtrsv')
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
        end function hipblasCtrsv
    end interface

    interface
        function hipblasZtrsv(handle, uplo, transA, diag, m, &
                A, lda, x, incx) &
                result(c_int) &
                bind(c, name = 'hipblasZtrsv')
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
        end function hipblasZtrsv
    end interface

    ! trsvBatched
    interface
        function hipblasStrsvBatched(handle, uplo, transA, diag, m, &
                A, lda, x, incx, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasStrsvBatched')
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
        end function hipblasStrsvBatched
    end interface

    interface
        function hipblasDtrsvBatched(handle, uplo, transA, diag, m, &
                A, lda, x, incx, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasDtrsvBatched')
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
        end function hipblasDtrsvBatched
    end interface

    interface
        function hipblasCtrsvBatched(handle, uplo, transA, diag, m, &
                A, lda, x, incx, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasCtrsvBatched')
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
        end function hipblasCtrsvBatched
    end interface

    interface
        function hipblasZtrsvBatched(handle, uplo, transA, diag, m, &
                A, lda, x, incx, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasZtrsvBatched')
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
        end function hipblasZtrsvBatched
    end interface

    ! trsvStridedBatched
    interface
        function hipblasStrsvStridedBatched(handle, uplo, transA, diag, m, &
                A, lda, stride_A, x, incx, stride_x, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasStrsvStridedBatched')
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
        end function hipblasStrsvStridedBatched
    end interface

    interface
        function hipblasDtrsvStridedBatched(handle, uplo, transA, diag, m, &
                A, lda, stride_A, x, incx, stride_x, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasDtrsvStridedBatched')
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
        end function hipblasDtrsvStridedBatched
    end interface

    interface
        function hipblasCtrsvStridedBatched(handle, uplo, transA, diag, m, &
                A, lda, stride_A, x, incx, stride_x, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasCtrsvStridedBatched')
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
        end function hipblasCtrsvStridedBatched
    end interface

    interface
        function hipblasZtrsvStridedBatched(handle, uplo, transA, diag, m, &
                A, lda, stride_A, x, incx, stride_x, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasZtrsvStridedBatched')
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
        end function hipblasZtrsvStridedBatched
    end interface

    ! tpsv
    interface
        function hipblasStpsv(handle, uplo, transA, diag, n, &
                AP, x, incx) &
                result(c_int) &
                bind(c, name = 'hipblasStpsv')
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
        end function hipblasStpsv
    end interface

    interface
        function hipblasDtpsv(handle, uplo, transA, diag, n, &
                AP, x, incx) &
                result(c_int) &
                bind(c, name = 'hipblasDtpsv')
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
        end function hipblasDtpsv
    end interface

    interface
        function hipblasCtpsv(handle, uplo, transA, diag, n, &
                AP, x, incx) &
                result(c_int) &
                bind(c, name = 'hipblasCtpsv')
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
        end function hipblasCtpsv
    end interface

    interface
        function hipblasZtpsv(handle, uplo, transA, diag, n, &
                AP, x, incx) &
                result(c_int) &
                bind(c, name = 'hipblasZtpsv')
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
        end function hipblasZtpsv
    end interface

    ! tpsvBatched
    interface
        function hipblasStpsvBatched(handle, uplo, transA, diag, n, &
                AP, x, incx, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasStpsvBatched')
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
        end function hipblasStpsvBatched
    end interface

    interface
        function hipblasDtpsvBatched(handle, uplo, transA, diag, n, &
                AP, x, incx, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasDtpsvBatched')
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
        end function hipblasDtpsvBatched
    end interface

    interface
        function hipblasCtpsvBatched(handle, uplo, transA, diag, n, &
                AP, x, incx, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasCtpsvBatched')
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
        end function hipblasCtpsvBatched
    end interface

    interface
        function hipblasZtpsvBatched(handle, uplo, transA, diag, n, &
                AP, x, incx, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasZtpsvBatched')
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
        end function hipblasZtpsvBatched
    end interface

    ! tpsvStridedBatched
    interface
        function hipblasStpsvStridedBatched(handle, uplo, transA, diag, n, &
                AP, stride_AP, x, incx, stride_x, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasStpsvStridedBatched')
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
        end function hipblasStpsvStridedBatched
    end interface

    interface
        function hipblasDtpsvStridedBatched(handle, uplo, transA, diag, n, &
                AP, stride_AP, x, incx, stride_x, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasDtpsvStridedBatched')
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
        end function hipblasDtpsvStridedBatched
    end interface

    interface
        function hipblasCtpsvStridedBatched(handle, uplo, transA, diag, n, &
                AP, stride_AP, x, incx, stride_x, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasCtpsvStridedBatched')
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
        end function hipblasCtpsvStridedBatched
    end interface

    interface
        function hipblasZtpsvStridedBatched(handle, uplo, transA, diag, n, &
                AP, stride_AP, x, incx, stride_x, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasZtpsvStridedBatched')
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
        end function hipblasZtpsvStridedBatched
    end interface

    ! symv
    interface
        function hipblasSsymv(handle, uplo, n, alpha, A, lda, &
                x, incx, beta, y, incy) &
                result(c_int) &
                bind(c, name = 'hipblasSsymv')
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
        end function hipblasSsymv
    end interface

    interface
        function hipblasDsymv(handle, uplo, n, alpha, A, lda, &
                x, incx, beta, y, incy) &
                result(c_int) &
                bind(c, name = 'hipblasDsymv')
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
        end function hipblasDsymv
    end interface

    interface
        function hipblasCsymv(handle, uplo, n, alpha, A, lda, &
                x, incx, beta, y, incy) &
                result(c_int) &
                bind(c, name = 'hipblasCsymv')
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
        end function hipblasCsymv
    end interface

    interface
        function hipblasZsymv(handle, uplo, n, alpha, A, lda, &
                x, incx, beta, y, incy) &
                result(c_int) &
                bind(c, name = 'hipblasZsymv')
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
        end function hipblasZsymv
    end interface

    ! symvBatched
    interface
        function hipblasSsymvBatched(handle, uplo, n, alpha, A, lda, &
                x, incx, beta, y, incy, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasSsymvBatched')
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
        end function hipblasSsymvBatched
    end interface

    interface
        function hipblasDsymvBatched(handle, uplo, n, alpha, A, lda, &
                x, incx, beta, y, incy, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasDsymvBatched')
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
        end function hipblasDsymvBatched
    end interface

    interface
        function hipblasCsymvBatched(handle, uplo, n, alpha, A, lda, &
                x, incx, beta, y, incy, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasCsymvBatched')
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
        end function hipblasCsymvBatched
    end interface

    interface
        function hipblasZsymvBatched(handle, uplo, n, alpha, A, lda, &
                x, incx, beta, y, incy, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasZsymvBatched')
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
        end function hipblasZsymvBatched
    end interface

    ! symvStridedBatched
    interface
        function hipblasSsymvStridedBatched(handle, uplo, n, alpha, A, lda, stride_A, &
                x, incx, stride_x, beta, y, incy, stride_y, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasSsymvStridedBatched')
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
        end function hipblasSsymvStridedBatched
    end interface

    interface
        function hipblasDsymvStridedBatched(handle, uplo, n, alpha, A, lda, stride_A, &
                x, incx, stride_x, beta, y, incy, stride_y, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasDsymvStridedBatched')
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
        end function hipblasDsymvStridedBatched
    end interface

    interface
        function hipblasCsymvStridedBatched(handle, uplo, n, alpha, A, lda, stride_A, &
                x, incx, stride_x, beta, y, incy, stride_y, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasCsymvStridedBatched')
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
        end function hipblasCsymvStridedBatched
    end interface

    interface
        function hipblasZsymvStridedBatched(handle, uplo, n, alpha, A, lda, stride_A, &
                x, incx, stride_x, beta, y, incy, stride_y, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasZsymvStridedBatched')
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
        end function hipblasZsymvStridedBatched
    end interface

    ! spmv
    interface
        function hipblasSspmv(handle, uplo, n, alpha, AP, &
                x, incx, beta, y, incy) &
                result(c_int) &
                bind(c, name = 'hipblasSspmv')
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
        end function hipblasSspmv
    end interface

    interface
        function hipblasDspmv(handle, uplo, n, alpha, AP, &
                x, incx, beta, y, incy) &
                result(c_int) &
                bind(c, name = 'hipblasDspmv')
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
        end function hipblasDspmv
    end interface

    ! spmvBatched
    interface
        function hipblasSspmvBatched(handle, uplo, n, alpha, AP, &
                x, incx, beta, y, incy, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasSspmvBatched')
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
        end function hipblasSspmvBatched
    end interface

    interface
        function hipblasDspmvBatched(handle, uplo, n, alpha, AP, &
                x, incx, beta, y, incy, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasDspmvBatched')
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
        end function hipblasDspmvBatched
    end interface

    ! spmvStridedBatched
    interface
        function hipblasSspmvStridedBatched(handle, uplo, n, alpha, AP, stride_AP, &
                x, incx, stride_x, beta, y, incy, stride_y, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasSspmvStridedBatched')
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
        end function hipblasSspmvStridedBatched
    end interface

    interface
        function hipblasDspmvStridedBatched(handle, uplo, n, alpha, AP, stride_AP, &
                x, incx, stride_x, beta, y, incy, stride_y, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasDspmvStridedBatched')
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
        end function hipblasDspmvStridedBatched
    end interface

    ! sbmv
    interface
        function hipblasSsbmv(handle, uplo, n, k, alpha, A, lda, &
                x, incx, beta, y, incy) &
                result(c_int) &
                bind(c, name = 'hipblasSsbmv')
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
        end function hipblasSsbmv
    end interface

    interface
        function hipblasDsbmv(handle, uplo, n, k, alpha, A, lda, &
                x, incx, beta, y, incy) &
                result(c_int) &
                bind(c, name = 'hipblasDsbmv')
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
        end function hipblasDsbmv
    end interface

    ! sbmvBatched
    interface
        function hipblasSsbmvBatched(handle, uplo, n, k, alpha, A, lda, &
                x, incx, beta, y, incy, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasSsbmvBatched')
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
        end function hipblasSsbmvBatched
    end interface

    interface
        function hipblasDsbmvBatched(handle, uplo, n, k, alpha, A, lda, &
                x, incx, beta, y, incy, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasDsbmvBatched')
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
        end function hipblasDsbmvBatched
    end interface

    ! sbmvStridedBatched
    interface
        function hipblasSsbmvStridedBatched(handle, uplo, n, k, alpha, A, lda, stride_A, &
                x, incx, stride_x, beta, y, incy, stride_y, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasSsbmvStridedBatched')
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
        end function hipblasSsbmvStridedBatched
    end interface

    interface
        function hipblasDsbmvStridedBatched(handle, uplo, n, k, alpha, A, lda, stride_A, &
                x, incx, stride_x, beta, y, incy, stride_y, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasDsbmvStridedBatched')
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
        end function hipblasDsbmvStridedBatched
    end interface

    ! ger
    interface
        function hipblasSger(handle, m, n, alpha, x, incx, &
                y, incy, A, lda) &
                result(c_int) &
                bind(c, name = 'hipblasSger')
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
        end function hipblasSger
    end interface

    interface
        function hipblasDger(handle, m, n, alpha, x, incx, &
                y, incy, A, lda) &
                result(c_int) &
                bind(c, name = 'hipblasDger')
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
        end function hipblasDger
    end interface

    interface
        function hipblasCgeru(handle, m, n, alpha, x, incx, &
                y, incy, A, lda) &
                result(c_int) &
                bind(c, name = 'hipblasCgeru')
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
        end function hipblasCgeru
    end interface

    interface
        function hipblasCgerc(handle, m, n, alpha, x, incx, &
                y, incy, A, lda) &
                result(c_int) &
                bind(c, name = 'hipblasCgerc')
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
        end function hipblasCgerc
    end interface

    interface
        function hipblasZgeru(handle, m, n, alpha, x, incx, &
                y, incy, A, lda) &
                result(c_int) &
                bind(c, name = 'hipblasZgeru')
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
        end function hipblasZgeru
    end interface

    interface
        function hipblasZgerc(handle, m, n, alpha, x, incx, &
                y, incy, A, lda) &
                result(c_int) &
                bind(c, name = 'hipblasZgerc')
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
        end function hipblasZgerc
    end interface

    ! gerBatched
    interface
        function hipblasSgerBatched(handle, m, n, alpha, x, incx, &
                y, incy, A, lda, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasSgerBatched')
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
        end function hipblasSgerBatched
    end interface

    interface
        function hipblasDgerBatched(handle, m, n, alpha, x, incx, &
                y, incy, A, lda, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasDgerBatched')
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
        end function hipblasDgerBatched
    end interface

    interface
        function hipblasCgeruBatched(handle, m, n, alpha, x, incx, &
                y, incy, A, lda, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasCgeruBatched')
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
        end function hipblasCgeruBatched
    end interface

    interface
        function hipblasCgercBatched(handle, m, n, alpha, x, incx, &
                y, incy, A, lda, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasCgercBatched')
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
        end function hipblasCgercBatched
    end interface

    interface
        function hipblasZgeruBatched(handle, m, n, alpha, x, incx, &
                y, incy, A, lda, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasZgeruBatched')
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
        end function hipblasZgeruBatched
    end interface

    interface
        function hipblasZgercBatched(handle, m, n, alpha, x, incx, &
                y, incy, A, lda, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasZgercBatched')
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
        end function hipblasZgercBatched
    end interface

    ! gerStridedBatched
    interface
        function hipblasSgerStridedBatched(handle, m, n, alpha, x, incx, stride_x, &
                y, incy, stride_y, A, lda, stride_A, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasSgerStridedBatched')
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
        end function hipblasSgerStridedBatched
    end interface

    interface
        function hipblasDgerStridedBatched(handle, m, n, alpha, x, incx, stride_x, &
                y, incy, stride_y, A, lda, stride_A, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasDgerStridedBatched')
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
        end function hipblasDgerStridedBatched
    end interface

    interface
        function hipblasCgeruStridedBatched(handle, m, n, alpha, x, incx, stride_x, &
                y, incy, stride_y, A, lda, stride_A, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasCgeruStridedBatched')
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
        end function hipblasCgeruStridedBatched
    end interface

    interface
        function hipblasCgercStridedBatched(handle, m, n, alpha, x, incx, stride_x, &
                y, incy, stride_y, A, lda, stride_A, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasCgercStridedBatched')
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
        end function hipblasCgercStridedBatched
    end interface

    interface
        function hipblasZgeruStridedBatched(handle, m, n, alpha, x, incx, stride_x, &
                y, incy, stride_y, A, lda, stride_A, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasZgeruStridedBatched')
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
        end function hipblasZgeruStridedBatched
    end interface

    interface
        function hipblasZgercStridedBatched(handle, m, n, alpha, x, incx, stride_x, &
                y, incy, stride_y, A, lda, stride_A, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasZgercStridedBatched')
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
        end function hipblasZgercStridedBatched
    end interface

    ! spr
    interface
        function hipblasSspr(handle, uplo, n, alpha, x, incx, AP) &
                result(c_int) &
                bind(c, name = 'hipblasSspr')
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
        end function hipblasSspr
    end interface

    interface
        function hipblasDspr(handle, uplo, n, alpha, x, incx, AP) &
                result(c_int) &
                bind(c, name = 'hipblasDspr')
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
        end function hipblasDspr
    end interface

    interface
        function hipblasCspr(handle, uplo, n, alpha, x, incx, AP) &
                result(c_int) &
                bind(c, name = 'hipblasCspr')
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
        end function hipblasCspr
    end interface

    interface
        function hipblasZspr(handle, uplo, n, alpha, x, incx, AP) &
                result(c_int) &
                bind(c, name = 'hipblasZspr')
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
        end function hipblasZspr
    end interface

    ! sprBatched
    interface
        function hipblasSsprBatched(handle, uplo, n, alpha, x, incx, AP, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasSsprBatched')
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
        end function hipblasSsprBatched
    end interface

    interface
        function hipblasDsprBatched(handle, uplo, n, alpha, x, incx, AP, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasDsprBatched')
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
        end function hipblasDsprBatched
    end interface

    interface
        function hipblasCsprBatched(handle, uplo, n, alpha, x, incx, AP, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasCsprBatched')
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
        end function hipblasCsprBatched
    end interface

    interface
        function hipblasZsprBatched(handle, uplo, n, alpha, x, incx, AP, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasZsprBatched')
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
        end function hipblasZsprBatched
    end interface

    ! sprStridedBatched
    interface
        function hipblasSsprStridedBatched(handle, uplo, n, alpha, x, incx, stride_x, &
                AP, stride_AP, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasSsprStridedBatched')
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
        end function hipblasSsprStridedBatched
    end interface

    interface
        function hipblasDsprStridedBatched(handle, uplo, n, alpha, x, incx, stride_x, &
                AP, stride_AP, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasDsprStridedBatched')
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
        end function hipblasDsprStridedBatched
    end interface

    interface
        function hipblasCsprStridedBatched(handle, uplo, n, alpha, x, incx, stride_x, &
                AP, stride_AP, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasCsprStridedBatched')
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
        end function hipblasCsprStridedBatched
    end interface

    interface
        function hipblasZsprStridedBatched(handle, uplo, n, alpha, x, incx, stride_x, &
                AP, stride_AP, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasZsprStridedBatched')
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
        end function hipblasZsprStridedBatched
    end interface

    ! spr2
    interface
        function hipblasSspr2(handle, uplo, n, alpha, x, incx, &
                y, incy, AP) &
                result(c_int) &
                bind(c, name = 'hipblasSspr2')
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
        end function hipblasSspr2
    end interface

    interface
        function hipblasDspr2(handle, uplo, n, alpha, x, incx, &
                y, incy, AP) &
                result(c_int) &
                bind(c, name = 'hipblasDspr2')
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
        end function hipblasDspr2
    end interface

    ! spr2Batched
    interface
        function hipblasSspr2Batched(handle, uplo, n, alpha, x, incx, &
                y, incy, AP, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasSspr2Batched')
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
        end function hipblasSspr2Batched
    end interface

    interface
        function hipblasDspr2Batched(handle, uplo, n, alpha, x, incx, &
                y, incy, AP, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasDspr2Batched')
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
        end function hipblasDspr2Batched
    end interface

    ! spr2StridedBatched
    interface
        function hipblasSspr2StridedBatched(handle, uplo, n, alpha, x, incx, stride_x, &
                y, incy, stride_y, AP, stride_AP, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasSspr2StridedBatched')
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
        end function hipblasSspr2StridedBatched
    end interface

    interface
        function hipblasDspr2StridedBatched(handle, uplo, n, alpha, x, incx, stride_x, &
                y, incy, stride_y, AP, stride_AP, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasDspr2StridedBatched')
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
        end function hipblasDspr2StridedBatched
    end interface

    ! syr
    interface
        function hipblasSsyr(handle, uplo, n, alpha, x, incx, A, lda) &
                result(c_int) &
                bind(c, name = 'hipblasSsyr')
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
        end function hipblasSsyr
    end interface

    interface
        function hipblasDsyr(handle, uplo, n, alpha, x, incx, A, lda) &
                result(c_int) &
                bind(c, name = 'hipblasDsyr')
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
        end function hipblasDsyr
    end interface

    interface
        function hipblasCsyr(handle, uplo, n, alpha, x, incx, A, lda) &
                result(c_int) &
                bind(c, name = 'hipblasCsyr')
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
        end function hipblasCsyr
    end interface

    interface
        function hipblasZsyr(handle, uplo, n, alpha, x, incx, A, lda) &
                result(c_int) &
                bind(c, name = 'hipblasZsyr')
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
        end function hipblasZsyr
    end interface

    ! syrBatched
    interface
        function hipblasSsyrBatched(handle, uplo, n, alpha, x, incx, A, lda, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasSsyrBatched')
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
        end function hipblasSsyrBatched
    end interface

    interface
        function hipblasDsyrBatched(handle, uplo, n, alpha, x, incx, A, lda, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasDsyrBatched')
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
        end function hipblasDsyrBatched
    end interface

    interface
        function hipblasCsyrBatched(handle, uplo, n, alpha, x, incx, A, lda, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasCsyrBatched')
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
        end function hipblasCsyrBatched
    end interface

    interface
        function hipblasZsyrBatched(handle, uplo, n, alpha, x, incx, A, lda, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasZsyrBatched')
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
        end function hipblasZsyrBatched
    end interface

    ! syrStridedBatched
    interface
        function hipblasSsyrStridedBatched(handle, uplo, n, alpha, x, incx, stride_x, &
                A, lda, stride_A, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasSsyrStridedBatched')
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
        end function hipblasSsyrStridedBatched
    end interface

    interface
        function hipblasDsyrStridedBatched(handle, uplo, n, alpha, x, incx, stride_x, &
                A, lda, stride_A, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasDsyrStridedBatched')
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
        end function hipblasDsyrStridedBatched
    end interface

    interface
        function hipblasCsyrStridedBatched(handle, uplo, n, alpha, x, incx, stride_x, &
                A, lda, stride_A, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasCsyrStridedBatched')
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
        end function hipblasCsyrStridedBatched
    end interface

    interface
        function hipblasZsyrStridedBatched(handle, uplo, n, alpha, x, incx, stride_x, &
                A, lda, stride_A, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasZsyrStridedBatched')
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
        end function hipblasZsyrStridedBatched
    end interface

    ! syr2
    interface
        function hipblasSsyr2(handle, uplo, n, alpha, x, incx, &
                y, incy, A, lda) &
                result(c_int) &
                bind(c, name = 'hipblasSsyr2')
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
        end function hipblasSsyr2
    end interface

    interface
        function hipblasDsyr2(handle, uplo, n, alpha, x, incx, &
                y, incy, A, lda) &
                result(c_int) &
                bind(c, name = 'hipblasDsyr2')
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
        end function hipblasDsyr2
    end interface

    interface
        function hipblasCsyr2(handle, uplo, n, alpha, x, incx, &
                y, incy, A, lda) &
                result(c_int) &
                bind(c, name = 'hipblasCsyr2')
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
        end function hipblasCsyr2
    end interface

    interface
        function hipblasZsyr2(handle, uplo, n, alpha, x, incx, &
                y, incy, A, lda) &
                result(c_int) &
                bind(c, name = 'hipblasZsyr2')
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
        end function hipblasZsyr2
    end interface

    ! syr2Batched
    interface
        function hipblasSsyr2Batched(handle, uplo, n, alpha, x, incx, &
                y, incy, A, lda, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasSsyr2Batched')
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
        end function hipblasSsyr2Batched
    end interface

    interface
        function hipblasDsyr2Batched(handle, uplo, n, alpha, x, incx, &
                y, incy, A, lda, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasDsyr2Batched')
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
        end function hipblasDsyr2Batched
    end interface

    interface
        function hipblasCsyr2Batched(handle, uplo, n, alpha, x, incx, &
                y, incy, A, lda, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasCsyr2Batched')
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
        end function hipblasCsyr2Batched
    end interface

    interface
        function hipblasZsyr2Batched(handle, uplo, n, alpha, x, incx, &
                y, incy, A, lda, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasZsyr2Batched')
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
        end function hipblasZsyr2Batched
    end interface

    ! syr2StridedBatched
    interface
        function hipblasSsyr2StridedBatched(handle, uplo, n, alpha, x, incx, stride_x, &
                y, incy, stride_y, A, lda, stride_A, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasSsyr2StridedBatched')
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
        end function hipblasSsyr2StridedBatched
    end interface

    interface
        function hipblasDsyr2StridedBatched(handle, uplo, n, alpha, x, incx, stride_x, &
                y, incy, stride_y, A, lda, stride_A, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasDsyr2StridedBatched')
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
        end function hipblasDsyr2StridedBatched
    end interface

    interface
        function hipblasCsyr2StridedBatched(handle, uplo, n, alpha, x, incx, stride_x, &
                y, incy, stride_y, A, lda, stride_A, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasCsyr2StridedBatched')
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
        end function hipblasCsyr2StridedBatched
    end interface

    interface
        function hipblasZsyr2StridedBatched(handle, uplo, n, alpha, x, incx, stride_x, &
                y, incy, stride_y, A, lda, stride_A, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasZsyr2StridedBatched')
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
        end function hipblasZsyr2StridedBatched
    end interface

    !--------!
    ! blas 3 !
    !--------!

    ! hemm
    interface
        function hipblasChemm(handle, side, uplo, n, k, alpha, &
                A, lda, B, ldb, beta, C, ldc) &
                result(c_int) &
                bind(c, name = 'hipblasChemm')
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
        end function hipblasChemm
    end interface

    interface
        function hipblasZhemm(handle, side, uplo, n, k, alpha, &
                A, lda, B, ldb, beta, C, ldc) &
                result(c_int) &
                bind(c, name = 'hipblasZhemm')
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
        end function hipblasZhemm
    end interface

    ! hemmBatched
    interface
        function hipblasChemmBatched(handle, side, uplo, n, k, alpha, &
                A, lda, B, ldb, beta, C, ldc, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasChemmBatched')
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
        end function hipblasChemmBatched
    end interface

    interface
        function hipblasZhemmBatched(handle, side, uplo, n, k, alpha, &
                A, lda, B, ldb, beta, C, ldc, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasZhemmBatched')
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
        end function hipblasZhemmBatched
    end interface

    ! hemmStridedBatched
    interface
        function hipblasChemmStridedBatched(handle, side, uplo, n, k, alpha, &
                A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasChemmStridedBatched')
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
        end function hipblasChemmStridedBatched
    end interface

    interface
        function hipblasZhemmStridedBatched(handle, side, uplo, n, k, alpha, &
                A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasZhemmStridedBatched')
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
        end function hipblasZhemmStridedBatched
    end interface

    ! herk
    interface
        function hipblasCherk(handle, uplo, transA, n, k, alpha, &
                A, lda, beta, C, ldc) &
                result(c_int) &
                bind(c, name = 'hipblasCherk')
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
        end function hipblasCherk
    end interface

    interface
        function hipblasZherk(handle, uplo, transA, n, k, alpha, &
                A, lda, beta, C, ldc) &
                result(c_int) &
                bind(c, name = 'hipblasZherk')
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
        end function hipblasZherk
    end interface

    ! herkBatched
    interface
        function hipblasCherkBatched(handle, uplo, transA, n, k, alpha, &
                A, lda, beta, C, ldc, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasCherkBatched')
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
        end function hipblasCherkBatched
    end interface

    interface
        function hipblasZherkBatched(handle, uplo, transA, n, k, alpha, &
                A, lda, beta, C, ldc, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasZherkBatched')
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
        end function hipblasZherkBatched
    end interface

    ! herkStridedBatched
    interface
        function hipblasCherkStridedBatched(handle, uplo, transA, n, k, alpha, &
                A, lda, stride_A, beta, C, ldc, stride_C, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasCherkStridedBatched')
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
        end function hipblasCherkStridedBatched
    end interface

    interface
        function hipblasZherkStridedBatched(handle, uplo, transA, n, k, alpha, &
                A, lda, stride_A, beta, C, ldc, stride_C, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasZherkStridedBatched')
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
        end function hipblasZherkStridedBatched
    end interface

    ! her2k
    interface
        function hipblasCher2k(handle, uplo, transA, n, k, alpha, &
                A, lda, B, ldb, beta, C, ldc) &
                result(c_int) &
                bind(c, name = 'hipblasCher2k')
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
        end function hipblasCher2k
    end interface

    interface
        function hipblasZher2k(handle, uplo, transA, n, k, alpha, &
                A, lda, B, ldb, beta, C, ldc) &
                result(c_int) &
                bind(c, name = 'hipblasZher2k')
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
        end function hipblasZher2k
    end interface

    ! her2kBatched
    interface
        function hipblasCher2kBatched(handle, uplo, transA, n, k, alpha, &
                A, lda, B, ldb, beta, C, ldc, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasCher2kBatched')
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
        end function hipblasCher2kBatched
    end interface

    interface
        function hipblasZher2kBatched(handle, uplo, transA, n, k, alpha, &
                A, lda, B, ldb, beta, C, ldc, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasZher2kBatched')
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
        end function hipblasZher2kBatched
    end interface

    ! her2kStridedBatched
    interface
        function hipblasCher2kStridedBatched(handle, uplo, transA, n, k, alpha, &
                A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasCher2kStridedBatched')
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
        end function hipblasCher2kStridedBatched
    end interface

    interface
        function hipblasZher2kStridedBatched(handle, uplo, transA, n, k, alpha, &
                A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasZher2kStridedBatched')
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
        end function hipblasZher2kStridedBatched
    end interface

    ! herkx
    interface
        function hipblasCherkx(handle, uplo, transA, n, k, alpha, &
                A, lda, B, ldb, beta, C, ldc) &
                result(c_int) &
                bind(c, name = 'hipblasCherkx')
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
        end function hipblasCherkx
    end interface

    interface
        function hipblasZherkx(handle, uplo, transA, n, k, alpha, &
                A, lda, B, ldb, beta, C, ldc) &
                result(c_int) &
                bind(c, name = 'hipblasZherkx')
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
        end function hipblasZherkx
    end interface

    ! herkxBatched
    interface
        function hipblasCherkxBatched(handle, uplo, transA, n, k, alpha, &
                A, lda, B, ldb, beta, C, ldc, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasCherkxBatched')
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
        end function hipblasCherkxBatched
    end interface

    interface
        function hipblasZherkxBatched(handle, uplo, transA, n, k, alpha, &
                A, lda, B, ldb, beta, C, ldc, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasZherkxBatched')
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
        end function hipblasZherkxBatched
    end interface

    ! herkxStridedBatched
    interface
        function hipblasCherkxStridedBatched(handle, uplo, transA, n, k, alpha, &
                A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasCherkxStridedBatched')
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
        end function hipblasCherkxStridedBatched
    end interface

    interface
        function hipblasZherkxStridedBatched(handle, uplo, transA, n, k, alpha, &
                A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasZherkxStridedBatched')
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
        end function hipblasZherkxStridedBatched
    end interface

    ! symm
    interface
        function hipblasSsymm(handle, side, uplo, m, n, alpha, &
                A, lda, B, ldb, beta, C, ldc) &
                result(c_int) &
                bind(c, name = 'hipblasSsymm')
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
        end function hipblasSsymm
    end interface

    interface
        function hipblasDsymm(handle, side, uplo, m, n, alpha, &
                A, lda, B, ldb, beta, C, ldc) &
                result(c_int) &
                bind(c, name = 'hipblasDsymm')
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
        end function hipblasDsymm
    end interface

    interface
        function hipblasCsymm(handle, side, uplo, m, n, alpha, &
                A, lda, B, ldb, beta, C, ldc) &
                result(c_int) &
                bind(c, name = 'hipblasCsymm')
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
        end function hipblasCsymm
    end interface

    interface
        function hipblasZsymm(handle, side, uplo, m, n, alpha, &
                A, lda, B, ldb, beta, C, ldc) &
                result(c_int) &
                bind(c, name = 'hipblasZsymm')
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
        end function hipblasZsymm
    end interface

    ! symmBatched
    interface
        function hipblasSsymmBatched(handle, side, uplo, m, n, alpha, &
                A, lda, B, ldb, beta, C, ldc, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasSsymmBatched')
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
        end function hipblasSsymmBatched
    end interface

    interface
        function hipblasDsymmBatched(handle, side, uplo, m, n, alpha, &
                A, lda, B, ldb, beta, C, ldc, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasDsymmBatched')
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
        end function hipblasDsymmBatched
    end interface

    interface
        function hipblasCsymmBatched(handle, side, uplo, m, n, alpha, &
                A, lda, B, ldb, beta, C, ldc, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasCsymmBatched')
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
        end function hipblasCsymmBatched
    end interface

    interface
        function hipblasZsymmBatched(handle, side, uplo, m, n, alpha, &
                A, lda, B, ldb, beta, C, ldc, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasZsymmBatched')
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
        end function hipblasZsymmBatched
    end interface

    ! symmStridedBatched
    interface
        function hipblasSsymmStridedBatched(handle, side, uplo, m, n, alpha, &
                A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasSsymmStridedBatched')
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
        end function hipblasSsymmStridedBatched
    end interface

    interface
        function hipblasDsymmStridedBatched(handle, side, uplo, m, n, alpha, &
                A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasDsymmStridedBatched')
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
        end function hipblasDsymmStridedBatched
    end interface

    interface
        function hipblasCsymmStridedBatched(handle, side, uplo, m, n, alpha, &
                A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasCsymmStridedBatched')
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
        end function hipblasCsymmStridedBatched
    end interface

    interface
        function hipblasZsymmStridedBatched(handle, side, uplo, m, n, alpha, &
                A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasZsymmStridedBatched')
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
        end function hipblasZsymmStridedBatched
    end interface

    ! syrk
    interface
        function hipblasSsyrk(handle, uplo, transA, n, k, alpha, &
                A, lda, beta, C, ldc) &
                result(c_int) &
                bind(c, name = 'hipblasSsyrk')
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
        end function hipblasSsyrk
    end interface

    interface
        function hipblasDsyrk(handle, uplo, transA, n, k, alpha, &
                A, lda, beta, C, ldc) &
                result(c_int) &
                bind(c, name = 'hipblasDsyrk')
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
        end function hipblasDsyrk
    end interface

    interface
        function hipblasCsyrk(handle, uplo, transA, n, k, alpha, &
                A, lda, beta, C, ldc) &
                result(c_int) &
                bind(c, name = 'hipblasCsyrk')
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
        end function hipblasCsyrk
    end interface

    interface
        function hipblasZsyrk(handle, uplo, transA, n, k, alpha, &
                A, lda, beta, C, ldc) &
                result(c_int) &
                bind(c, name = 'hipblasZsyrk')
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
        end function hipblasZsyrk
    end interface

    ! syrkBatched
    interface
        function hipblasSsyrkBatched(handle, uplo, transA, n, k, alpha, &
                A, lda, beta, C, ldc, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasSsyrkBatched')
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
        end function hipblasSsyrkBatched
    end interface

    interface
        function hipblasDsyrkBatched(handle, uplo, transA, n, k, alpha, &
                A, lda, beta, C, ldc, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasDsyrkBatched')
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
        end function hipblasDsyrkBatched
    end interface

    interface
        function hipblasCsyrkBatched(handle, uplo, transA, n, k, alpha, &
                A, lda, beta, C, ldc, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasCsyrkBatched')
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
        end function hipblasCsyrkBatched
    end interface

    interface
        function hipblasZsyrkBatched(handle, uplo, transA, n, k, alpha, &
                A, lda, beta, C, ldc, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasZsyrkBatched')
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
        end function hipblasZsyrkBatched
    end interface

    ! syrkStridedBatched
    interface
        function hipblasSsyrkStridedBatched(handle, uplo, transA, n, k, alpha, &
                A, lda, stride_A, beta, C, ldc, stride_C, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasSsyrkStridedBatched')
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
        end function hipblasSsyrkStridedBatched
    end interface

    interface
        function hipblasDsyrkStridedBatched(handle, uplo, transA, n, k, alpha, &
                A, lda, stride_A, beta, C, ldc, stride_C, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasDsyrkStridedBatched')
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
        end function hipblasDsyrkStridedBatched
    end interface

    interface
        function hipblasCsyrkStridedBatched(handle, uplo, transA, n, k, alpha, &
                A, lda, stride_A, beta, C, ldc, stride_C, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasCsyrkStridedBatched')
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
        end function hipblasCsyrkStridedBatched
    end interface

    interface
        function hipblasZsyrkStridedBatched(handle, uplo, transA, n, k, alpha, &
                A, lda, stride_A, beta, C, ldc, stride_C, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasZsyrkStridedBatched')
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
        end function hipblasZsyrkStridedBatched
    end interface

    ! syr2k
    interface
        function hipblasSsyr2k(handle, uplo, transA, n, k, alpha, &
                A, lda, B, ldb, beta, C, ldc) &
                result(c_int) &
                bind(c, name = 'hipblasSsyr2k')
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
        end function hipblasSsyr2k
    end interface

    interface
        function hipblasDsyr2k(handle, uplo, transA, n, k, alpha, &
                A, lda, B, ldb, beta, C, ldc) &
                result(c_int) &
                bind(c, name = 'hipblasDsyr2k')
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
        end function hipblasDsyr2k
    end interface

    interface
        function hipblasCsyr2k(handle, uplo, transA, n, k, alpha, &
                A, lda, B, ldb, beta, C, ldc) &
                result(c_int) &
                bind(c, name = 'hipblasCsyr2k')
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
        end function hipblasCsyr2k
    end interface

    interface
        function hipblasZsyr2k(handle, uplo, transA, n, k, alpha, &
                A, lda, B, ldb, beta, C, ldc) &
                result(c_int) &
                bind(c, name = 'hipblasZsyr2k')
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
        end function hipblasZsyr2k
    end interface

    ! syr2kBatched
    interface
        function hipblasSsyr2kBatched(handle, uplo, transA, n, k, alpha, &
                A, lda, B, ldb, beta, C, ldc, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasSsyr2kBatched')
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
        end function hipblasSsyr2kBatched
    end interface

    interface
        function hipblasDsyr2kBatched(handle, uplo, transA, n, k, alpha, &
                A, lda, B, ldb, beta, C, ldc, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasDsyr2kBatched')
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
        end function hipblasDsyr2kBatched
    end interface

    interface
        function hipblasCsyr2kBatched(handle, uplo, transA, n, k, alpha, &
                A, lda, B, ldb, beta, C, ldc, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasCsyr2kBatched')
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
        end function hipblasCsyr2kBatched
    end interface

    interface
        function hipblasZsyr2kBatched(handle, uplo, transA, n, k, alpha, &
                A, lda, B, ldb, beta, C, ldc, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasZsyr2kBatched')
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
        end function hipblasZsyr2kBatched
    end interface

    ! syr2kStridedBatched
    interface
        function hipblasSsyr2kStridedBatched(handle, uplo, transA, n, k, alpha, &
                A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasSsyr2kStridedBatched')
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
        end function hipblasSsyr2kStridedBatched
    end interface

    interface
        function hipblasDsyr2kStridedBatched(handle, uplo, transA, n, k, alpha, &
                A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasDsyr2kStridedBatched')
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
        end function hipblasDsyr2kStridedBatched
    end interface

    interface
        function hipblasCsyr2kStridedBatched(handle, uplo, transA, n, k, alpha, &
                A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasCsyr2kStridedBatched')
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
        end function hipblasCsyr2kStridedBatched
    end interface

    interface
        function hipblasZsyr2kStridedBatched(handle, uplo, transA, n, k, alpha, &
                A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasZsyr2kStridedBatched')
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
        end function hipblasZsyr2kStridedBatched
    end interface

    ! syrkx
    interface
        function hipblasSsyrkx(handle, uplo, transA, n, k, alpha, &
                A, lda, B, ldb, beta, C, ldc) &
                result(c_int) &
                bind(c, name = 'hipblasSsyrkx')
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
        end function hipblasSsyrkx
    end interface

    interface
        function hipblasDsyrkx(handle, uplo, transA, n, k, alpha, &
                A, lda, B, ldb, beta, C, ldc) &
                result(c_int) &
                bind(c, name = 'hipblasDsyrkx')
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
        end function hipblasDsyrkx
    end interface

    interface
        function hipblasCsyrkx(handle, uplo, transA, n, k, alpha, &
                A, lda, B, ldb, beta, C, ldc) &
                result(c_int) &
                bind(c, name = 'hipblasCsyrkx')
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
        end function hipblasCsyrkx
    end interface

    interface
        function hipblasZsyrkx(handle, uplo, transA, n, k, alpha, &
                A, lda, B, ldb, beta, C, ldc) &
                result(c_int) &
                bind(c, name = 'hipblasZsyrkx')
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
        end function hipblasZsyrkx
    end interface

    ! syrkxBatched
    interface
        function hipblasSsyrkxBatched(handle, uplo, transA, n, k, alpha, &
                A, lda, B, ldb, beta, C, ldc, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasSsyrkxBatched')
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
        end function hipblasSsyrkxBatched
    end interface

    interface
        function hipblasDsyrkxBatched(handle, uplo, transA, n, k, alpha, &
                A, lda, B, ldb, beta, C, ldc, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasDsyrkxBatched')
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
        end function hipblasDsyrkxBatched
    end interface

    interface
        function hipblasCsyrkxBatched(handle, uplo, transA, n, k, alpha, &
                A, lda, B, ldb, beta, C, ldc, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasCsyrkxBatched')
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
        end function hipblasCsyrkxBatched
    end interface

    interface
        function hipblasZsyrkxBatched(handle, uplo, transA, n, k, alpha, &
                A, lda, B, ldb, beta, C, ldc, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasZsyrkxBatched')
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
        end function hipblasZsyrkxBatched
    end interface

    ! syrkxStridedBatched
    interface
        function hipblasSsyrkxStridedBatched(handle, uplo, transA, n, k, alpha, &
                A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasSsyrkxStridedBatched')
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
        end function hipblasSsyrkxStridedBatched
    end interface

    interface
        function hipblasDsyrkxStridedBatched(handle, uplo, transA, n, k, alpha, &
                A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasDsyrkxStridedBatched')
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
        end function hipblasDsyrkxStridedBatched
    end interface

    interface
        function hipblasCsyrkxStridedBatched(handle, uplo, transA, n, k, alpha, &
                A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasCsyrkxStridedBatched')
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
        end function hipblasCsyrkxStridedBatched
    end interface

    interface
        function hipblasZsyrkxStridedBatched(handle, uplo, transA, n, k, alpha, &
                A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasZsyrkxStridedBatched')
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
        end function hipblasZsyrkxStridedBatched
    end interface

    ! trmm
    interface
        function hipblasStrmm(handle, side, uplo, transA, diag, m, n, alpha, &
                A, lda, B, ldb) &
                result(c_int) &
                bind(c, name = 'hipblasStrmm')
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
        end function hipblasStrmm
    end interface

    interface
        function hipblasDtrmm(handle, side, uplo, transA, diag, m, n, alpha, &
                A, lda, B, ldb) &
                result(c_int) &
                bind(c, name = 'hipblasDtrmm')
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
        end function hipblasDtrmm
    end interface

    interface
        function hipblasCtrmm(handle, side, uplo, transA, diag, m, n, alpha, &
                A, lda, B, ldb) &
                result(c_int) &
                bind(c, name = 'hipblasCtrmm')
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
        end function hipblasCtrmm
    end interface

    interface
        function hipblasZtrmm(handle, side, uplo, transA, diag, m, n, alpha, &
                A, lda, B, ldb) &
                result(c_int) &
                bind(c, name = 'hipblasZtrmm')
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
        end function hipblasZtrmm
    end interface

    ! trmmBatched
    interface
        function hipblasStrmmBatched(handle, side, uplo, transA, diag, m, n, alpha, &
                A, lda, B, ldb, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasStrmmBatched')
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
        end function hipblasStrmmBatched
    end interface

    interface
        function hipblasDtrmmBatched(handle, side, uplo, transA, diag, m, n, alpha, &
                A, lda, B, ldb, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasDtrmmBatched')
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
        end function hipblasDtrmmBatched
    end interface

    interface
        function hipblasCtrmmBatched(handle, side, uplo, transA, diag, m, n, alpha, &
                A, lda, B, ldb, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasCtrmmBatched')
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
        end function hipblasCtrmmBatched
    end interface

    interface
        function hipblasZtrmmBatched(handle, side, uplo, transA, diag, m, n, alpha, &
                A, lda, B, ldb, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasZtrmmBatched')
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
        end function hipblasZtrmmBatched
    end interface

    ! trmmStridedBatched
    interface
        function hipblasStrmmStridedBatched(handle, side, uplo, transA, diag, m, n, alpha, &
                A, lda, stride_A, B, ldb, stride_B, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasStrmmStridedBatched')
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
        end function hipblasStrmmStridedBatched
    end interface

    interface
        function hipblasDtrmmStridedBatched(handle, side, uplo, transA, diag, m, n, alpha, &
                A, lda, stride_A, B, ldb, stride_B, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasDtrmmStridedBatched')
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
        end function hipblasDtrmmStridedBatched
    end interface

    interface
        function hipblasCtrmmStridedBatched(handle, side, uplo, transA, diag, m, n, alpha, &
                A, lda, stride_A, B, ldb, stride_B, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasCtrmmStridedBatched')
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
        end function hipblasCtrmmStridedBatched
    end interface

    interface
        function hipblasZtrmmStridedBatched(handle, side, uplo, transA, diag, m, n, alpha, &
                A, lda, stride_A, B, ldb, stride_B, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasZtrmmStridedBatched')
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
        end function hipblasZtrmmStridedBatched
    end interface

    ! trtri
    interface
        function hipblasStrtri(handle, uplo, diag, n, &
                A, lda, invA, ldinvA) &
                result(c_int) &
                bind(c, name = 'hipblasStrtri')
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
        end function hipblasStrtri
    end interface

    interface
        function hipblasDtrtri(handle, uplo, diag, n, &
                A, lda, invA, ldinvA) &
                result(c_int) &
                bind(c, name = 'hipblasDtrtri')
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
        end function hipblasDtrtri
    end interface

    interface
        function hipblasCtrtri(handle, uplo, diag, n, &
                A, lda, invA, ldinvA) &
                result(c_int) &
                bind(c, name = 'hipblasCtrtri')
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
        end function hipblasCtrtri
    end interface

    interface
        function hipblasZtrtri(handle, uplo, diag, n, &
                A, lda, invA, ldinvA) &
                result(c_int) &
                bind(c, name = 'hipblasZtrtri')
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
        end function hipblasZtrtri
    end interface

    ! trtriBatched
    interface
        function hipblasStrtriBatched(handle, uplo, diag, n, &
                A, lda, invA, ldinvA, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasStrtriBatched')
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
        end function hipblasStrtriBatched
    end interface

    interface
        function hipblasDtrtriBatched(handle, uplo, diag, n, &
                A, lda, invA, ldinvA, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasDtrtriBatched')
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
        end function hipblasDtrtriBatched
    end interface

    interface
        function hipblasCtrtriBatched(handle, uplo, diag, n, &
                A, lda, invA, ldinvA, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasCtrtriBatched')
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
        end function hipblasCtrtriBatched
    end interface

    interface
        function hipblasZtrtriBatched(handle, uplo, diag, n, &
                A, lda, invA, ldinvA, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasZtrtriBatched')
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
        end function hipblasZtrtriBatched
    end interface

    ! trtriStridedBatched
    interface
        function hipblasStrtriStridedBatched(handle, uplo, diag, n, &
                A, lda, stride_A, invA, ldinvA, stride_invA, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasStrtriStridedBatched')
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
        end function hipblasStrtriStridedBatched
    end interface

    interface
        function hipblasDtrtriStridedBatched(handle, uplo, diag, n, &
                A, lda, stride_A, invA, ldinvA, stride_invA, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasDtrtriStridedBatched')
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
        end function hipblasDtrtriStridedBatched
    end interface

    interface
        function hipblasCtrtriStridedBatched(handle, uplo, diag, n, &
                A, lda, stride_A, invA, ldinvA, stride_invA, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasCtrtriStridedBatched')
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
        end function hipblasCtrtriStridedBatched
    end interface

    interface
        function hipblasZtrtriStridedBatched(handle, uplo, diag, n, &
                A, lda, stride_A, invA, ldinvA, stride_invA, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasZtrtriStridedBatched')
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
        end function hipblasZtrtriStridedBatched
    end interface

    ! trsm
    interface
        function hipblasStrsm(handle, side, uplo, transA, diag, m, n, alpha, &
                A, lda, B, ldb) &
                result(c_int) &
                bind(c, name = 'hipblasStrsm')
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
        end function hipblasStrsm
    end interface

    interface
        function hipblasDtrsm(handle, side, uplo, transA, diag, m, n, alpha, &
                A, lda, B, ldb) &
                result(c_int) &
                bind(c, name = 'hipblasDtrsm')
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
        end function hipblasDtrsm
    end interface

    interface
        function hipblasCtrsm(handle, side, uplo, transA, diag, m, n, alpha, &
                A, lda, B, ldb) &
                result(c_int) &
                bind(c, name = 'hipblasCtrsm')
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
        end function hipblasCtrsm
    end interface

    interface
        function hipblasZtrsm(handle, side, uplo, transA, diag, m, n, alpha, &
                A, lda, B, ldb) &
                result(c_int) &
                bind(c, name = 'hipblasZtrsm')
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
        end function hipblasZtrsm
    end interface

    ! trsmBatched
    interface
        function hipblasStrsmBatched(handle, side, uplo, transA, diag, m, n, alpha, &
                A, lda, B, ldb, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasStrsmBatched')
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
        end function hipblasStrsmBatched
    end interface

    interface
        function hipblasDtrsmBatched(handle, side, uplo, transA, diag, m, n, alpha, &
                A, lda, B, ldb, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasDtrsmBatched')
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
        end function hipblasDtrsmBatched
    end interface

    interface
        function hipblasCtrsmBatched(handle, side, uplo, transA, diag, m, n, alpha, &
                A, lda, B, ldb, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasCtrsmBatched')
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
        end function hipblasCtrsmBatched
    end interface

    interface
        function hipblasZtrsmBatched(handle, side, uplo, transA, diag, m, n, alpha, &
                A, lda, B, ldb, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasZtrsmBatched')
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
        end function hipblasZtrsmBatched
    end interface

    ! trsmStridedBatched
    interface
        function hipblasStrsmStridedBatched(handle, side, uplo, transA, diag, m, n, alpha, &
                A, lda, stride_A, B, ldb, stride_B, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasStrsmStridedBatched')
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
        end function hipblasStrsmStridedBatched
    end interface

    interface
        function hipblasDtrsmStridedBatched(handle, side, uplo, transA, diag, m, n, alpha, &
                A, lda, stride_A, B, ldb, stride_B, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasDtrsmStridedBatched')
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
        end function hipblasDtrsmStridedBatched
    end interface

    interface
        function hipblasCtrsmStridedBatched(handle, side, uplo, transA, diag, m, n, alpha, &
                A, lda, stride_A, B, ldb, stride_B, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasCtrsmStridedBatched')
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
        end function hipblasCtrsmStridedBatched
    end interface

    interface
        function hipblasZtrsmStridedBatched(handle, side, uplo, transA, diag, m, n, alpha, &
                A, lda, stride_A, B, ldb, stride_B, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasZtrsmStridedBatched')
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
        end function hipblasZtrsmStridedBatched
    end interface

    ! gemm
    interface
        function hipblasHgemm(handle, transA, transB, m, n, k, alpha, &
                A, lda, B, ldb, beta, C, ldc) &
                result(c_int) &
                bind(c, name = 'hipblasHgemm')
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
        end function hipblasHgemm
    end interface

    interface
        function hipblasSgemm(handle, transA, transB, m, n, k, alpha, &
                A, lda, B, ldb, beta, C, ldc) &
                result(c_int) &
                bind(c, name = 'hipblasSgemm')
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
        end function hipblasSgemm
    end interface

    interface
        function hipblasDgemm(handle, transA, transB, m, n, k, alpha, &
                A, lda, B, ldb, beta, C, ldc) &
                result(c_int) &
                bind(c, name = 'hipblasDgemm')
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
        end function hipblasDgemm
    end interface

    interface
        function hipblasCgemm(handle, transA, transB, m, n, k, alpha, &
                A, lda, B, ldb, beta, C, ldc) &
                result(c_int) &
                bind(c, name = 'hipblasCgemm')
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
        end function hipblasCgemm
    end interface

    interface
        function hipblasZgemm(handle, transA, transB, m, n, k, alpha, &
                A, lda, B, ldb, beta, C, ldc) &
                result(c_int) &
                bind(c, name = 'hipblasZgemm')
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
        end function hipblasZgemm
    end interface

    ! gemmBatched
    interface
        function hipblasHgemmBatched(handle, transA, transB, m, n, k, alpha, &
                A, lda, B, ldb, beta, C, ldc, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasHgemmBatched')
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
        end function hipblasHgemmBatched
    end interface

    interface
        function hipblasSgemmBatched(handle, transA, transB, m, n, k, alpha, &
                A, lda, B, ldb, beta, C, ldc, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasSgemmBatched')
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
        end function hipblasSgemmBatched
    end interface

    interface
        function hipblasDgemmBatched(handle, transA, transB, m, n, k, alpha, &
                A, lda, B, ldb, beta, C, ldc, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasDgemmBatched')
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
        end function hipblasDgemmBatched
    end interface

    interface
        function hipblasCgemmBatched(handle, transA, transB, m, n, k, alpha, &
                A, lda, B, ldb, beta, C, ldc, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasCgemmBatched')
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
        end function hipblasCgemmBatched
    end interface

    interface
        function hipblasZgemmBatched(handle, transA, transB, m, n, k, alpha, &
                A, lda, B, ldb, beta, C, ldc, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasZgemmBatched')
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
        end function hipblasZgemmBatched
    end interface

    ! gemmStridedBatched
    interface
        function hipblasHgemmStridedBatched(handle, transA, transB, m, n, k, alpha, &
                A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasHgemmStridedBatched')
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
        end function hipblasHgemmStridedBatched
    end interface

    interface
        function hipblasSgemmStridedBatched(handle, transA, transB, m, n, k, alpha, &
                A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasSgemmStridedBatched')
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
        end function hipblasSgemmStridedBatched
    end interface

    interface
        function hipblasDgemmStridedBatched(handle, transA, transB, m, n, k, alpha, &
                A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasDgemmStridedBatched')
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
        end function hipblasDgemmStridedBatched
    end interface

    interface
        function hipblasCgemmStridedBatched(handle, transA, transB, m, n, k, alpha, &
                A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasCgemmStridedBatched')
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
        end function hipblasCgemmStridedBatched
    end interface

    interface
        function hipblasZgemmStridedBatched(handle, transA, transB, m, n, k, alpha, &
                A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasZgemmStridedBatched')
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
        end function hipblasZgemmStridedBatched
    end interface

    ! dgmm
    interface
        function hipblasSdgmm(handle, side, m, n, &
                A, lda, x, incx, C, ldc) &
                result(c_int) &
                bind(c, name = 'hipblasSdgmm')
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
        end function hipblasSdgmm
    end interface

    interface
        function hipblasDdgmm(handle, side, m, n, &
                A, lda, x, incx, C, ldc) &
                result(c_int) &
                bind(c, name = 'hipblasDdgmm')
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
        end function hipblasDdgmm
    end interface

    interface
        function hipblasCdgmm(handle, side, m, n, &
                A, lda, x, incx, C, ldc) &
                result(c_int) &
                bind(c, name = 'hipblasCdgmm')
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
        end function hipblasCdgmm
    end interface

    interface
        function hipblasZdgmm(handle, side, m, n, &
                A, lda, x, incx, C, ldc) &
                result(c_int) &
                bind(c, name = 'hipblasZdgmm')
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
        end function hipblasZdgmm
    end interface

    ! dgmmBatched
    interface
        function hipblasSdgmmBatched(handle, side, m, n, &
                A, lda, x, incx, C, ldc, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasSdgmmBatched')
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
        end function hipblasSdgmmBatched
    end interface

    interface
        function hipblasDdgmmBatched(handle, side, m, n, &
                A, lda, x, incx, C, ldc, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasDdgmmBatched')
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
        end function hipblasDdgmmBatched
    end interface

    interface
        function hipblasCdgmmBatched(handle, side, m, n, &
                A, lda, x, incx, C, ldc, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasCdgmmBatched')
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
        end function hipblasCdgmmBatched
    end interface

    interface
        function hipblasZdgmmBatched(handle, side, m, n, &
                A, lda, x, incx, C, ldc, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasZdgmmBatched')
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
        end function hipblasZdgmmBatched
    end interface

    ! dgmmStridedBatched
    interface
        function hipblasSdgmmStridedBatched(handle, side, m, n, &
                A, lda, stride_A, x, incx, stride_x, C, ldc, stride_C, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasSdgmmStridedBatched')
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
        end function hipblasSdgmmStridedBatched
    end interface

    interface
        function hipblasDdgmmStridedBatched(handle, side, m, n, &
                A, lda, stride_A, x, incx, stride_x, C, ldc, stride_C, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasDdgmmStridedBatched')
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
        end function hipblasDdgmmStridedBatched
    end interface

    interface
        function hipblasCdgmmStridedBatched(handle, side, m, n, &
                A, lda, stride_A, x, incx, stride_x, C, ldc, stride_C, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasCdgmmStridedBatched')
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
        end function hipblasCdgmmStridedBatched
    end interface

    interface
        function hipblasZdgmmStridedBatched(handle, side, m, n, &
                A, lda, stride_A, x, incx, stride_x, C, ldc, stride_C, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasZdgmmStridedBatched')
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
        end function hipblasZdgmmStridedBatched
    end interface

    ! geam
    interface
        function hipblasSgeam(handle, transA, transB, m, n, alpha, &
                A, lda, beta, B, ldb, C, ldc) &
                result(c_int) &
                bind(c, name = 'hipblasSgeam')
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
        end function hipblasSgeam
    end interface

    interface
        function hipblasDgeam(handle, transA, transB, m, n, alpha, &
                A, lda, beta, B, ldb, C, ldc) &
                result(c_int) &
                bind(c, name = 'hipblasDgeam')
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
        end function hipblasDgeam
    end interface

    interface
        function hipblasCgeam(handle, transA, transB, m, n, alpha, &
                A, lda, beta, B, ldb, C, ldc) &
                result(c_int) &
                bind(c, name = 'hipblasCgeam')
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
        end function hipblasCgeam
    end interface

    interface
        function hipblasZgeam(handle, transA, transB, m, n, alpha, &
                A, lda, beta, B, ldb, C, ldc) &
                result(c_int) &
                bind(c, name = 'hipblasZgeam')
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
        end function hipblasZgeam
    end interface

    ! geamBatched
    interface
        function hipblasSgeamBatched(handle, transA, transB, m, n, alpha, &
                A, lda, beta, B, ldb, C, ldc, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasSgeamBatched')
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
        end function hipblasSgeamBatched
    end interface

    interface
        function hipblasDgeamBatched(handle, transA, transB, m, n, alpha, &
                A, lda, beta, B, ldb, C, ldc, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasDgeamBatched')
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
        end function hipblasDgeamBatched
    end interface

    interface
        function hipblasCgeamBatched(handle, transA, transB, m, n, alpha, &
                A, lda, beta, B, ldb, C, ldc, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasCgeamBatched')
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
        end function hipblasCgeamBatched
    end interface

    interface
        function hipblasZgeamBatched(handle, transA, transB, m, n, alpha, &
                A, lda, beta, B, ldb, C, ldc, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasZgeamBatched')
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
        end function hipblasZgeamBatched
    end interface

    ! geamStridedBatched
    interface
        function hipblasSgeamStridedBatched(handle, transA, transB, m, n, alpha, &
                A, lda, stride_A, beta, B, ldb, stride_B, C, ldc, stride_C, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasSgeamStridedBatched')
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
        end function hipblasSgeamStridedBatched
    end interface

    interface
        function hipblasDgeamStridedBatched(handle, transA, transB, m, n, alpha, &
                A, lda, stride_A, beta, B, ldb, stride_B, C, ldc, stride_C, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasDgeamStridedBatched')
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
        end function hipblasDgeamStridedBatched
    end interface

    interface
        function hipblasCgeamStridedBatched(handle, transA, transB, m, n, alpha, &
                A, lda, stride_A, beta, B, ldb, stride_B, C, ldc, stride_C, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasCgeamStridedBatched')
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
        end function hipblasCgeamStridedBatched
    end interface

    interface
        function hipblasZgeamStridedBatched(handle, transA, transB, m, n, alpha, &
                A, lda, stride_A, beta, B, ldb, stride_B, C, ldc, stride_C, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasZgeamStridedBatched')
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
        end function hipblasZgeamStridedBatched
    end interface

    !-----------------!
    ! blas Extensions !
    !-----------------!

    ! gemmEx
    interface
        function hipblasGemmEx(handle, transA, transB, m, n, k, alpha, a, a_type, lda, &
                b, b_type, ldb, beta, c, c_type, ldc,&
                compute_type, algo, solution_index, flags) &
                result(c_int) &
                bind(c, name = 'hipblasGemmEx')
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
        end function hipblasGemmEx
    end interface

    interface
        function hipblasGemmBatchedEx(handle, transA, transB, m, n, k, alpha, a, a_type, lda, &
                b, b_type, ldb, beta, c, c_type, ldc,&
                batch_count, compute_type, algo, solution_index, flags) &
                result(c_int) &
                bind(c, name = 'hipblasGemmBatchedEx')
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
        end function hipblasGemmBatchedEx
    end interface

    interface
        function hipblasGemmStridedBatchedEx(handle, transA, transB, m, n, k, alpha, a, a_type, lda, stride_a, &
                b, b_type, ldb, stride_b, beta, c, c_type, ldc, stride_c,&
                batch_count, compute_type, algo, solution_index, flags) &
                result(c_int) &
                bind(c, name = 'hipblasGemmStridedBatchedEx')
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
        end function hipblasGemmStridedBatchedEx
    end interface

    ! trsmEx
    interface
        function hipblasTrsmEx(handle, side, uplo, transA, diag, m, n, alpha, A, lda, &
                B, ldb, invA, invA_size, compute_type) &
                result(c_int) &
                bind(c, name = 'hipblasTrsmEx')
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
        end function hipblasTrsmEx
    end interface

    interface
        function hipblasTrsmBatchedEx(handle, side, uplo, transA, diag, m, n, alpha, A, lda, &
                B, ldb, batch_count, invA, invA_size, compute_type) &
                result(c_int) &
                bind(c, name = 'hipblasTrsmBatchedEx')
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
        end function hipblasTrsmBatchedEx
    end interface

    interface
        function hipblasTrsmStridedBatchedEx(handle, side, uplo, transA, diag, m, n, alpha, A, lda, stride_A, &
                B, ldb, stride_B, batch_count, invA, invA_size, stride_invA, compute_type) &
                result(c_int) &
                bind(c, name = 'hipblasTrsmStridedBatchedEx')
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
        end function hipblasTrsmStridedBatchedEx
    end interface

    ! ! syrkEx
    ! interface
    !     function hipblasCsyrkEx(handle, uplo, trans, n, k, alpha, A, Atype, lda, beta, C, Ctype, ldc) &
    !             result(c_int) &
    !             bind(c, name = 'hipblasCsyrkEx')
    !         use iso_c_binding
    !         use hipblas_enums
    !         implicit none
    !         type(c_ptr), value :: handle
    !         integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    !         integer(kind(HIPBLAS_OP_N)), value :: trans
    !         integer(c_int), value :: n
    !         integer(c_int), value :: k
    !         type(c_ptr), value :: alpha
    !         type(c_ptr), value :: A
    !         integer(kind(HIPBLAS_R_16F)), value :: Atype
    !         integer(c_int), value :: lda
    !         type(c_ptr), value :: beta
    !         type(c_ptr), value:: C
    !         integer(kind(HIPBLAS_R_16F)), value :: Ctype
    !         integer(c_int), value :: ldc
    !     end function hipblasCsyrkEx
    ! end interface

    ! ! herkEx
    ! interface
    !     function hipblasCherkEx(handle, uplo, trans, n, k, alpha, A, Atype, lda, beta, C, Ctype, ldc) &
    !             result(c_int) &
    !             bind(c, name = 'hipblasCherkEx')
    !         use iso_c_binding
    !         use hipblas_enums
    !         implicit none
    !         type(c_ptr), value :: handle
    !         integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    !         integer(kind(HIPBLAS_OP_N)), value :: trans
    !         integer(c_int), value :: n
    !         integer(c_int), value :: k
    !         type(c_ptr), value :: alpha
    !         type(c_ptr), value :: A
    !         integer(kind(HIPBLAS_R_16F)), value :: Atype
    !         integer(c_int), value :: lda
    !         type(c_ptr), value :: beta
    !         type(c_ptr), value:: C
    !         integer(kind(HIPBLAS_R_16F)), value :: Ctype
    !         integer(c_int), value :: ldc
    !     end function hipblasCherkEx
    ! end interface

    ! axpyEx
    interface
        function hipblasAxpyEx(handle, n, alpha, alphaType, x, xType, incx, y, yType, incy, executionType) &
                result(c_int) &
                bind(c, name = 'hipblasAxpyEx')
            use iso_c_binding
            use hipblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            integer(kind(HIPBLAS_R_16F)), value :: alphaType
            type(c_ptr), value :: x
            integer(kind(HIPBLAS_R_16F)), value :: xType
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(kind(HIPBLAS_R_16F)), value :: yType
            integer(c_int), value :: incy
            integer(kind(HIPBLAS_R_16F)), value :: executionType
        end function hipblasAxpyEx
    end interface

    interface
        function hipblasAxpyBatchedEx(handle, n, alpha, alphaType, x, xType, incx, &
                y, yType, incy, batch_count, executionType) &
                result(c_int) &
                bind(c, name = 'hipblasAxpyBatchedEx')
            use iso_c_binding
            use hipblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            integer(kind(HIPBLAS_R_16F)), value :: alphaType
            type(c_ptr), value :: x
            integer(kind(HIPBLAS_R_16F)), value :: xType
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(kind(HIPBLAS_R_16F)), value :: yType
            integer(c_int), value :: incy
            integer(c_int), value :: batch_count
            integer(kind(HIPBLAS_R_16F)), value :: executionType
        end function hipblasAxpyBatchedEx
    end interface

    interface
        function hipblasAxpyStridedBatchedEx(handle, n, alpha, alphaType, x, xType, incx, stridex, &
                y, yType, incy, stridey, batch_count, executionType) &
                result(c_int) &
                bind(c, name = 'hipblasAxpyStridedBatchedEx')
            use iso_c_binding
            use hipblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            integer(kind(HIPBLAS_R_16F)), value :: alphaType
            type(c_ptr), value :: x
            integer(kind(HIPBLAS_R_16F)), value :: xType
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stridex
            type(c_ptr), value :: y
            integer(kind(HIPBLAS_R_16F)), value :: yType
            integer(c_int), value :: incy
            integer(c_int64_t), value :: stridey
            integer(c_int), value :: batch_count
            integer(kind(HIPBLAS_R_16F)), value :: executionType
        end function hipblasAxpyStridedBatchedEx
    end interface

    ! dotEx
    interface
        function hipblasDotEx(handle, n, x, xType, incx, y, yType, incy, result, resultType, executionType) &
                result(c_int) &
                bind(c, name = 'hipblasDotEx')
            use iso_c_binding
            use hipblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(kind(HIPBLAS_R_16F)), value :: xType
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(kind(HIPBLAS_R_16F)), value :: yType
            integer(c_int), value :: incy
            type(c_ptr), value :: result
            integer(kind(HIPBLAS_R_16F)), value :: resultType
            integer(kind(HIPBLAS_R_16F)), value :: executionType
        end function hipblasDotEx
    end interface

    interface
        function hipblasDotcEx(handle, n, x, xType, incx, y, yType, incy, result, resultType, executionType) &
                result(c_int) &
                bind(c, name = 'hipblasDotcEx')
            use iso_c_binding
            use hipblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(kind(HIPBLAS_R_16F)), value :: xType
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(kind(HIPBLAS_R_16F)), value :: yType
            integer(c_int), value :: incy
            type(c_ptr), value :: result
            integer(kind(HIPBLAS_R_16F)), value :: resultType
            integer(kind(HIPBLAS_R_16F)), value :: executionType
        end function hipblasDotcEx
    end interface

    interface
        function hipblasDotBatchedEx(handle, n, x, xType, incx, &
                y, yType, incy, batch_count, result, resultType, executionType) &
                result(c_int) &
                bind(c, name = 'hipblasDotBatchedEx')
            use iso_c_binding
            use hipblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(kind(HIPBLAS_R_16F)), value :: xType
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(kind(HIPBLAS_R_16F)), value :: yType
            integer(c_int), value :: incy
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
            integer(kind(HIPBLAS_R_16F)), value :: resultType
            integer(kind(HIPBLAS_R_16F)), value :: executionType
        end function hipblasDotBatchedEx
    end interface

    interface
        function hipblasDotcBatchedEx(handle, n, x, xType, incx, &
                y, yType, incy, batch_count, result, resultType, executionType) &
                result(c_int) &
                bind(c, name = 'hipblasDotcBatchedEx')
            use iso_c_binding
            use hipblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(kind(HIPBLAS_R_16F)), value :: xType
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(kind(HIPBLAS_R_16F)), value :: yType
            integer(c_int), value :: incy
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
            integer(kind(HIPBLAS_R_16F)), value :: resultType
            integer(kind(HIPBLAS_R_16F)), value :: executionType
        end function hipblasDotcBatchedEx
    end interface

    interface
        function hipblasDotStridedBatchedEx(handle, n, x, xType, incx, stridex, &
                y, yType, incy, stridey, batch_count, result, resultType, executionType) &
                result(c_int) &
                bind(c, name = 'hipblasDotStridedBatchedEx')
            use iso_c_binding
            use hipblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(kind(HIPBLAS_R_16F)), value :: xType
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stridex
            type(c_ptr), value :: y
            integer(kind(HIPBLAS_R_16F)), value :: yType
            integer(c_int), value :: incy
            integer(c_int64_t), value :: stridey
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
            integer(kind(HIPBLAS_R_16F)), value :: resultType
            integer(kind(HIPBLAS_R_16F)), value :: executionType
        end function hipblasDotStridedBatchedEx
    end interface

    interface
        function hipblasDotcStridedBatchedEx(handle, n, x, xType, incx, stridex, &
                y, yType, incy, stridey, batch_count, result, resultType, executionType) &
                result(c_int) &
                bind(c, name = 'hipblasDotcStridedBatchedEx')
            use iso_c_binding
            use hipblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(kind(HIPBLAS_R_16F)), value :: xType
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stridex
            type(c_ptr), value :: y
            integer(kind(HIPBLAS_R_16F)), value :: yType
            integer(c_int), value :: incy
            integer(c_int64_t), value :: stridey
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
            integer(kind(HIPBLAS_R_16F)), value :: resultType
            integer(kind(HIPBLAS_R_16F)), value :: executionType
        end function hipblasDotcStridedBatchedEx
    end interface

    ! nrm2Ex
    interface
        function hipblasNrm2Ex(handle, n, x, xType, incx, result, resultType, executionType) &
                result(c_int) &
                bind(c, name = 'hipblasNrm2Ex')
            use iso_c_binding
            use hipblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(kind(HIPBLAS_R_16F)), value :: xType
            integer(c_int), value :: incx
            type(c_ptr), value :: result
            integer(kind(HIPBLAS_R_16F)), value :: resultType
            integer(kind(HIPBLAS_R_16F)), value :: executionType
        end function hipblasNrm2Ex
    end interface

    interface
        function hipblasNrm2BatchedEx(handle, n, x, xType, incx, &
                batch_count, result, resultType, executionType) &
                result(c_int) &
                bind(c, name = 'hipblasNrm2BatchedEx')
            use iso_c_binding
            use hipblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(kind(HIPBLAS_R_16F)), value :: xType
            integer(c_int), value :: incx
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
            integer(kind(HIPBLAS_R_16F)), value :: resultType
            integer(kind(HIPBLAS_R_16F)), value :: executionType
        end function hipblasNrm2BatchedEx
    end interface

    interface
        function hipblasNrm2StridedBatchedEx(handle, n, x, xType, incx, stridex, &
                batch_count, result, resultType, executionType) &
                result(c_int) &
                bind(c, name = 'hipblasNrm2StridedBatchedEx')
            use iso_c_binding
            use hipblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(kind(HIPBLAS_R_16F)), value :: xType
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stridex
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
            integer(kind(HIPBLAS_R_16F)), value :: resultType
            integer(kind(HIPBLAS_R_16F)), value :: executionType
        end function hipblasNrm2StridedBatchedEx
    end interface

    ! rotEx
    interface
        function hipblasRotEx(handle, n, x, xType, incx, y, yType, incy, c, s, csType, executionType) &
                result(c_int) &
                bind(c, name = 'hipblasRotEx')
            use iso_c_binding
            use hipblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(kind(HIPBLAS_R_16F)), value :: xType
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(kind(HIPBLAS_R_16F)), value :: yType
            integer(c_int), value :: incy
            type(c_ptr), value :: c
            type(c_ptr), value :: s
            integer(kind(HIPBLAS_R_16F)), value :: csType
            integer(kind(HIPBLAS_R_16F)), value :: executionType
        end function hipblasRotEx
    end interface

    interface
        function hipblasRotBatchedEx(handle, n, x, xType, incx, &
                y, yType, incy, c, s, csType, batch_count, executionType) &
                result(c_int) &
                bind(c, name = 'hipblasRotBatchedEx')
            use iso_c_binding
            use hipblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(kind(HIPBLAS_R_16F)), value :: xType
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(kind(HIPBLAS_R_16F)), value :: yType
            integer(c_int), value :: incy
            type(c_ptr), value :: c
            type(c_ptr), value :: s
            integer(kind(HIPBLAS_R_16F)), value :: csType
            integer(c_int), value :: batch_count
            integer(kind(HIPBLAS_R_16F)), value :: executionType
        end function hipblasRotBatchedEx
    end interface

    interface
        function hipblasRotStridedBatchedEx(handle, n, x, xType, incx, stridex, &
                y, yType, incy, stridey, c, s, csType, batch_count, executionType) &
                result(c_int) &
                bind(c, name = 'hipblasRotStridedBatchedEx')
            use iso_c_binding
            use hipblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(kind(HIPBLAS_R_16F)), value :: xType
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stridex
            type(c_ptr), value :: y
            integer(kind(HIPBLAS_R_16F)), value :: yType
            integer(c_int), value :: incy
            integer(c_int64_t), value :: stridey
            type(c_ptr), value :: c
            type(c_ptr), value :: s
            integer(kind(HIPBLAS_R_16F)), value :: csType
            integer(c_int), value :: batch_count
            integer(kind(HIPBLAS_R_16F)), value :: executionType
        end function hipblasRotStridedBatchedEx
    end interface

    ! scalEx
    interface
        function hipblasScalEx(handle, n, alpha, alphaType, x, xType, incx, executionType) &
                result(c_int) &
                bind(c, name = 'hipblasScalEx')
            use iso_c_binding
            use hipblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            integer(kind(HIPBLAS_R_16F)), value :: alphaType
            type(c_ptr), value :: x
            integer(kind(HIPBLAS_R_16F)), value :: xType
            integer(c_int), value :: incx
            integer(kind(HIPBLAS_R_16F)), value :: executionType
        end function hipblasScalEx
    end interface

    interface
        function hipblasScalBatchedEx(handle, n, alpha, alphaType, x, xType, incx, &
                batch_count, executionType) &
                result(c_int) &
                bind(c, name = 'hipblasScalBatchedEx')
            use iso_c_binding
            use hipblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            integer(kind(HIPBLAS_R_16F)), value :: alphaType
            type(c_ptr), value :: x
            integer(kind(HIPBLAS_R_16F)), value :: xType
            integer(c_int), value :: incx
            integer(c_int), value :: batch_count
            integer(kind(HIPBLAS_R_16F)), value :: executionType
        end function hipblasScalBatchedEx
    end interface

    interface
        function hipblasScalStridedBatchedEx(handle, n, alpha, alphaType, x, xType, incx, stridex, &
                batch_count, executionType) &
                result(c_int) &
                bind(c, name = 'hipblasScalStridedBatchedEx')
            use iso_c_binding
            use hipblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            integer(kind(HIPBLAS_R_16F)), value :: alphaType
            type(c_ptr), value :: x
            integer(kind(HIPBLAS_R_16F)), value :: xType
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stridex
            integer(c_int), value :: batch_count
            integer(kind(HIPBLAS_R_16F)), value :: executionType
        end function hipblasScalStridedBatchedEx
    end interface

    !--------!
    ! Solver !
    !--------!

    ! getrf
    interface
        function hipblasSgetrf(handle, n, A, lda, ipiv, info) &
                result(c_int) &
                bind(c, name = 'hipblasSgetrf')
            use iso_c_binding
            use hipblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: ipiv
            type(c_ptr), value :: info
        end function hipblasSgetrf
    end interface

    interface
        function hipblasDgetrf(handle, n, A, lda, ipiv, info) &
                result(c_int) &
                bind(c, name = 'hipblasDgetrf')
            use iso_c_binding
            use hipblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: ipiv
            type(c_ptr), value :: info
        end function hipblasDgetrf
    end interface

    interface
        function hipblasCgetrf(handle, n, A, lda, ipiv, info) &
                result(c_int) &
                bind(c, name = 'hipblasCgetrf')
            use iso_c_binding
            use hipblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: ipiv
            type(c_ptr), value :: info
        end function hipblasCgetrf
    end interface

    interface
        function hipblasZgetrf(handle, n, A, lda, ipiv, info) &
                result(c_int) &
                bind(c, name = 'hipblasZgetrf')
            use iso_c_binding
            use hipblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: ipiv
            type(c_ptr), value :: info
        end function hipblasZgetrf
    end interface

    ! getrf_batched
    interface
        function hipblasSgetrfBatched(handle, n, A, lda, ipiv, info, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasSgetrfBatched')
            use iso_c_binding
            use hipblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: ipiv
            type(c_ptr), value :: info
            integer(c_int), value :: batch_count
        end function hipblasSgetrfBatched
    end interface

    interface
        function hipblasDgetrfBatched(handle, n, A, lda, ipiv, info, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasDgetrfBatched')
            use iso_c_binding
            use hipblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: ipiv
            type(c_ptr), value :: info
            integer(c_int), value :: batch_count
        end function hipblasDgetrfBatched
    end interface

    interface
        function hipblasCgetrfBatched(handle, n, A, lda, ipiv, info, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasCgetrfBatched')
            use iso_c_binding
            use hipblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: ipiv
            type(c_ptr), value :: info
            integer(c_int), value :: batch_count
        end function hipblasCgetrfBatched
    end interface

    interface
        function hipblasZgetrfBatched(handle, n, A, lda, ipiv, info, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasZgetrfBatched')
            use iso_c_binding
            use hipblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: ipiv
            type(c_ptr), value :: info
            integer(c_int), value :: batch_count
        end function hipblasZgetrfBatched
    end interface

    ! getrf_strided_batched
    interface
        function hipblasSgetrfStridedBatched(handle, n, A, lda, stride_A,&
                    ipiv, stride_P, info, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasSgetrfStridedBatched')
            use iso_c_binding
            use hipblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int), value :: stride_A
            type(c_ptr), value :: ipiv
            integer(c_int), value :: stride_P
            type(c_ptr), value :: info
            integer(c_int), value :: batch_count
        end function hipblasSgetrfStridedBatched
    end interface

    interface
        function hipblasDgetrfStridedBatched(handle, n, A, lda, stride_A,&
                    ipiv, stride_P, info, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasDgetrfStridedBatched')
            use iso_c_binding
            use hipblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int), value :: stride_A
            type(c_ptr), value :: ipiv
            integer(c_int), value :: stride_P
            type(c_ptr), value :: info
            integer(c_int), value :: batch_count
        end function hipblasDgetrfStridedBatched
    end interface

    interface
        function hipblasCgetrfStridedBatched(handle, n, A, lda, stride_A,&
                    ipiv, stride_P, info, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasCgetrfStridedBatched')
            use iso_c_binding
            use hipblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int), value :: stride_A
            type(c_ptr), value :: ipiv
            integer(c_int), value :: stride_P
            type(c_ptr), value :: info
            integer(c_int), value :: batch_count
        end function hipblasCgetrfStridedBatched
    end interface

    interface
        function hipblasZgetrfStridedBatched(handle, n, A, lda, stride_A,&
                    ipiv, stride_P, info, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasZgetrfStridedBatched')
            use iso_c_binding
            use hipblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int), value :: stride_A
            type(c_ptr), value :: ipiv
            integer(c_int), value :: stride_P
            type(c_ptr), value :: info
            integer(c_int), value :: batch_count
        end function hipblasZgetrfStridedBatched
    end interface

    ! getrs
    interface
        function hipblasSgetrs(handle, trans, n, nrhs, A, lda, ipiv,&
                    B, ldb, info) &
                result(c_int) &
                bind(c, name = 'hipblasSgetrs')
            use iso_c_binding
            use hipblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(HIPBLAS_OP_N)), value :: trans
            integer(c_int), value :: n
            integer(c_int), value :: nrhs
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: ipiv
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: info
        end function hipblasSgetrs
    end interface

    interface
        function hipblasDgetrs(handle, trans, n, nrhs, A, lda, ipiv,&
                    B, ldb, info) &
                result(c_int) &
                bind(c, name = 'hipblasDgetrs')
            use iso_c_binding
            use hipblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(HIPBLAS_OP_N)), value :: trans
            integer(c_int), value :: n
            integer(c_int), value :: nrhs
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: ipiv
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: info
        end function hipblasDgetrs
    end interface

    interface
        function hipblasCgetrs(handle, trans, n, nrhs, A, lda, ipiv,&
                    B, ldb, info) &
                result(c_int) &
                bind(c, name = 'hipblasCgetrs')
            use iso_c_binding
            use hipblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(HIPBLAS_OP_N)), value :: trans
            integer(c_int), value :: n
            integer(c_int), value :: nrhs
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: ipiv
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: info
        end function hipblasCgetrs
    end interface

    interface
        function hipblasZgetrs(handle, trans, n, nrhs, A, lda, ipiv,&
                    B, ldb, info) &
                result(c_int) &
                bind(c, name = 'hipblasZgetrs')
            use iso_c_binding
            use hipblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(HIPBLAS_OP_N)), value :: trans
            integer(c_int), value :: n
            integer(c_int), value :: nrhs
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: ipiv
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: info
        end function hipblasZgetrs
    end interface

    ! getrs_batched
    interface
        function hipblasSgetrsBatched(handle, trans, n, nrhs, A, lda, ipiv,&
                    B, ldb, info, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasSgetrsBatched')
            use iso_c_binding
            use hipblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(HIPBLAS_OP_N)), value :: trans
            integer(c_int), value :: n
            integer(c_int), value :: nrhs
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: ipiv
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: info
            integer(c_int), value :: batch_count
        end function hipblasSgetrsBatched
    end interface

    interface
        function hipblasDgetrsBatched(handle, trans, n, nrhs, A, lda, ipiv,&
                    B, ldb, info, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasDgetrsBatched')
            use iso_c_binding
            use hipblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(HIPBLAS_OP_N)), value :: trans
            integer(c_int), value :: n
            integer(c_int), value :: nrhs
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: ipiv
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: info
            integer(c_int), value :: batch_count
        end function hipblasDgetrsBatched
    end interface

    interface
        function hipblasCgetrsBatched(handle, trans, n, nrhs, A, lda, ipiv,&
                    B, ldb, info, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasCgetrsBatched')
            use iso_c_binding
            use hipblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(HIPBLAS_OP_N)), value :: trans
            integer(c_int), value :: n
            integer(c_int), value :: nrhs
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: ipiv
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: info
            integer(c_int), value :: batch_count
        end function hipblasCgetrsBatched
    end interface

    interface
        function hipblasZgetrsBatched(handle, trans, n, nrhs, A, lda, ipiv,&
                    B, ldb, info, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasZgetrsBatched')
            use iso_c_binding
            use hipblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(HIPBLAS_OP_N)), value :: trans
            integer(c_int), value :: n
            integer(c_int), value :: nrhs
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: ipiv
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: info
            integer(c_int), value :: batch_count
        end function hipblasZgetrsBatched
    end interface

    ! getrs_strided_batched
    interface
        function hipblasSgetrsStridedBatched(handle, trans, n, nrhs, A, lda, stride_A, ipiv,&
                    stride_P, B, ldb, stride_B, info, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasSgetrsStridedBatched')
            use iso_c_binding
            use hipblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(HIPBLAS_OP_N)), value :: trans
            integer(c_int), value :: n
            integer(c_int), value :: nrhs
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int), value :: stride_A
            type(c_ptr), value :: ipiv
            integer(c_int), value :: stride_P
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            integer(c_int), value :: stride_B
            type(c_ptr), value :: info
            integer(c_int), value :: batch_count
        end function hipblasSgetrsStridedBatched
    end interface

    interface
        function hipblasDgetrsStridedBatched(handle, trans, n, nrhs, A, lda, stride_A, ipiv,&
                    stride_P, B, ldb, stride_B, info, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasDgetrsStridedBatched')
            use iso_c_binding
            use hipblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(HIPBLAS_OP_N)), value :: trans
            integer(c_int), value :: n
            integer(c_int), value :: nrhs
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int), value :: stride_A
            type(c_ptr), value :: ipiv
            integer(c_int), value :: stride_P
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            integer(c_int), value :: stride_B
            type(c_ptr), value :: info
            integer(c_int), value :: batch_count
        end function hipblasDgetrsStridedBatched
    end interface

    interface
        function hipblasCgetrsStridedBatched(handle, trans, n, nrhs, A, lda, stride_A, ipiv,&
                    stride_P, B, ldb, stride_B, info, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasCgetrsStridedBatched')
            use iso_c_binding
            use hipblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(HIPBLAS_OP_N)), value :: trans
            integer(c_int), value :: n
            integer(c_int), value :: nrhs
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int), value :: stride_A
            type(c_ptr), value :: ipiv
            integer(c_int), value :: stride_P
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            integer(c_int), value :: stride_B
            type(c_ptr), value :: info
            integer(c_int), value :: batch_count
        end function hipblasCgetrsStridedBatched
    end interface

    interface
        function hipblasZgetrsStridedBatched(handle, trans, n, nrhs, A, lda, stride_A, ipiv,&
                    stride_P, B, ldb, stride_B, info, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasZgetrsStridedBatched')
            use iso_c_binding
            use hipblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(HIPBLAS_OP_N)), value :: trans
            integer(c_int), value :: n
            integer(c_int), value :: nrhs
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int), value :: stride_A
            type(c_ptr), value :: ipiv
            integer(c_int), value :: stride_P
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            integer(c_int), value :: stride_B
            type(c_ptr), value :: info
            integer(c_int), value :: batch_count
        end function hipblasZgetrsStridedBatched
    end interface

    ! getri_batched
    interface
        function hipblasSgetriBatched(handle, n, A, lda, ipiv, C, ldc, info, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasSgetriBatched')
            use iso_c_binding
            use hipblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: ipiv
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
            type(c_ptr), value :: info
            integer(c_int), value :: batch_count
        end function hipblasSgetriBatched
    end interface

    interface
        function hipblasDgetriBatched(handle, n, A, lda, ipiv, C, ldc, info, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasDgetriBatched')
            use iso_c_binding
            use hipblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: ipiv
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
            type(c_ptr), value :: info
            integer(c_int), value :: batch_count
        end function hipblasDgetriBatched
    end interface

    interface
        function hipblasCgetriBatched(handle, n, A, lda, ipiv, C, ldc, info, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasCgetriBatched')
            use iso_c_binding
            use hipblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: ipiv
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
            type(c_ptr), value :: info
            integer(c_int), value :: batch_count
        end function hipblasCgetriBatched
    end interface

    interface
        function hipblasZgetriBatched(handle, n, A, lda, ipiv, C, ldc, info, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasZgetriBatched')
            use iso_c_binding
            use hipblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: ipiv
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
            type(c_ptr), value :: info
            integer(c_int), value :: batch_count
        end function hipblasZgetriBatched
    end interface

    ! geqrf
    interface
        function hipblasSgeqrf(handle, m, n, A, lda, tau, info) &
                result(c_int) &
                bind(c, name = 'hipblasSgeqrf')
            use iso_c_binding
            use hipblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: tau
            type(c_ptr), value :: info
        end function hipblasSgeqrf
    end interface

    interface
        function hipblasDgeqrf(handle, m, n, A, lda, tau, info) &
                result(c_int) &
                bind(c, name = 'hipblasDgeqrf')
            use iso_c_binding
            use hipblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: tau
            type(c_ptr), value :: info
        end function hipblasDgeqrf
    end interface

    interface
        function hipblasCgeqrf(handle, m, n, A, lda, tau, info) &
                result(c_int) &
                bind(c, name = 'hipblasCgeqrf')
            use iso_c_binding
            use hipblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: tau
            type(c_ptr), value :: info
        end function hipblasCgeqrf
    end interface

    interface
        function hipblasZgeqrf(handle, m, n, A, lda, tau, info) &
                result(c_int) &
                bind(c, name = 'hipblasZgeqrf')
            use iso_c_binding
            use hipblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: tau
            type(c_ptr), value :: info
        end function hipblasZgeqrf
    end interface

    ! geqrf_batched
    interface
        function hipblasSgeqrfBatched(handle, m, n, A, lda, tau, info, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasSgeqrfBatched')
            use iso_c_binding
            use hipblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: tau
            type(c_ptr), value :: info
            integer(c_int), value :: batch_count
        end function hipblasSgeqrfBatched
    end interface

    interface
        function hipblasDgeqrfBatched(handle, m, n, A, lda, tau, info, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasDgeqrfBatched')
            use iso_c_binding
            use hipblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: tau
            type(c_ptr), value :: info
            integer(c_int), value :: batch_count
        end function hipblasDgeqrfBatched
    end interface

    interface
        function hipblasCgeqrfBatched(handle, m, n, A, lda, tau, info, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasCgeqrfBatched')
            use iso_c_binding
            use hipblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: tau
            type(c_ptr), value :: info
            integer(c_int), value :: batch_count
        end function hipblasCgeqrfBatched
    end interface

    interface
        function hipblasZgeqrfBatched(handle, m, n, A, lda, tau, info, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasZgeqrfBatched')
            use iso_c_binding
            use hipblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: tau
            type(c_ptr), value :: info
            integer(c_int), value :: batch_count
        end function hipblasZgeqrfBatched
    end interface

    ! geqrf_strided_batched
    interface
        function hipblasSgeqrfStridedBatched(handle, m, n, A, lda, stride_A,&
                    tau, stride_T, info, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasSgeqrfStridedBatched')
            use iso_c_binding
            use hipblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int), value :: stride_A
            type(c_ptr), value :: tau
            integer(c_int), value :: stride_T
            type(c_ptr), value :: info
            integer(c_int), value :: batch_count
        end function hipblasSgeqrfStridedBatched
    end interface

    interface
    function hipblasDgeqrfStridedBatched(handle, m, n, A, lda, stride_A,&
                tau, stride_T, info, batch_count) &
            result(c_int) &
            bind(c, name = 'hipblasDgeqrfStridedBatched')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: m
        integer(c_int), value :: n
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        integer(c_int), value :: stride_A
        type(c_ptr), value :: tau
        integer(c_int), value :: stride_T
        type(c_ptr), value :: info
        integer(c_int), value :: batch_count
    end function hipblasDgeqrfStridedBatched
    end interface

    interface
        function hipblasCgeqrfStridedBatched(handle, m, n, A, lda, stride_A,&
                    tau, stride_T, info, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasCgeqrfStridedBatched')
            use iso_c_binding
            use hipblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int), value :: stride_A
            type(c_ptr), value :: tau
            integer(c_int), value :: stride_T
            type(c_ptr), value :: info
            integer(c_int), value :: batch_count
        end function hipblasCgeqrfStridedBatched
    end interface

    interface
        function hipblasZgeqrfStridedBatched(handle, m, n, A, lda, stride_A,&
                    tau, stride_T, info, batch_count) &
                result(c_int) &
                bind(c, name = 'hipblasZgeqrfStridedBatched')
            use iso_c_binding
            use hipblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int), value :: stride_A
            type(c_ptr), value :: tau
            integer(c_int), value :: stride_T
            type(c_ptr), value :: info
            integer(c_int), value :: batch_count
        end function hipblasZgeqrfStridedBatched
    end interface

end module hipblas
