! Hello.


module hipblas
    use iso_c_binding

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

end module hipblas