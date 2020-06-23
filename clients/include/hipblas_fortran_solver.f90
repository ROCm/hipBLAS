module hipblas_interface
    use iso_c_binding
    use hipblas

    contains

    !--------!
    ! Solver !
    !--------!

    ! getrf
    function hipblasSgetrfFortran(handle, n, A, lda, ipiv, info) &
            result(res) &
            bind(c, name = 'hipblasSgetrfFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: ipiv
        type(c_ptr), value :: info
        integer(c_int) :: res
        res = hipblasSgetrf(handle, n, A, lda, ipiv, info)
    end function hipblasSgetrfFortran

    function hipblasDgetrfFortran(handle, n, A, lda, ipiv, info) &
            result(res) &
            bind(c, name = 'hipblasDgetrfFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: ipiv
        type(c_ptr), value :: info
        integer(c_int) :: res
        res = hipblasDgetrf(handle, n, A, lda, ipiv, info)
    end function hipblasDgetrfFortran

    function hipblasCgetrfFortran(handle, n, A, lda, ipiv, info) &
            result(res) &
            bind(c, name = 'hipblasCgetrfFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: ipiv
        type(c_ptr), value :: info
        integer(c_int) :: res
        res = hipblasCgetrf(handle, n, A, lda, ipiv, info)
    end function hipblasCgetrfFortran

    function hipblasZgetrfFortran(handle, n, A, lda, ipiv, info) &
            result(res) &
            bind(c, name = 'hipblasZgetrfFortran')
        use iso_c_binding
        use hipblas_enums
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: ipiv
        type(c_ptr), value :: info
        integer(c_int) :: res
        res = hipblasZgetrf(handle, n, A, lda, ipiv, info)
    end function hipblasZgetrfFortran

    ! getrf_batched
    function hipblasSgetrfBatchedFortran(handle, n, A, lda, ipiv, info, batch_count) &
            result(res) &
            bind(c, name = 'hipblasSgetrfBatchedFortran')
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
        integer(c_int) :: res
        res = hipblasSgetrfBatched(handle, n, A, lda, ipiv, info, batch_count)
    end function hipblasSgetrfBatchedFortran

    function hipblasDgetrfBatchedFortran(handle, n, A, lda, ipiv, info, batch_count) &
            result(res) &
            bind(c, name = 'hipblasDgetrfBatchedFortran')
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
        integer(c_int) :: res
        res = hipblasDgetrfBatched(handle, n, A, lda, ipiv, info, batch_count)
    end function hipblasDgetrfBatchedFortran

    function hipblasCgetrfBatchedFortran(handle, n, A, lda, ipiv, info, batch_count) &
            result(res) &
            bind(c, name = 'hipblasCgetrfBatchedFortran')
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
        integer(c_int) :: res
        res = hipblasCgetrfBatched(handle, n, A, lda, ipiv, info, batch_count)
    end function hipblasCgetrfBatchedFortran

    function hipblasZgetrfBatchedFortran(handle, n, A, lda, ipiv, info, batch_count) &
            result(res) &
            bind(c, name = 'hipblasZgetrfBatchedFortran')
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
        integer(c_int) :: res
        res = hipblasZgetrfBatched(handle, n, A, lda, ipiv, info, batch_count)
    end function hipblasZgetrfBatchedFortran

    ! getrf_strided_batched
    function hipblasSgetrfStridedBatchedFortran(handle, n, A, lda, stride_A,&
                ipiv, stride_P, info, batch_count) &
            result(res) &
            bind(c, name = 'hipblasSgetrfStridedBatchedFortran')
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
        integer(c_int) :: res
        res = hipblasSgetrfStridedBatched(handle, n, A, lda, stride_A,&
                    ipiv, stride_P, info, batch_count)
    end function hipblasSgetrfStridedBatchedFortran

    function hipblasDgetrfStridedBatchedFortran(handle, n, A, lda, stride_A,&
                ipiv, stride_P, info, batch_count) &
            result(res) &
            bind(c, name = 'hipblasDgetrfStridedBatchedFortran')
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
        integer(c_int) :: res
        res = hipblasDgetrfStridedBatched(handle, n, A, lda, stride_A,&
                    ipiv, stride_P, info, batch_count)
    end function hipblasDgetrfStridedBatchedFortran

    function hipblasCgetrfStridedBatchedFortran(handle, n, A, lda, stride_A,&
                ipiv, stride_P, info, batch_count) &
            result(res) &
            bind(c, name = 'hipblasCgetrfStridedBatchedFortran')
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
        integer(c_int) :: res
        res = hipblasCgetrfStridedBatched(handle, n, A, lda, stride_A,&
                    ipiv, stride_P, info, batch_count)
    end function hipblasCgetrfStridedBatchedFortran

    function hipblasZgetrfStridedBatchedFortran(handle, n, A, lda, stride_A,&
                ipiv, stride_P, info, batch_count) &
            result(res) &
            bind(c, name = 'hipblasZgetrfStridedBatchedFortran')
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
        integer(c_int) :: res
        res = hipblasZgetrfStridedBatched(handle, n, A, lda, stride_A,&
                    ipiv, stride_P, info, batch_count)
    end function hipblasZgetrfStridedBatchedFortran

    ! getrs
    function hipblasSgetrsFortran(handle, trans, n, nrhs, A, lda, ipiv,&
                B, ldb, info) &
            result(res) &
            bind(c, name = 'hipblasSgetrsFortran')
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
        integer(c_int) :: res
        res = hipblasSgetrs(handle, trans, n, nrhs, A, lda,&
                ipiv, B, ldb, info)
    end function hipblasSgetrsFortran

    function hipblasDgetrsFortran(handle, trans, n, nrhs, A, lda, ipiv,&
                B, ldb, info) &
            result(res) &
            bind(c, name = 'hipblasDgetrsFortran')
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
        integer(c_int) :: res
        res = hipblasDgetrs(handle, trans, n, nrhs, A, lda,&
                ipiv, B, ldb, info)
    end function hipblasDgetrsFortran

    function hipblasCgetrsFortran(handle, trans, n, nrhs, A, lda, ipiv,&
                B, ldb, info) &
            result(res) &
            bind(c, name = 'hipblasCgetrsFortran')
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
        integer(c_int) :: res
        res = hipblasCgetrs(handle, trans, n, nrhs, A, lda,&
                ipiv, B, ldb, info)
    end function hipblasCgetrsFortran

    function hipblasZgetrsFortran(handle, trans, n, nrhs, A, lda, ipiv,&
                B, ldb, info) &
            result(res) &
            bind(c, name = 'hipblasZgetrsFortran')
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
        integer(c_int) :: res
        res = hipblasZgetrs(handle, trans, n, nrhs, A, lda,&
                ipiv, B, ldb, info)
    end function hipblasZgetrsFortran

    ! getrs_batched
    function hipblasSgetrsBatchedFortran(handle, trans, n, nrhs, A, lda, ipiv,&
                B, ldb, info, batch_count) &
            result(res) &
            bind(c, name = 'hipblasSgetrsBatchedFortran')
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
        integer(c_int) :: res
        res = hipblasSgetrsBatched(handle, trans, n, nrhs, A, lda,&
                ipiv, B, ldb, info, batch_count)
    end function hipblasSgetrsBatchedFortran

    function hipblasDgetrsBatchedFortran(handle, trans, n, nrhs, A, lda, ipiv,&
                B, ldb, info, batch_count) &
            result(res) &
            bind(c, name = 'hipblasDgetrsBatchedFortran')
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
        integer(c_int) :: res
        res = hipblasDgetrsBatched(handle, trans, n, nrhs, A, lda,&
                ipiv, B, ldb, info, batch_count)
    end function hipblasDgetrsBatchedFortran

    function hipblasCgetrsBatchedFortran(handle, trans, n, nrhs, A, lda, ipiv,&
                B, ldb, info, batch_count) &
            result(res) &
            bind(c, name = 'hipblasCgetrsBatchedFortran')
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
        integer(c_int) :: res
        res = hipblasCgetrsBatched(handle, trans, n, nrhs, A, lda,&
                ipiv, B, ldb, info, batch_count)
    end function hipblasCgetrsBatchedFortran

    function hipblasZgetrsBatchedFortran(handle, trans, n, nrhs, A, lda, ipiv,&
                B, ldb, info, batch_count) &
            result(res) &
            bind(c, name = 'hipblasZgetrsBatchedFortran')
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
        integer(c_int) :: res
        res = hipblasZgetrsBatched(handle, trans, n, nrhs, A, lda,&
                ipiv, B, ldb, info, batch_count)
    end function hipblasZgetrsBatchedFortran

    ! getrs_strided_batched
    function hipblasSgetrsStridedBatchedFortran(handle, trans, n, nrhs, A, lda, stride_A, ipiv,&
                stride_P, B, ldb, stride_B, info, batch_count) &
            result(res) &
            bind(c, name = 'hipblasSgetrsStridedBatchedFortran')
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
        integer(c_int) :: res
        res = hipblasSgetrsStridedBatched(handle, trans, n, nrhs, A, lda, stride_A,&
                ipiv, stride_P, B, ldb, stride_B, info, batch_count)
    end function hipblasSgetrsStridedBatchedFortran

    function hipblasDgetrsStridedBatchedFortran(handle, trans, n, nrhs, A, lda, stride_A, ipiv,&
                stride_P, B, ldb, stride_B, info, batch_count) &
            result(res) &
            bind(c, name = 'hipblasDgetrsStridedBatchedFortran')
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
        integer(c_int) :: res
        res = hipblasDgetrsStridedBatched(handle, trans, n, nrhs, A, lda, stride_A,&
                ipiv, stride_P, B, ldb, stride_B, info, batch_count)
    end function hipblasDgetrsStridedBatchedFortran

    function hipblasCgetrsStridedBatchedFortran(handle, trans, n, nrhs, A, lda, stride_A, ipiv,&
                stride_P, B, ldb, stride_B, info, batch_count) &
            result(res) &
            bind(c, name = 'hipblasCgetrsStridedBatchedFortran')
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
        integer(c_int) :: res
        res = hipblasCgetrsStridedBatched(handle, trans, n, nrhs, A, lda, stride_A,&
                ipiv, stride_P, B, ldb, stride_B, info, batch_count)
    end function hipblasCgetrsStridedBatchedFortran

    function hipblasZgetrsStridedBatchedFortran(handle, trans, n, nrhs, A, lda, stride_A, ipiv,&
                stride_P, B, ldb, stride_B, info, batch_count) &
            result(res) &
            bind(c, name = 'hipblasZgetrsStridedBatchedFortran')
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
        integer(c_int) :: res
        res = hipblasZgetrsStridedBatched(handle, trans, n, nrhs, A, lda, stride_A,&
                ipiv, stride_P, B, ldb, stride_B, info, batch_count)
    end function hipblasZgetrsStridedBatchedFortran

    ! getri_batched
    function hipblasSgetriBatchedFortran(handle, n, A, lda, ipiv, C, ldc, info, batch_count) &
            result(res) &
            bind(c, name = 'hipblasSgetriBatchedFortran')
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
        integer(c_int) :: res
        res = hipblasSgetriBatched(handle, n, A, lda, ipiv, C, ldc, info, batch_count)
    end function hipblasSgetriBatchedFortran

    function hipblasDgetriBatchedFortran(handle, n, A, lda, ipiv, C, ldc, info, batch_count) &
            result(res) &
            bind(c, name = 'hipblasDgetriBatchedFortran')
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
        integer(c_int) :: res
        res = hipblasDgetriBatched(handle, n, A, lda, ipiv, C, ldc, info, batch_count)
    end function hipblasDgetriBatchedFortran

    function hipblasCgetriBatchedFortran(handle, n, A, lda, ipiv, C, ldc, info, batch_count) &
            result(res) &
            bind(c, name = 'hipblasCgetriBatchedFortran')
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
        integer(c_int) :: res
        res = hipblasCgetriBatched(handle, n, A, lda, ipiv, C, ldc, info, batch_count)
    end function hipblasCgetriBatchedFortran

    function hipblasZgetriBatchedFortran(handle, n, A, lda, ipiv, C, ldc, info, batch_count) &
            result(res) &
            bind(c, name = 'hipblasZgetriBatchedFortran')
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
        integer(c_int) :: res
        res = hipblasZgetriBatched(handle, n, A, lda, ipiv, C, ldc, info, batch_count)
    end function hipblasZgetriBatchedFortran

    ! geqrf
    function hipblasSgeqrfFortran(handle, m, n, A, lda, tau, info) &
            result(res) &
            bind(c, name = 'hipblasSgeqrfFortran')
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
        integer(c_int) :: res
        res = hipblasSgeqrf(handle, m, n, A, lda, tau, info)
    end function hipblasSgeqrfFortran

    function hipblasDgeqrfFortran(handle, m, n, A, lda, tau, info) &
            result(res) &
            bind(c, name = 'hipblasDgeqrfFortran')
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
        integer(c_int) :: res
        res = hipblasDgeqrf(handle, m, n, A, lda, tau, info)
    end function hipblasDgeqrfFortran

    function hipblasCgeqrfFortran(handle, m, n, A, lda, tau, info) &
            result(res) &
            bind(c, name = 'hipblasCgeqrfFortran')
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
        integer(c_int) :: res
        res = hipblasCgeqrf(handle, m, n, A, lda, tau, info)
    end function hipblasCgeqrfFortran

    function hipblasZgeqrfFortran(handle, m, n, A, lda, tau, info) &
            result(res) &
            bind(c, name = 'hipblasZgeqrfFortran')
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
        integer(c_int) :: res
        res = hipblasZgeqrf(handle, m, n, A, lda, tau, info)
    end function hipblasZgeqrfFortran

    ! geqrf_batched
    function hipblasSgeqrfBatchedFortran(handle, m, n, A, lda, tau, info, batch_count) &
            result(res) &
            bind(c, name = 'hipblasSgeqrfBatchedFortran')
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
        integer(c_int) :: res
        res = hipblasSgeqrfBatched(handle, m, n, A, lda, tau, info, batch_count)
    end function hipblasSgeqrfBatchedFortran

    function hipblasDgeqrfBatchedFortran(handle, m, n, A, lda, tau, info, batch_count) &
            result(res) &
            bind(c, name = 'hipblasDgeqrfBatchedFortran')
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
        integer(c_int) :: res
        res = hipblasDgeqrfBatched(handle, m, n, A, lda, tau, info, batch_count)
    end function hipblasDgeqrfBatchedFortran

    function hipblasCgeqrfBatchedFortran(handle, m, n, A, lda, tau, info, batch_count) &
            result(res) &
            bind(c, name = 'hipblasCgeqrfBatchedFortran')
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
        integer(c_int) :: res
        res = hipblasCgeqrfBatched(handle, m, n, A, lda, tau, info, batch_count)
    end function hipblasCgeqrfBatchedFortran

    function hipblasZgeqrfBatchedFortran(handle, m, n, A, lda, tau, info, batch_count) &
            result(res) &
            bind(c, name = 'hipblasZgeqrfBatchedFortran')
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
        integer(c_int) :: res
        res = hipblasZgeqrfBatched(handle, m, n, A, lda, tau, info, batch_count)
    end function hipblasZgeqrfBatchedFortran

    ! geqrf_strided_batched
    function hipblasSgeqrfStridedBatchedFortran(handle, m, n, A, lda, stride_A,&
                tau, stride_T, info, batch_count) &
            result(res) &
            bind(c, name = 'hipblasSgeqrfStridedBatchedFortran')
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
        integer(c_int) :: res
        res = hipblasSgeqrfStridedBatched(handle, m, n, A, lda, stride_A,&
                    tau, stride_T, info, batch_count)
    end function hipblasSgeqrfStridedBatchedFortran

function hipblasDgeqrfStridedBatchedFortran(handle, m, n, A, lda, stride_A,&
            tau, stride_T, info, batch_count) &
        result(res) &
        bind(c, name = 'hipblasDgeqrfStridedBatchedFortran')
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
    integer(c_int) :: res
    res = hipblasDgeqrfStridedBatched(handle, m, n, A, lda, stride_A,&
                    tau, stride_T, info, batch_count)
end function hipblasDgeqrfStridedBatchedFortran

    function hipblasCgeqrfStridedBatchedFortran(handle, m, n, A, lda, stride_A,&
                tau, stride_T, info, batch_count) &
            result(res) &
            bind(c, name = 'hipblasCgeqrfStridedBatchedFortran')
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
        integer(c_int) :: res
        res = hipblasCgeqrfStridedBatched(handle, m, n, A, lda, stride_A,&
                    tau, stride_T, info, batch_count)
    end function hipblasCgeqrfStridedBatchedFortran

    function hipblasZgeqrfStridedBatchedFortran(handle, m, n, A, lda, stride_A,&
                tau, stride_T, info, batch_count) &
            result(res) &
            bind(c, name = 'hipblasZgeqrfStridedBatchedFortran')
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
        integer(c_int) :: res
        res = hipblasZgeqrfStridedBatched(handle, m, n, A, lda, stride_A,&
                    tau, stride_T, info, batch_count)
    end function hipblasZgeqrfStridedBatchedFortran

end module hipblas_interface