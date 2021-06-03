#ifndef _HIPBLAS_NO_FORTRAN_HPP
#define _HIPBLAS_NO_FORTRAN_HPP

/*!\file
 *  This file interfaces with our Fortran BLAS interface.
 */

/*
 * ============================================================================
 *     Fortran functions
 * ============================================================================
 */


/* ==========
 *    Aux
 * ========== */
#define hipblasSetVectorFortran hipblasSetVector
#define hipblasGetVectorFortran hipblasGetVector
#define hipblasSetMatrixFortran hipblasSetMatrix
#define hipblasGetMatrixFortran hipblasGetMatrix
#define hipblasSetVectorAsyncFortran hipblasSetVectorAsync
#define hipblasGetVectorAsyncFortran hipblasGetVectorAsync
#define hipblasSetMatrixAsyncFortran hipblasSetMatrixAsync
#define hipblasGetMatrixAsyncFortran hipblasGetMatrixAsync
#define hipblasSetAtomicsModeFortran hipblasSetAtomicsMode
#define hipblasGetAtomicsModeFortran hipblasGetAtomicsMode

/* ==========
 *    L1
 * ========== */

#define hipblasSscalFortran hipblasSscal
#define hipblasDscalFortran hipblasDscal
#define hipblasCscalFortran hipblasCscal
#define hipblasZscalFortran hipblasZscal
#define hipblasCsscalFortran hipblasCsscal
#define hipblasZdscalFortran hipblasZdscal
#define hipblasSscalBatchedFortran hipblasSscalBatched
#define hipblasDscalBatchedFortran hipblasDscalBatched
#define hipblasCscalBatchedFortran hipblasCscalBatched
#define hipblasZscalBatchedFortran hipblasZscalBatched
#define hipblasCsscalBatchedFortran hipblasCsscalBatched
#define hipblasZdscalBatchedFortran hipblasZdscalBatched
#define hipblasSscalStridedBatchedFortran hipblasSscalStridedBatched
#define hipblasDscalStridedBatchedFortran hipblasDscalStridedBatched
#define hipblasCscalStridedBatchedFortran hipblasCscalStridedBatched
#define hipblasZscalStridedBatchedFortran hipblasZscalStridedBatched
#define hipblasCsscalStridedBatchedFortran hipblasCsscalStridedBatched
#define hipblasZdscalStridedBatchedFortran hipblasZdscalStridedBatched
#define hipblasScopyFortran hipblasScopy
#define hipblasDcopyFortran hipblasDcopy
#define hipblasCcopyFortran hipblasCcopy
#define hipblasZcopyFortran hipblasZcopy
#define hipblasScopyBatchedFortran hipblasScopyBatched
#define hipblasDcopyBatchedFortran hipblasDcopyBatched
#define hipblasCcopyBatchedFortran hipblasCcopyBatched
#define hipblasZcopyBatchedFortran hipblasZcopyBatched
#define hipblasScopyStridedBatchedFortran hipblasScopyStridedBatched
#define hipblasDcopyStridedBatchedFortran hipblasDcopyStridedBatched
#define hipblasCcopyStridedBatchedFortran hipblasCcopyStridedBatched
#define hipblasZcopyStridedBatchedFortran hipblasZcopyStridedBatched
#define hipblasSdotFortran hipblasSdot
#define hipblasDdotFortran hipblasDdot
#define hipblasHdotFortran hipblasHdot
#define hipblasBfdotFortran hipblasBfdot
#define hipblasCdotuFortran hipblasCdotu
#define hipblasCdotcFortran hipblasCdotc
#define hipblasZdotuFortran hipblasZdotu
#define hipblasZdotcFortran hipblasZdotc
#define hipblasSdotBatchedFortran hipblasSdotBatched
#define hipblasDdotBatchedFortran hipblasDdotBatched
#define hipblasHdotBatchedFortran hipblasHdotBatched
#define hipblasBfdotBatchedFortran hipblasBfdotBatched
#define hipblasCdotuBatchedFortran hipblasCdotuBatched
#define hipblasCdotcBatchedFortran hipblasCdotcBatched
#define hipblasZdotuBatchedFortran hipblasZdotuBatched
#define hipblasZdotcBatchedFortran hipblasZdotcBatched
#define hipblasSdotStridedBatchedFortran hipblasSdotStridedBatched
#define hipblasDdotStridedBatchedFortran hipblasDdotStridedBatched
#define hipblasHdotStridedBatchedFortran hipblasHdotStridedBatched
#define hipblasBfdotStridedBatchedFortran hipblasBfdotStridedBatched
#define hipblasCdotuStridedBatchedFortran hipblasCdotuStridedBatched
#define hipblasCdotcStridedBatchedFortran hipblasCdotcStridedBatched
#define hipblasZdotuStridedBatchedFortran hipblasZdotuStridedBatched
#define hipblasZdotcStridedBatchedFortran hipblasZdotcStridedBatched
#define hipblasSswapFortran hipblasSswap
#define hipblasDswapFortran hipblasDswap
#define hipblasCswapFortran hipblasCswap
#define hipblasZswapFortran hipblasZswap
#define hipblasSswapBatchedFortran hipblasSswapBatched
#define hipblasDswapBatchedFortran hipblasDswapBatched
#define hipblasCswapBatchedFortran hipblasCswapBatched
#define hipblasZswapBatchedFortran hipblasZswapBatched
#define hipblasSswapStridedBatchedFortran hipblasSswapStridedBatched
#define hipblasDswapStridedBatchedFortran hipblasDswapStridedBatched
#define hipblasCswapStridedBatchedFortran hipblasCswapStridedBatched
#define hipblasZswapStridedBatchedFortran hipblasZswapStridedBatched
#define hipblasHaxpyFortran hipblasHaxpy
#define hipblasSaxpyFortran hipblasSaxpy
#define hipblasDaxpyFortran hipblasDaxpy
#define hipblasCaxpyFortran hipblasCaxpy
#define hipblasZaxpyFortran hipblasZaxpy
#define hipblasHaxpyBatchedFortran hipblasHaxpyBatched
#define hipblasSaxpyBatchedFortran hipblasSaxpyBatched
#define hipblasDaxpyBatchedFortran hipblasDaxpyBatched
#define hipblasCaxpyBatchedFortran hipblasCaxpyBatched
#define hipblasZaxpyBatchedFortran hipblasZaxpyBatched
#define hipblasHaxpyStridedBatchedFortran hipblasHaxpyStridedBatched
#define hipblasSaxpyStridedBatchedFortran hipblasSaxpyStridedBatched
#define hipblasDaxpyStridedBatchedFortran hipblasDaxpyStridedBatched
#define hipblasCaxpyStridedBatchedFortran hipblasCaxpyStridedBatched
#define hipblasZaxpyStridedBatchedFortran hipblasZaxpyStridedBatched
#define hipblasSasumFortran hipblasSasum
#define hipblasDasumFortran hipblasDasum
#define hipblasScasumFortran hipblasScasum
#define hipblasDzasumFortran hipblasDzasum
#define hipblasSasumBatchedFortran hipblasSasumBatched
#define hipblasDasumBatchedFortran hipblasDasumBatched
#define hipblasScasumBatchedFortran hipblasScasumBatched
#define hipblasDzasumBatchedFortran hipblasDzasumBatched
#define hipblasSasumStridedBatchedFortran hipblasSasumStridedBatched
#define hipblasDasumStridedBatchedFortran hipblasDasumStridedBatched
#define hipblasScasumStridedBatchedFortran hipblasScasumStridedBatched
#define hipblasDzasumStridedBatchedFortran hipblasDzasumStridedBatched
#define hipblasSnrm2Fortran hipblasSnrm2
#define hipblasDnrm2Fortran hipblasDnrm2
#define hipblasScnrm2Fortran hipblasScnrm2
#define hipblasDznrm2Fortran hipblasDznrm2
#define hipblasSnrm2BatchedFortran hipblasSnrm2Batched
#define hipblasDnrm2BatchedFortran hipblasDnrm2Batched
#define hipblasScnrm2BatchedFortran hipblasScnrm2Batched
#define hipblasDznrm2BatchedFortran hipblasDznrm2Batched
#define hipblasSnrm2StridedBatchedFortran hipblasSnrm2StridedBatched
#define hipblasDnrm2StridedBatchedFortran hipblasDnrm2StridedBatched
#define hipblasScnrm2StridedBatchedFortran hipblasScnrm2StridedBatched
#define hipblasDznrm2StridedBatchedFortran hipblasDznrm2StridedBatched
#define hipblasIsamaxFortran hipblasIsamax
#define hipblasIdamaxFortran hipblasIdamax
#define hipblasIcamaxFortran hipblasIcamax
#define hipblasIzamaxFortran hipblasIzamax
#define hipblasIsamaxBatchedFortran hipblasIsamaxBatched
#define hipblasIdamaxBatchedFortran hipblasIdamaxBatched
#define hipblasIcamaxBatchedFortran hipblasIcamaxBatched
#define hipblasIzamaxBatchedFortran hipblasIzamaxBatched
#define hipblasIsamaxStridedBatchedFortran hipblasIsamaxStridedBatched
#define hipblasIdamaxStridedBatchedFortran hipblasIdamaxStridedBatched
#define hipblasIcamaxStridedBatchedFortran hipblasIcamaxStridedBatched
#define hipblasIzamaxStridedBatchedFortran hipblasIzamaxStridedBatched
#define hipblasIsaminFortran hipblasIsamin
#define hipblasIdaminFortran hipblasIdamin
#define hipblasIcaminFortran hipblasIcamin
#define hipblasIzaminFortran hipblasIzamin
#define hipblasIsaminBatchedFortran hipblasIsaminBatched
#define hipblasIdaminBatchedFortran hipblasIdaminBatched
#define hipblasIcaminBatchedFortran hipblasIcaminBatched
#define hipblasIzaminBatchedFortran hipblasIzaminBatched
#define hipblasIsaminStridedBatchedFortran hipblasIsaminStridedBatched
#define hipblasIdaminStridedBatchedFortran hipblasIdaminStridedBatched
#define hipblasIcaminStridedBatchedFortran hipblasIcaminStridedBatched
#define hipblasIzaminStridedBatchedFortran hipblasIzaminStridedBatched
#define hipblasSrotFortran hipblasSrot
#define hipblasDrotFortran hipblasDrot
#define hipblasCsrotFortran hipblasCsrot
#define hipblasZdrotFortran hipblasZdrot
#define hipblasCrotFortran hipblasCrot
#define hipblasZrotFortran hipblasZrot
#define hipblasSrotBatchedFortran hipblasSrotBatched
#define hipblasDrotBatchedFortran hipblasDrotBatched
#define hipblasCsrotBatchedFortran hipblasCsrotBatched
#define hipblasZdrotBatchedFortran hipblasZdrotBatched
#define hipblasCrotBatchedFortran hipblasCrotBatched
#define hipblasZrotBatchedFortran hipblasZrotBatched
#define hipblasSrotStridedBatchedFortran hipblasSrotStridedBatched
#define hipblasDrotStridedBatchedFortran hipblasDrotStridedBatched
#define hipblasCsrotStridedBatchedFortran hipblasCsrotStridedBatched
#define hipblasZdrotStridedBatchedFortran hipblasZdrotStridedBatched
#define hipblasCrotStridedBatchedFortran hipblasCrotStridedBatched
#define hipblasZrotStridedBatchedFortran hipblasZrotStridedBatched
#define hipblasSrotgFortran hipblasSrotg
#define hipblasDrotgFortran hipblasDrotg
#define hipblasCrotgFortran hipblasCrotg
#define hipblasZrotgFortran hipblasZrotg
#define hipblasSrotgBatchedFortran hipblasSrotgBatched
#define hipblasDrotgBatchedFortran hipblasDrotgBatched
#define hipblasCrotgBatchedFortran hipblasCrotgBatched
#define hipblasZrotgBatchedFortran hipblasZrotgBatched
#define hipblasSrotgStridedBatchedFortran hipblasSrotgStridedBatched
#define hipblasDrotgStridedBatchedFortran hipblasDrotgStridedBatched
#define hipblasCrotgStridedBatchedFortran hipblasCrotgStridedBatched
#define hipblasZrotgStridedBatchedFortran hipblasZrotgStridedBatched
#define hipblasSrotmFortran hipblasSrotm
#define hipblasDrotmFortran hipblasDrotm
#define hipblasSrotmBatchedFortran hipblasSrotmBatched
#define hipblasDrotmBatchedFortran hipblasDrotmBatched
#define hipblasSrotmStridedBatchedFortran hipblasSrotmStridedBatched
#define hipblasDrotmStridedBatchedFortran hipblasDrotmStridedBatched
#define hipblasSrotmgFortran hipblasSrotmg
#define hipblasDrotmgFortran hipblasDrotmg
#define hipblasSrotmgBatchedFortran hipblasSrotmgBatched
#define hipblasDrotmgBatchedFortran hipblasDrotmgBatched
#define hipblasSrotmgStridedBatchedFortran hipblasSrotmgStridedBatched
#define hipblasDrotmgStridedBatchedFortran hipblasDrotmgStridedBatched

/* ==========
 *    L2
 * ========== */

#define hipblasSgerFortran hipblasSger
#define hipblasDgerFortran hipblasDger
#define hipblasCgeruFortran hipblasCgeru
#define hipblasZgeruFortran hipblasZgeru
#define hipblasCgercFortran hipblasCgerc
#define hipblasZgercFortran hipblasZgerc
#define hipblasSgerBatchedFortran hipblasSgerBatched
#define hipblasDgerBatchedFortran hipblasDgerBatched
#define hipblasCgeruBatchedFortran hipblasCgeruBatched
#define hipblasZgeruBatchedFortran hipblasZgeruBatched
#define hipblasCgercBatchedFortran hipblasCgercBatched
#define hipblasZgercBatchedFortran hipblasZgercBatched
#define hipblasSgerStridedBatchedFortran hipblasSgerStridedBatched
#define hipblasDgerStridedBatchedFortran hipblasDgerStridedBatched
#define hipblasCgeruStridedBatchedFortran hipblasCgeruStridedBatched
#define hipblasZgeruStridedBatchedFortran hipblasZgeruStridedBatched
#define hipblasCgercStridedBatchedFortran hipblasCgercStridedBatched
#define hipblasZgercStridedBatchedFortran hipblasZgercStridedBatched
#define hipblasChbmvFortran hipblasChbmv
#define hipblasZhbmvFortran hipblasZhbmv
#define hipblasChbmvBatchedFortran hipblasChbmvBatched
#define hipblasZhbmvBatchedFortran hipblasZhbmvBatched
#define hipblasChbmvStridedBatchedFortran hipblasChbmvStridedBatched
#define hipblasZhbmvStridedBatchedFortran hipblasZhbmvStridedBatched
#define hipblasChemvFortran hipblasChemv
#define hipblasZhemvFortran hipblasZhemv
#define hipblasChemvBatchedFortran hipblasChemvBatched
#define hipblasZhemvBatchedFortran hipblasZhemvBatched
#define hipblasChemvStridedBatchedFortran hipblasChemvStridedBatched
#define hipblasZhemvStridedBatchedFortran hipblasZhemvStridedBatched
#define hipblasCherFortran hipblasCher
#define hipblasZherFortran hipblasZher
#define hipblasCherBatchedFortran hipblasCherBatched
#define hipblasZherBatchedFortran hipblasZherBatched
#define hipblasCherStridedBatchedFortran hipblasCherStridedBatched
#define hipblasZherStridedBatchedFortran hipblasZherStridedBatched
#define hipblasCher2Fortran hipblasCher2
#define hipblasZher2Fortran hipblasZher2
#define hipblasCher2BatchedFortran hipblasCher2Batched
#define hipblasZher2BatchedFortran hipblasZher2Batched
#define hipblasCher2StridedBatchedFortran hipblasCher2StridedBatched
#define hipblasZher2StridedBatchedFortran hipblasZher2StridedBatched
#define hipblasChpmvFortran hipblasChpmv
#define hipblasZhpmvFortran hipblasZhpmv
#define hipblasChpmvBatchedFortran hipblasChpmvBatched
#define hipblasZhpmvBatchedFortran hipblasZhpmvBatched
#define hipblasChpmvStridedBatchedFortran hipblasChpmvStridedBatched
#define hipblasZhpmvStridedBatchedFortran hipblasZhpmvStridedBatched
#define hipblasChprFortran hipblasChpr
#define hipblasZhprFortran hipblasZhpr
#define hipblasChprBatchedFortran hipblasChprBatched
#define hipblasZhprBatchedFortran hipblasZhprBatched
#define hipblasChprStridedBatchedFortran hipblasChprStridedBatched
#define hipblasZhprStridedBatchedFortran hipblasZhprStridedBatched
#define hipblasChpr2Fortran hipblasChpr2
#define hipblasZhpr2Fortran hipblasZhpr2
#define hipblasChpr2BatchedFortran hipblasChpr2Batched
#define hipblasZhpr2BatchedFortran hipblasZhpr2Batched
#define hipblasChpr2StridedBatchedFortran hipblasChpr2StridedBatched
#define hipblasZhpr2StridedBatchedFortran hipblasZhpr2StridedBatched
#define hipblasSsbmvFortran hipblasSsbmv
#define hipblasDsbmvFortran hipblasDsbmv
#define hipblasSsbmvBatchedFortran hipblasSsbmvBatched
#define hipblasDsbmvBatchedFortran hipblasDsbmvBatched
#define hipblasSsbmvStridedBatchedFortran hipblasSsbmvStridedBatched
#define hipblasDsbmvStridedBatchedFortran hipblasDsbmvStridedBatched
#define hipblasSspmvFortran hipblasSspmv
#define hipblasDspmvFortran hipblasDspmv
#define hipblasSspmvBatchedFortran hipblasSspmvBatched
#define hipblasDspmvBatchedFortran hipblasDspmvBatched
#define hipblasSspmvStridedBatchedFortran hipblasSspmvStridedBatched
#define hipblasDspmvStridedBatchedFortran hipblasDspmvStridedBatched
#define hipblasSsprFortran hipblasSspr
#define hipblasDsprFortran hipblasDspr
#define hipblasCsprFortran hipblasCspr
#define hipblasZsprFortran hipblasZspr
#define hipblasSsprBatchedFortran hipblasSsprBatched
#define hipblasDsprBatchedFortran hipblasDsprBatched
#define hipblasCsprBatchedFortran hipblasCsprBatched
#define hipblasZsprBatchedFortran hipblasZsprBatched
#define hipblasSsprStridedBatchedFortran hipblasSsprStridedBatched
#define hipblasDsprStridedBatchedFortran hipblasDsprStridedBatched
#define hipblasCsprStridedBatchedFortran hipblasCsprStridedBatched
#define hipblasZsprStridedBatchedFortran hipblasZsprStridedBatched
#define hipblasSspr2Fortran hipblasSspr2
#define hipblasDspr2Fortran hipblasDspr2
#define hipblasSspr2BatchedFortran hipblasSspr2Batched
#define hipblasDspr2BatchedFortran hipblasDspr2Batched
#define hipblasSspr2StridedBatchedFortran hipblasSspr2StridedBatched
#define hipblasDspr2StridedBatchedFortran hipblasDspr2StridedBatched
#define hipblasSsymvFortran hipblasSsymv
#define hipblasDsymvFortran hipblasDsymv
#define hipblasCsymvFortran hipblasCsymv
#define hipblasZsymvFortran hipblasZsymv
#define hipblasSsymvBatchedFortran hipblasSsymvBatched
#define hipblasDsymvBatchedFortran hipblasDsymvBatched
#define hipblasCsymvBatchedFortran hipblasCsymvBatched
#define hipblasZsymvBatchedFortran hipblasZsymvBatched
#define hipblasSsymvStridedBatchedFortran hipblasSsymvStridedBatched
#define hipblasDsymvStridedBatchedFortran hipblasDsymvStridedBatched
#define hipblasCsymvStridedBatchedFortran hipblasCsymvStridedBatched
#define hipblasZsymvStridedBatchedFortran hipblasZsymvStridedBatched
#define hipblasSsyrFortran hipblasSsyr
#define hipblasDsyrFortran hipblasDsyr
#define hipblasCsyrFortran hipblasCsyr
#define hipblasZsyrFortran hipblasZsyr
#define hipblasSsyrBatchedFortran hipblasSsyrBatched
#define hipblasDsyrBatchedFortran hipblasDsyrBatched
#define hipblasCsyrBatchedFortran hipblasCsyrBatched
#define hipblasZsyrBatchedFortran hipblasZsyrBatched
#define hipblasSsyrStridedBatchedFortran hipblasSsyrStridedBatched
#define hipblasDsyrStridedBatchedFortran hipblasDsyrStridedBatched
#define hipblasCsyrStridedBatchedFortran hipblasCsyrStridedBatched
#define hipblasZsyrStridedBatchedFortran hipblasZsyrStridedBatched
#define hipblasSsyr2Fortran hipblasSsyr2
#define hipblasDsyr2Fortran hipblasDsyr2
#define hipblasCsyr2Fortran hipblasCsyr2
#define hipblasZsyr2Fortran hipblasZsyr2
#define hipblasSsyr2BatchedFortran hipblasSsyr2Batched
#define hipblasDsyr2BatchedFortran hipblasDsyr2Batched
#define hipblasCsyr2BatchedFortran hipblasCsyr2Batched
#define hipblasZsyr2BatchedFortran hipblasZsyr2Batched
#define hipblasSsyr2StridedBatchedFortran hipblasSsyr2StridedBatched
#define hipblasDsyr2StridedBatchedFortran hipblasDsyr2StridedBatched
#define hipblasCsyr2StridedBatchedFortran hipblasCsyr2StridedBatched
#define hipblasZsyr2StridedBatchedFortran hipblasZsyr2StridedBatched
#define hipblasStbmvFortran hipblasStbmv
#define hipblasDtbmvFortran hipblasDtbmv
#define hipblasCtbmvFortran hipblasCtbmv
#define hipblasZtbmvFortran hipblasZtbmv
#define hipblasStbmvBatchedFortran hipblasStbmvBatched
#define hipblasDtbmvBatchedFortran hipblasDtbmvBatched
#define hipblasCtbmvBatchedFortran hipblasCtbmvBatched
#define hipblasZtbmvBatchedFortran hipblasZtbmvBatched
#define hipblasStbmvStridedBatchedFortran hipblasStbmvStridedBatched
#define hipblasDtbmvStridedBatchedFortran hipblasDtbmvStridedBatched
#define hipblasCtbmvStridedBatchedFortran hipblasCtbmvStridedBatched
#define hipblasZtbmvStridedBatchedFortran hipblasZtbmvStridedBatched
#define hipblasStbsvFortran hipblasStbsv
#define hipblasDtbsvFortran hipblasDtbsv
#define hipblasCtbsvFortran hipblasCtbsv
#define hipblasZtbsvFortran hipblasZtbsv
#define hipblasStbsvBatchedFortran hipblasStbsvBatched
#define hipblasDtbsvBatchedFortran hipblasDtbsvBatched
#define hipblasCtbsvBatchedFortran hipblasCtbsvBatched
#define hipblasZtbsvBatchedFortran hipblasZtbsvBatched
#define hipblasStbsvStridedBatchedFortran hipblasStbsvStridedBatched
#define hipblasDtbsvStridedBatchedFortran hipblasDtbsvStridedBatched
#define hipblasCtbsvStridedBatchedFortran hipblasCtbsvStridedBatched
#define hipblasZtbsvStridedBatchedFortran hipblasZtbsvStridedBatched
#define hipblasStpmvFortran hipblasStpmv
#define hipblasDtpmvFortran hipblasDtpmv
#define hipblasCtpmvFortran hipblasCtpmv
#define hipblasZtpmvFortran hipblasZtpmv
#define hipblasStpmvBatchedFortran hipblasStpmvBatched
#define hipblasDtpmvBatchedFortran hipblasDtpmvBatched
#define hipblasCtpmvBatchedFortran hipblasCtpmvBatched
#define hipblasZtpmvBatchedFortran hipblasZtpmvBatched
#define hipblasStpmvStridedBatchedFortran hipblasStpmvStridedBatched
#define hipblasDtpmvStridedBatchedFortran hipblasDtpmvStridedBatched
#define hipblasCtpmvStridedBatchedFortran hipblasCtpmvStridedBatched
#define hipblasZtpmvStridedBatchedFortran hipblasZtpmvStridedBatched
#define hipblasStpsvFortran hipblasStpsv
#define hipblasDtpsvFortran hipblasDtpsv
#define hipblasCtpsvFortran hipblasCtpsv
#define hipblasZtpsvFortran hipblasZtpsv
#define hipblasStpsvBatchedFortran hipblasStpsvBatched
#define hipblasDtpsvBatchedFortran hipblasDtpsvBatched
#define hipblasCtpsvBatchedFortran hipblasCtpsvBatched
#define hipblasZtpsvBatchedFortran hipblasZtpsvBatched
#define hipblasStpsvStridedBatchedFortran hipblasStpsvStridedBatched
#define hipblasDtpsvStridedBatchedFortran hipblasDtpsvStridedBatched
#define hipblasCtpsvStridedBatchedFortran hipblasCtpsvStridedBatched
#define hipblasZtpsvStridedBatchedFortran hipblasZtpsvStridedBatched
#define hipblasStrmvFortran hipblasStrmv
#define hipblasDtrmvFortran hipblasDtrmv
#define hipblasCtrmvFortran hipblasCtrmv
#define hipblasZtrmvFortran hipblasZtrmv
#define hipblasStrmvBatchedFortran hipblasStrmvBatched
#define hipblasDtrmvBatchedFortran hipblasDtrmvBatched
#define hipblasCtrmvBatchedFortran hipblasCtrmvBatched
#define hipblasZtrmvBatchedFortran hipblasZtrmvBatched
#define hipblasStrmvStridedBatchedFortran hipblasStrmvStridedBatched
#define hipblasDtrmvStridedBatchedFortran hipblasDtrmvStridedBatched
#define hipblasCtrmvStridedBatchedFortran hipblasCtrmvStridedBatched
#define hipblasZtrmvStridedBatchedFortran hipblasZtrmvStridedBatched
#define hipblasStrsvFortran hipblasStrsv
#define hipblasDtrsvFortran hipblasDtrsv
#define hipblasCtrsvFortran hipblasCtrsv
#define hipblasZtrsvFortran hipblasZtrsv
#define hipblasStrsvBatchedFortran hipblasStrsvBatched
#define hipblasDtrsvBatchedFortran hipblasDtrsvBatched
#define hipblasCtrsvBatchedFortran hipblasCtrsvBatched
#define hipblasZtrsvBatchedFortran hipblasZtrsvBatched
#define hipblasStrsvStridedBatchedFortran hipblasStrsvStridedBatched
#define hipblasDtrsvStridedBatchedFortran hipblasDtrsvStridedBatched
#define hipblasCtrsvStridedBatchedFortran hipblasCtrsvStridedBatched
#define hipblasZtrsvStridedBatchedFortran hipblasZtrsvStridedBatched
#define hipblasSgbmvFortran hipblasSgbmv
#define hipblasDgbmvFortran hipblasDgbmv
#define hipblasCgbmvFortran hipblasCgbmv
#define hipblasZgbmvFortran hipblasZgbmv
#define hipblasSgbmvBatchedFortran hipblasSgbmvBatched
#define hipblasDgbmvBatchedFortran hipblasDgbmvBatched
#define hipblasCgbmvBatchedFortran hipblasCgbmvBatched
#define hipblasZgbmvBatchedFortran hipblasZgbmvBatched
#define hipblasSgbmvStridedBatchedFortran hipblasSgbmvStridedBatched
#define hipblasDgbmvStridedBatchedFortran hipblasDgbmvStridedBatched
#define hipblasCgbmvStridedBatchedFortran hipblasCgbmvStridedBatched
#define hipblasZgbmvStridedBatchedFortran hipblasZgbmvStridedBatched
#define hipblasSgemvFortran hipblasSgemv
#define hipblasDgemvFortran hipblasDgemv
#define hipblasCgemvFortran hipblasCgemv
#define hipblasZgemvFortran hipblasZgemv
#define hipblasSgemvBatchedFortran hipblasSgemvBatched
#define hipblasDgemvBatchedFortran hipblasDgemvBatched
#define hipblasCgemvBatchedFortran hipblasCgemvBatched
#define hipblasZgemvBatchedFortran hipblasZgemvBatched
#define hipblasSgemvStridedBatchedFortran hipblasSgemvStridedBatched
#define hipblasDgemvStridedBatchedFortran hipblasDgemvStridedBatched
#define hipblasCgemvStridedBatchedFortran hipblasCgemvStridedBatched
#define hipblasZgemvStridedBatchedFortran hipblasZgemvStridedBatched

/* ==========
 *    L3
 * ========== */

#define hipblasCherkFortran hipblasCherk
#define hipblasZherkFortran hipblasZherk
#define hipblasCherkBatchedFortran hipblasCherkBatched
#define hipblasZherkBatchedFortran hipblasZherkBatched
#define hipblasCherkStridedBatchedFortran hipblasCherkStridedBatched
#define hipblasZherkStridedBatchedFortran hipblasZherkStridedBatched
#define hipblasCher2kFortran hipblasCher2k
#define hipblasZher2kFortran hipblasZher2k
#define hipblasCher2kBatchedFortran hipblasCher2kBatched
#define hipblasZher2kBatchedFortran hipblasZher2kBatched
#define hipblasCher2kStridedBatchedFortran hipblasCher2kStridedBatched
#define hipblasZher2kStridedBatchedFortran hipblasZher2kStridedBatched
#define hipblasCherkxFortran hipblasCherkx
#define hipblasZherkxFortran hipblasZherkx
#define hipblasCherkxBatchedFortran hipblasCherkxBatched
#define hipblasZherkxBatchedFortran hipblasZherkxBatched
#define hipblasCherkxStridedBatchedFortran hipblasCherkxStridedBatched
#define hipblasZherkxStridedBatchedFortran hipblasZherkxStridedBatched
#define hipblasSsymmFortran hipblasSsymm
#define hipblasDsymmFortran hipblasDsymm
#define hipblasCsymmFortran hipblasCsymm
#define hipblasZsymmFortran hipblasZsymm
#define hipblasSsymmBatchedFortran hipblasSsymmBatched
#define hipblasDsymmBatchedFortran hipblasDsymmBatched
#define hipblasCsymmBatchedFortran hipblasCsymmBatched
#define hipblasZsymmBatchedFortran hipblasZsymmBatched
#define hipblasSsymmStridedBatchedFortran hipblasSsymmStridedBatched
#define hipblasDsymmStridedBatchedFortran hipblasDsymmStridedBatched
#define hipblasCsymmStridedBatchedFortran hipblasCsymmStridedBatched
#define hipblasZsymmStridedBatchedFortran hipblasZsymmStridedBatched
#define hipblasSsyrkFortran hipblasSsyrk
#define hipblasDsyrkFortran hipblasDsyrk
#define hipblasCsyrkFortran hipblasCsyrk
#define hipblasZsyrkFortran hipblasZsyrk
#define hipblasSsyrkBatchedFortran hipblasSsyrkBatched
#define hipblasDsyrkBatchedFortran hipblasDsyrkBatched
#define hipblasCsyrkBatchedFortran hipblasCsyrkBatched
#define hipblasZsyrkBatchedFortran hipblasZsyrkBatched
#define hipblasSsyrkStridedBatchedFortran hipblasSsyrkStridedBatched
#define hipblasDsyrkStridedBatchedFortran hipblasDsyrkStridedBatched
#define hipblasCsyrkStridedBatchedFortran hipblasCsyrkStridedBatched
#define hipblasZsyrkStridedBatchedFortran hipblasZsyrkStridedBatched
#define hipblasSsyr2kFortran hipblasSsyr2k
#define hipblasDsyr2kFortran hipblasDsyr2k
#define hipblasCsyr2kFortran hipblasCsyr2k
#define hipblasZsyr2kFortran hipblasZsyr2k
#define hipblasSsyr2kBatchedFortran hipblasSsyr2kBatched
#define hipblasDsyr2kBatchedFortran hipblasDsyr2kBatched
#define hipblasCsyr2kBatchedFortran hipblasCsyr2kBatched
#define hipblasZsyr2kBatchedFortran hipblasZsyr2kBatched
#define hipblasSsyr2kStridedBatchedFortran hipblasSsyr2kStridedBatched
#define hipblasDsyr2kStridedBatchedFortran hipblasDsyr2kStridedBatched
#define hipblasCsyr2kStridedBatchedFortran hipblasCsyr2kStridedBatched
#define hipblasZsyr2kStridedBatchedFortran hipblasZsyr2kStridedBatched
#define hipblasSsyrkxFortran hipblasSsyrkx
#define hipblasDsyrkxFortran hipblasDsyrkx
#define hipblasCsyrkxFortran hipblasCsyrkx
#define hipblasZsyrkxFortran hipblasZsyrkx
#define hipblasSsyrkxBatchedFortran hipblasSsyrkxBatched
#define hipblasDsyrkxBatchedFortran hipblasDsyrkxBatched
#define hipblasCsyrkxBatchedFortran hipblasCsyrkxBatched
#define hipblasZsyrkxBatchedFortran hipblasZsyrkxBatched
#define hipblasSsyrkxStridedBatchedFortran hipblasSsyrkxStridedBatched
#define hipblasDsyrkxStridedBatchedFortran hipblasDsyrkxStridedBatched
#define hipblasCsyrkxStridedBatchedFortran hipblasCsyrkxStridedBatched
#define hipblasZsyrkxStridedBatchedFortran hipblasZsyrkxStridedBatched
#define hipblasSgeamFortran hipblasSgeam
#define hipblasDgeamFortran hipblasDgeam
#define hipblasCgeamFortran hipblasCgeam
#define hipblasZgeamFortran hipblasZgeam
#define hipblasSgeamBatchedFortran hipblasSgeamBatched
#define hipblasDgeamBatchedFortran hipblasDgeamBatched
#define hipblasCgeamBatchedFortran hipblasCgeamBatched
#define hipblasZgeamBatchedFortran hipblasZgeamBatched
#define hipblasSgeamStridedBatchedFortran hipblasSgeamStridedBatched
#define hipblasDgeamStridedBatchedFortran hipblasDgeamStridedBatched
#define hipblasCgeamStridedBatchedFortran hipblasCgeamStridedBatched
#define hipblasZgeamStridedBatchedFortran hipblasZgeamStridedBatched
#define hipblasChemmFortran hipblasChemm
#define hipblasZhemmFortran hipblasZhemm
#define hipblasChemmBatchedFortran hipblasChemmBatched
#define hipblasZhemmBatchedFortran hipblasZhemmBatched
#define hipblasChemmStridedBatchedFortran hipblasChemmStridedBatched
#define hipblasZhemmStridedBatchedFortran hipblasZhemmStridedBatched
#define hipblasStrmmFortran hipblasStrmm
#define hipblasDtrmmFortran hipblasDtrmm
#define hipblasCtrmmFortran hipblasCtrmm
#define hipblasZtrmmFortran hipblasZtrmm
#define hipblasStrmmBatchedFortran hipblasStrmmBatched
#define hipblasDtrmmBatchedFortran hipblasDtrmmBatched
#define hipblasCtrmmBatchedFortran hipblasCtrmmBatched
#define hipblasZtrmmBatchedFortran hipblasZtrmmBatched
#define hipblasStrmmStridedBatchedFortran hipblasStrmmStridedBatched
#define hipblasDtrmmStridedBatchedFortran hipblasDtrmmStridedBatched
#define hipblasCtrmmStridedBatchedFortran hipblasCtrmmStridedBatched
#define hipblasZtrmmStridedBatchedFortran hipblasZtrmmStridedBatched
#define hipblasStrtriFortran hipblasStrtri
#define hipblasDtrtriFortran hipblasDtrtri
#define hipblasCtrtriFortran hipblasCtrtri
#define hipblasZtrtriFortran hipblasZtrtri
#define hipblasStrtriBatchedFortran hipblasStrtriBatched
#define hipblasDtrtriBatchedFortran hipblasDtrtriBatched
#define hipblasCtrtriBatchedFortran hipblasCtrtriBatched
#define hipblasZtrtriBatchedFortran hipblasZtrtriBatched
#define hipblasStrtriStridedBatchedFortran hipblasStrtriStridedBatched
#define hipblasDtrtriStridedBatchedFortran hipblasDtrtriStridedBatched
#define hipblasCtrtriStridedBatchedFortran hipblasCtrtriStridedBatched
#define hipblasZtrtriStridedBatchedFortran hipblasZtrtriStridedBatched
#define hipblasSdgmmFortran hipblasSdgmm
#define hipblasDdgmmFortran hipblasDdgmm
#define hipblasCdgmmFortran hipblasCdgmm
#define hipblasZdgmmFortran hipblasZdgmm
#define hipblasSdgmmBatchedFortran hipblasSdgmmBatched
#define hipblasDdgmmBatchedFortran hipblasDdgmmBatched
#define hipblasCdgmmBatchedFortran hipblasCdgmmBatched
#define hipblasZdgmmBatchedFortran hipblasZdgmmBatched
#define hipblasSdgmmStridedBatchedFortran hipblasSdgmmStridedBatched
#define hipblasDdgmmStridedBatchedFortran hipblasDdgmmStridedBatched
#define hipblasCdgmmStridedBatchedFortran hipblasCdgmmStridedBatched
#define hipblasZdgmmStridedBatchedFortran hipblasZdgmmStridedBatched
#define hipblasStrsmFortran hipblasStrsm
#define hipblasDtrsmFortran hipblasDtrsm
#define hipblasCtrsmFortran hipblasCtrsm
#define hipblasZtrsmFortran hipblasZtrsm
#define hipblasStrsmBatchedFortran hipblasStrsmBatched
#define hipblasDtrsmBatchedFortran hipblasDtrsmBatched
#define hipblasCtrsmBatchedFortran hipblasCtrsmBatched
#define hipblasZtrsmBatchedFortran hipblasZtrsmBatched
#define hipblasStrsmStridedBatchedFortran hipblasStrsmStridedBatched
#define hipblasDtrsmStridedBatchedFortran hipblasDtrsmStridedBatched
#define hipblasCtrsmStridedBatchedFortran hipblasCtrsmStridedBatched
#define hipblasZtrsmStridedBatchedFortran hipblasZtrsmStridedBatched
#define hipblasHgemmFortran hipblasHgemm
#define hipblasSgemmFortran hipblasSgemm
#define hipblasDgemmFortran hipblasDgemm
#define hipblasCgemmFortran hipblasCgemm
#define hipblasZgemmFortran hipblasZgemm
#define hipblasHgemmBatchedFortran hipblasHgemmBatched
#define hipblasSgemmBatchedFortran hipblasSgemmBatched
#define hipblasDgemmBatchedFortran hipblasDgemmBatched
#define hipblasCgemmBatchedFortran hipblasCgemmBatched
#define hipblasZgemmBatchedFortran hipblasZgemmBatched
#define hipblasHgemmStridedBatchedFortran hipblasHgemmStridedBatched
#define hipblasSgemmStridedBatchedFortran hipblasSgemmStridedBatched
#define hipblasDgemmStridedBatchedFortran hipblasDgemmStridedBatched
#define hipblasCgemmStridedBatchedFortran hipblasCgemmStridedBatched
#define hipblasZgemmStridedBatchedFortran hipblasZgemmStridedBatched
#define hipblasGemmExFortran hipblasGemmEx
#define hipblasGemmBatchedExFortran hipblasGemmBatchedEx
#define hipblasGemmStridedBatchedExFortran hipblasGemmStridedBatchedEx
#define hipblasTrsmExFortran hipblasTrsmEx
#define hipblasTrsmBatchedExFortran hipblasTrsmBatchedEx
#define hipblasTrsmStridedBatchedExFortran hipblasTrsmStridedBatchedEx
#define hipblasAxpyExFortran hipblasAxpyEx
#define hipblasAxpyBatchedExFortran hipblasAxpyBatchedEx
#define hipblasAxpyStridedBatchedExFortran hipblasAxpyStridedBatchedEx
#define hipblasDotExFortran hipblasDotEx
#define hipblasDotBatchedExFortran hipblasDotBatchedEx
#define hipblasDotStridedBatchedExFortran hipblasDotStridedBatchedEx
#define hipblasDotcExFortran hipblasDotcEx
#define hipblasDotcBatchedExFortran hipblasDotcBatchedEx
#define hipblasDotcStridedBatchedExFortran hipblasDotcStridedBatchedEx
#define hipblasNrm2ExFortran hipblasNrm2Ex
#define hipblasNrm2BatchedExFortran hipblasNrm2BatchedEx
#define hipblasNrm2StridedBatchedExFortran hipblasNrm2StridedBatchedEx
#define hipblasRotExFortran hipblasRotEx
#define hipblasRotBatchedExFortran hipblasRotBatchedEx
#define hipblasRotStridedBatchedExFortran hipblasRotStridedBatchedEx
#define hipblasScalExFortran hipblasScalEx
#define hipblasScalBatchedExFortran hipblasScalBatchedEx
#define hipblasScalStridedBatchedExFortran hipblasScalStridedBatchedEx
#if 0
extern "C" {
/* ==========
 *    Solver
 * ========== */

// getrf
hipblasStatus_t hipblasSgetrfFortran(
    hipblasHandle_t handle, const int n, float* A, const int lda, int* ipiv, int* info);

hipblasStatus_t hipblasDgetrfFortran(
    hipblasHandle_t handle, const int n, double* A, const int lda, int* ipiv, int* info);

hipblasStatus_t hipblasCgetrfFortran(
    hipblasHandle_t handle, const int n, hipblasComplex* A, const int lda, int* ipiv, int* info);

hipblasStatus_t hipblasZgetrfFortran(hipblasHandle_t       handle,
                                     const int             n,
                                     hipblasDoubleComplex* A,
                                     const int             lda,
                                     int*                  ipiv,
                                     int*                  info);

// getrf_batched
hipblasStatus_t hipblasSgetrfBatchedFortran(hipblasHandle_t handle,
                                            const int       n,
                                            float* const    A[],
                                            const int       lda,
                                            int*            ipiv,
                                            int*            info,
                                            const int       batch_count);

hipblasStatus_t hipblasDgetrfBatchedFortran(hipblasHandle_t handle,
                                            const int       n,
                                            double* const   A[],
                                            const int       lda,
                                            int*            ipiv,
                                            int*            info,
                                            const int       batch_count);

hipblasStatus_t hipblasCgetrfBatchedFortran(hipblasHandle_t       handle,
                                            const int             n,
                                            hipblasComplex* const A[],
                                            const int             lda,
                                            int*                  ipiv,
                                            int*                  info,
                                            const int             batch_count);

hipblasStatus_t hipblasZgetrfBatchedFortran(hipblasHandle_t             handle,
                                            const int                   n,
                                            hipblasDoubleComplex* const A[],
                                            const int                   lda,
                                            int*                        ipiv,
                                            int*                        info,
                                            const int                   batch_count);

// getrf_strided_batched
hipblasStatus_t hipblasSgetrfStridedBatchedFortran(hipblasHandle_t     handle,
                                                   const int           n,
                                                   float*              A,
                                                   const int           lda,
                                                   const hipblasStride stride_A,
                                                   int*                ipiv,
                                                   const hipblasStride stride_P,
                                                   int*                info,
                                                   const int           batch_count);

hipblasStatus_t hipblasDgetrfStridedBatchedFortran(hipblasHandle_t     handle,
                                                   const int           n,
                                                   double*             A,
                                                   const int           lda,
                                                   const hipblasStride stride_A,
                                                   int*                ipiv,
                                                   const hipblasStride stride_P,
                                                   int*                info,
                                                   const int           batch_count);

hipblasStatus_t hipblasCgetrfStridedBatchedFortran(hipblasHandle_t     handle,
                                                   const int           n,
                                                   hipblasComplex*     A,
                                                   const int           lda,
                                                   const hipblasStride stride_A,
                                                   int*                ipiv,
                                                   const hipblasStride stride_P,
                                                   int*                info,
                                                   const int           batch_count);

hipblasStatus_t hipblasZgetrfStridedBatchedFortran(hipblasHandle_t       handle,
                                                   const int             n,
                                                   hipblasDoubleComplex* A,
                                                   const int             lda,
                                                   const hipblasStride   stride_A,
                                                   int*                  ipiv,
                                                   const hipblasStride   stride_P,
                                                   int*                  info,
                                                   const int             batch_count);

// getrs
hipblasStatus_t hipblasSgetrsFortran(hipblasHandle_t          handle,
                                     const hipblasOperation_t trans,
                                     const int                n,
                                     const int                nrhs,
                                     float*                   A,
                                     const int                lda,
                                     const int*               ipiv,
                                     float*                   B,
                                     const int                ldb,
                                     int*                     info);

hipblasStatus_t hipblasDgetrsFortran(hipblasHandle_t          handle,
                                     const hipblasOperation_t trans,
                                     const int                n,
                                     const int                nrhs,
                                     double*                  A,
                                     const int                lda,
                                     const int*               ipiv,
                                     double*                  B,
                                     const int                ldb,
                                     int*                     info);

hipblasStatus_t hipblasCgetrsFortran(hipblasHandle_t          handle,
                                     const hipblasOperation_t trans,
                                     const int                n,
                                     const int                nrhs,
                                     hipblasComplex*          A,
                                     const int                lda,
                                     const int*               ipiv,
                                     hipblasComplex*          B,
                                     const int                ldb,
                                     int*                     info);

hipblasStatus_t hipblasZgetrsFortran(hipblasHandle_t          handle,
                                     const hipblasOperation_t trans,
                                     const int                n,
                                     const int                nrhs,
                                     hipblasDoubleComplex*    A,
                                     const int                lda,
                                     const int*               ipiv,
                                     hipblasDoubleComplex*    B,
                                     const int                ldb,
                                     int*                     info);

// getrs_batched
hipblasStatus_t hipblasSgetrsBatchedFortran(hipblasHandle_t          handle,
                                            const hipblasOperation_t trans,
                                            const int                n,
                                            const int                nrhs,
                                            float* const             A[],
                                            const int                lda,
                                            const int*               ipiv,
                                            float* const             B[],
                                            const int                ldb,
                                            int*                     info,
                                            const int                batch_count);

hipblasStatus_t hipblasDgetrsBatchedFortran(hipblasHandle_t          handle,
                                            const hipblasOperation_t trans,
                                            const int                n,
                                            const int                nrhs,
                                            double* const            A[],
                                            const int                lda,
                                            const int*               ipiv,
                                            double* const            B[],
                                            const int                ldb,
                                            int*                     info,
                                            const int                batch_count);

hipblasStatus_t hipblasCgetrsBatchedFortran(hipblasHandle_t          handle,
                                            const hipblasOperation_t trans,
                                            const int                n,
                                            const int                nrhs,
                                            hipblasComplex* const    A[],
                                            const int                lda,
                                            const int*               ipiv,
                                            hipblasComplex* const    B[],
                                            const int                ldb,
                                            int*                     info,
                                            const int                batch_count);

hipblasStatus_t hipblasZgetrsBatchedFortran(hipblasHandle_t             handle,
                                            const hipblasOperation_t    trans,
                                            const int                   n,
                                            const int                   nrhs,
                                            hipblasDoubleComplex* const A[],
                                            const int                   lda,
                                            const int*                  ipiv,
                                            hipblasDoubleComplex* const B[],
                                            const int                   ldb,
                                            int*                        info,
                                            const int                   batch_count);

// getrs_strided_batched
hipblasStatus_t hipblasSgetrsStridedBatchedFortran(hipblasHandle_t          handle,
                                                   const hipblasOperation_t trans,
                                                   const int                n,
                                                   const int                nrhs,
                                                   float*                   A,
                                                   const int                lda,
                                                   const hipblasStride      stride_A,
                                                   const int*               ipiv,
                                                   const hipblasStride      stride_P,
                                                   float*                   B,
                                                   const int                ldb,
                                                   const hipblasStride      stride_B,
                                                   int*                     info,
                                                   const int                batch_count);

hipblasStatus_t hipblasDgetrsStridedBatchedFortran(hipblasHandle_t          handle,
                                                   const hipblasOperation_t trans,
                                                   const int                n,
                                                   const int                nrhs,
                                                   double*                  A,
                                                   const int                lda,
                                                   const hipblasStride      stride_A,
                                                   const int*               ipiv,
                                                   const hipblasStride      stride_P,
                                                   double*                  B,
                                                   const int                ldb,
                                                   const hipblasStride      stride_B,
                                                   int*                     info,
                                                   const int                batch_count);

hipblasStatus_t hipblasCgetrsStridedBatchedFortran(hipblasHandle_t          handle,
                                                   const hipblasOperation_t trans,
                                                   const int                n,
                                                   const int                nrhs,
                                                   hipblasComplex*          A,
                                                   const int                lda,
                                                   const hipblasStride      stride_A,
                                                   const int*               ipiv,
                                                   const hipblasStride      stride_P,
                                                   hipblasComplex*          B,
                                                   const int                ldb,
                                                   const hipblasStride      stride_B,
                                                   int*                     info,
                                                   const int                batch_count);

hipblasStatus_t hipblasZgetrsStridedBatchedFortran(hipblasHandle_t          handle,
                                                   const hipblasOperation_t trans,
                                                   const int                n,
                                                   const int                nrhs,
                                                   hipblasDoubleComplex*    A,
                                                   const int                lda,
                                                   const hipblasStride      stride_A,
                                                   const int*               ipiv,
                                                   const hipblasStride      stride_P,
                                                   hipblasDoubleComplex*    B,
                                                   const int                ldb,
                                                   const hipblasStride      stride_B,
                                                   int*                     info,
                                                   const int                batch_count);

// getri_batched
hipblasStatus_t hipblasSgetriBatchedFortran(hipblasHandle_t handle,
                                            const int       n,
                                            float* const    A[],
                                            const int       lda,
                                            int*            ipiv,
                                            float* const    C[],
                                            const int       ldc,
                                            int*            info,
                                            const int       batch_count);

hipblasStatus_t hipblasDgetriBatchedFortran(hipblasHandle_t handle,
                                            const int       n,
                                            double* const   A[],
                                            const int       lda,
                                            int*            ipiv,
                                            double* const   C[],
                                            const int       ldc,
                                            int*            info,
                                            const int       batch_count);

hipblasStatus_t hipblasCgetriBatchedFortran(hipblasHandle_t       handle,
                                            const int             n,
                                            hipblasComplex* const A[],
                                            const int             lda,
                                            int*                  ipiv,
                                            hipblasComplex* const C[],
                                            const int             ldc,
                                            int*                  info,
                                            const int             batch_count);

hipblasStatus_t hipblasZgetriBatchedFortran(hipblasHandle_t             handle,
                                            const int                   n,
                                            hipblasDoubleComplex* const A[],
                                            const int                   lda,
                                            int*                        ipiv,
                                            hipblasDoubleComplex* const C[],
                                            const int                   ldc,
                                            int*                        info,
                                            const int                   batch_count);

// geqrf
hipblasStatus_t hipblasSgeqrfFortran(hipblasHandle_t handle,
                                     const int       m,
                                     const int       n,
                                     float*          A,
                                     const int       lda,
                                     float*          tau,
                                     int*            info);

hipblasStatus_t hipblasDgeqrfFortran(hipblasHandle_t handle,
                                     const int       m,
                                     const int       n,
                                     double*         A,
                                     const int       lda,
                                     double*         tau,
                                     int*            info);

hipblasStatus_t hipblasCgeqrfFortran(hipblasHandle_t handle,
                                     const int       m,
                                     const int       n,
                                     hipblasComplex* A,
                                     const int       lda,
                                     hipblasComplex* tau,
                                     int*            info);

hipblasStatus_t hipblasZgeqrfFortran(hipblasHandle_t       handle,
                                     const int             m,
                                     const int             n,
                                     hipblasDoubleComplex* A,
                                     const int             lda,
                                     hipblasDoubleComplex* tau,
                                     int*                  info);

// geqrf_batched
hipblasStatus_t hipblasSgeqrfBatchedFortran(hipblasHandle_t handle,
                                            const int       m,
                                            const int       n,
                                            float* const    A[],
                                            const int       lda,
                                            float* const    tau[],
                                            int*            info,
                                            const int       batch_count);

hipblasStatus_t hipblasDgeqrfBatchedFortran(hipblasHandle_t handle,
                                            const int       m,
                                            const int       n,
                                            double* const   A[],
                                            const int       lda,
                                            double* const   tau[],
                                            int*            info,
                                            const int       batch_count);

hipblasStatus_t hipblasCgeqrfBatchedFortran(hipblasHandle_t       handle,
                                            const int             m,
                                            const int             n,
                                            hipblasComplex* const A[],
                                            const int             lda,
                                            hipblasComplex* const tau[],
                                            int*                  info,
                                            const int             batch_count);

hipblasStatus_t hipblasZgeqrfBatchedFortran(hipblasHandle_t             handle,
                                            const int                   m,
                                            const int                   n,
                                            hipblasDoubleComplex* const A[],
                                            const int                   lda,
                                            hipblasDoubleComplex* const tau[],
                                            int*                        info,
                                            const int                   batch_count);

// geqrf_strided_batched
hipblasStatus_t hipblasSgeqrfStridedBatchedFortran(hipblasHandle_t     handle,
                                                   const int           m,
                                                   const int           n,
                                                   float*              A,
                                                   const int           lda,
                                                   const hipblasStride stride_A,
                                                   float*              tau,
                                                   const hipblasStride stride_T,
                                                   int*                info,
                                                   const int           batch_count);

hipblasStatus_t hipblasDgeqrfStridedBatchedFortran(hipblasHandle_t     handle,
                                                   const int           m,
                                                   const int           n,
                                                   double*             A,
                                                   const int           lda,
                                                   const hipblasStride stride_A,
                                                   double*             tau,
                                                   const hipblasStride stride_T,
                                                   int*                info,
                                                   const int           batch_count);

hipblasStatus_t hipblasCgeqrfStridedBatchedFortran(hipblasHandle_t     handle,
                                                   const int           m,
                                                   const int           n,
                                                   hipblasComplex*     A,
                                                   const int           lda,
                                                   const hipblasStride stride_A,
                                                   hipblasComplex*     tau,
                                                   const hipblasStride stride_T,
                                                   int*                info,
                                                   const int           batch_count);

hipblasStatus_t hipblasZgeqrfStridedBatchedFortran(hipblasHandle_t       handle,
                                                   const int             m,
                                                   const int             n,
                                                   hipblasDoubleComplex* A,
                                                   const int             lda,
                                                   const hipblasStride   stride_A,
                                                   hipblasDoubleComplex* tau,
                                                   const hipblasStride   stride_T,
                                                   int*                  info,
                                                   const int             batch_count);
}
#endif

#endif
