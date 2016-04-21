
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __SDSCC__
#include "sds_lib.h"
#endif

#include "gemm_sds.h"

#include "gemm_utils.h"
#include "gemm_consts.h"
#include "gemm_types.h"
#include "gemm_trans.h"
#include "gemm_block.h"

void gemm_sds(int TA, int TB, int M, int N, int K,
    float ALPHA,
    float *A, int lda,
    float *B, int ldb,
    float BETA,
    float *C, int ldc)
{
  BlockedMatrix *A_blk = flatten_matrix_to_blocked(TA, A, M, K, lda, BLK_M, BLK_K);
  BlockedMatrix *B_blk = flatten_matrix_to_blocked(TB, B, K, N, ldb, BLK_K, BLK_N);
  BlockedMatrix *C_blk = flatten_matrix_to_blocked(0,  C, M, N, ldc, BLK_M, BLK_N);

  gemm_block_main(ALPHA,BETA,A_blk,B_blk,C_blk);

  blocked_matrix_to_flatten(C_blk, C);
}
