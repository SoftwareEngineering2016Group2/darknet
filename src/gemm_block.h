#ifndef GEMM_BLOCK_H__
#define GEMM_BLOCK_H__

#include "gemm_types.h"

void gemm_block_main(float ALPHA, float BETA, BlockedMatrix *A_blk, BlockedMatrix *B_blk, BlockedMatrix *C_blk);
#endif
