#ifndef GEMM_BLOCK_UNIT_H__
#define GEMM_BLOCK_UNIT_H__

#ifdef __SDSCC__
#include <stdlib.h>
#include "sds_lib.h"
#endif
/* constants are defined here */
#include "gemm_consts.h"

void gemm_block_units_mmult(
        float A[NUM_DEPTH*NUM_PIPES*BLK_M*BLK_K],
        float B[NUM_DEPTH*NUM_PIPES*BLK_K*BLK_N],
        float T[NUM_DEPTH*NUM_PIPES*BLK_M*BLK_N]);

void gemm_block_units_mplus(
        float T[NUM_DEPTH*NUM_PIPES*BLK_M*BLK_N],
        float C[NUM_DEPTH*BLK_M*BLK_N],
        float R[NUM_DEPTH*BLK_M*BLK_N]);

#endif
