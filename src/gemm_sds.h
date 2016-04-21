#ifndef GEMM_SDS_H__
#define GEMM_SDS_H__

#include "gemm_types.h"

void gemm_sds(int TA, int TB, int M, int N, int K,
    float ALPHA,
    float *A, int lda,
    float *B, int ldb,
    float BETA,
    float *C, int ldc);

#endif
