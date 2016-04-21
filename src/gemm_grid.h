#ifndef GEMM_GRID_H__
#define GEMM_GRID_H__

#define BLK_M 32
#define BLK_N 32
#define BLK_K 32

void gemm_grid(int TA, int TB, int M, int N, int K, float ALPHA,
	float *A, int lda,
	float *B, int ldb,
	float BETA,
	float *C, int ldc);

float *trans_to_blocked(float *A, int m, int n, int lda, int blk_m, int blk_n);

#endif