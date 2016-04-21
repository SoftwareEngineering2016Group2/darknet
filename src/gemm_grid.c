
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

// #include "gemm_utils.h"
#include "gemm_grid.h"

#ifdef SDS
#include "sds_lib.h"
#define MALLOC sds_alloc
#define FREE sds_free
#else
#define MALLOC malloc
#define FREE free
#endif

int get_blocked_width(int orig, int blk_width) {
	return (int) ceil((double)orig/blk_width)*blk_width;
}

float *trans_to_blocked(float *A, int m, int n, int lda, int blk_m, int blk_n) {

	// m_align and n_align decides the matrix size
	int m_align = (int) ceil((double)m/blk_m)*blk_m;
	int n_align = (int) ceil((double)n/blk_n)*blk_n;

	int blk_size = blk_m*blk_n;
	int blk_per_m = m_align/blk_m;
	int blk_per_n = n_align/blk_n;

	float *A_block = (float *) MALLOC(sizeof(float)*m_align*n_align);

	int i, j, x, y;
	for (i = 0; i < m_align; i += blk_m) {
		for (j = 0; j < n_align; j += blk_n) {
			int blk_i = i/blk_m;
			int blk_j = j/blk_n;
			int blk_id = blk_i*blk_per_n+blk_j;
			// inside block
			for (x = 0; x < blk_m; x ++) {
				for (y = 0; y < blk_n; y ++) {
					int dst_idx = blk_id*blk_size+x*blk_n+y;
					int src_idx = (i+x)*n+j+y;
					A_block[dst_idx] = 
						(x + i >= m || y + j >= n) ? 0.0 : A[src_idx];
				}
			}
		}
	}

	return A_block;
}

void trans_from_blocked(float *A, float *A_block, int M, int N, int blk_m, int blk_n) {
	int i, j;

	int M_align = get_blocked_width(M, blk_m);
	int N_align = get_blocked_width(N, blk_n);
	
	int blk_size = blk_m*blk_n;
	int num_blk = M_align*N_align/blk_size;
	int num_blk_m = M_align/blk_m;
	int num_blk_n = N_align/blk_n;

	for (i = 0; i < M_align*N_align; i += blk_size) {
		int blk_id = i/blk_size;
		int blk_id_m = blk_id/num_blk_n;
		int blk_id_n = blk_id%num_blk_n;

		for (j = 0; j < blk_size; j++) {
			int x = blk_id_m*blk_m+j/blk_n;
			int y = blk_id_n*blk_n+j%blk_n;
			if (x < M && y < N)
				A[x*N+y] = A_block[i+j];
		}
	}
}

/* synthesis on hardware */
void gemm_grid_nn_blk(float ALPHA, 
	float A_BLK[BLK_M*BLK_K], int BLK_LDA, 
	float B_BLK[BLK_K*BLK_N], int BLK_LDB,
	float C_BLK[BLK_M*BLK_N], int BLK_LDC)
{
	int i, j, k;
	for (i = 0; i < BLK_M; i++) {
		for (j = 0; j < BLK_N; j++) {
			float sum = 0.0;
			for (k = 0; k < BLK_K; k++) {
				sum += ALPHA*A_BLK[i*BLK_LDA+k]*B_BLK[k*BLK_LDB+j];
			}
			// here is +=
			C_BLK[i*BLK_LDC+j] = sum;
		}
	}
}

void gemm_grid_madd_blk(float A[BLK_M*BLK_N], float B[BLK_M*BLK_N], float C[BLK_M*BLK_N]) {
	int i;
	for (i = 0; i < BLK_M*BLK_N; i++)
		C[i] = A[i] + B[i];
}

void gemm_grid_nn(int M, int N, int K, float ALPHA, 
	float *A, int lda,
	float *B, int ldb,
	float *C, int ldc)
{
	int blk_i, blk_j, blk_k;
	int i;
	
	int num_blk_m = M/BLK_M;
	int num_blk_n = N/BLK_N;
	int num_blk_k = K/BLK_K;

	int blk_size = BLK_M*BLK_N;

	for (blk_i = 0; blk_i < num_blk_m; blk_i++) {
		for (blk_j = 0; blk_j < num_blk_n; blk_j++) {
			int blk_id = blk_i*num_blk_n+blk_j;

			float tmp1_BLK[BLK_M*BLK_N]; // for C
			float tmp2_BLK[BLK_M*BLK_N]; // for C
			float *C_BLK = C+blk_id*blk_size;
			for (blk_k = 0; blk_k < num_blk_k; blk_k++) {
				int blk_id_A = blk_i*num_blk_k+blk_k;
				int blk_id_B = blk_k*num_blk_n+blk_j;
				// printf("id_A=%d id_B=%d id_C=%d\n", blk_id_A, blk_id_B, blk_id);
				float *A_BLK = A+blk_id_A*blk_size;
				float *B_BLK = B+blk_id_B*blk_size;

				// call matrix ops
				gemm_grid_nn_blk(ALPHA,A_BLK,BLK_K,B_BLK,BLK_N,tmp1_BLK,BLK_N);
				gemm_grid_madd_blk(tmp1_BLK,C_BLK,tmp2_BLK);

				memcpy(C_BLK, tmp2_BLK, sizeof(float)*blk_size);
			}
		}
	}
}

void gemm_grid(int TA, int TB, int M, int N, int K, float ALPHA,
	float *A, int lda,
	float *B, int ldb,
	float BETA,
	float *C, int ldc)
{
	float *A_block = trans_to_blocked(A,M,K,lda,BLK_M,BLK_K);
	float *B_block = trans_to_blocked(B,K,N,ldb,BLK_K,BLK_N);
	float *C_block = trans_to_blocked(C,M,N,ldc,BLK_M,BLK_N);

	// print_blocked_matrix(A_block,M,K,BLK_M,BLK_K);
	// print_blocked_matrix(B_block,K,N,BLK_K,BLK_N);

	int M_align = get_blocked_width(M, BLK_M);
	int N_align = get_blocked_width(N, BLK_N);
	int K_align = get_blocked_width(K, BLK_K);

	int i;
	for (i = 0; i < (M_align*N_align); i++)
		C_block[i] *= BETA;

	if (!TA && !TB)
		gemm_grid_nn(M_align,N_align,K_align,ALPHA,A_block,lda,B_block,ldb,C_block,ldc);
	else 
		; // TODO

	// print_blocked_matrix(C_block,M,N,BLK_M,BLK_N);
	trans_from_blocked(C,C_block,M,N,BLK_M,BLK_N);

	FREE(A_block);
	FREE(B_block);
	FREE(C_block);
}
