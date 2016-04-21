#include <stdio.h>
#include <string.h>

#include "gemm_types.h"
#include "gemm_utils.h"
#include "gemm_block_unit.h"

inline void gemm_block_calc_A(
    float ALPHA, float BETA, 
    BlockedMatrix *A_blk, 
    BlockedMatrix *B_blk, 
    BlockedMatrix *C_blk, 
    float *A_buf, float *B_buf, float *T_buf, 
    float *C_buf, float *R_buf); 

inline void gemm_block_calc_B(
    float ALPHA, float BETA, 
    BlockedMatrix *A_blk, 
    BlockedMatrix *B_blk, 
    BlockedMatrix *C_blk, 
    float *A_buf, float *B_buf, float *T_buf, 
    float *C_buf, float *R_buf);

void gemm_block_main(float ALPHA, float BETA, BlockedMatrix *A_blk, BlockedMatrix *B_blk, BlockedMatrix *C_blk)
{
#if BUFTYPE == 0
  static float A_buf[NUM_DEPTH*PIPE_SIZE_MK];
  static float B_buf[NUM_DEPTH*PIPE_SIZE_KN];
  static float T_buf[NUM_DEPTH*PIPE_SIZE_MN];
  static float C_buf[NUM_DEPTH*BLK_SIZE_MN];
  static float R_buf[NUM_DEPTH*BLK_SIZE_MN];
#elif BUFTYPE == 1
  float* A_buf = MALLOC(sizeof(float)*NUM_DEPTH*PIPE_SIZE_MK);
  float* B_buf = MALLOC(sizeof(float)*NUM_DEPTH*PIPE_SIZE_KN);
  float* T_buf = MALLOC(sizeof(float)*NUM_DEPTH*PIPE_SIZE_MN);
  float* C_buf = MALLOC(sizeof(float)*NUM_DEPTH*BLK_SIZE_MN);
  float* R_buf = MALLOC(sizeof(float)*NUM_DEPTH*BLK_SIZE_MN);

  if (!A_buf || !B_buf || !C_buf || !T_buf || !R_buf) {
    fprintf(stderr, "Error buffer allocation\n");
    exit(1);
  }
#endif

  memset(A_buf, 0, sizeof(float)*NUM_DEPTH*PIPE_SIZE_MK);
  memset(B_buf, 0, sizeof(float)*NUM_DEPTH*PIPE_SIZE_KN);
  memset(T_buf, 0, sizeof(float)*NUM_DEPTH*PIPE_SIZE_MN);
  memset(C_buf, 0, sizeof(float)*NUM_DEPTH*BLK_SIZE_MN);
  memset(R_buf, 0, sizeof(float)*NUM_DEPTH*BLK_SIZE_MN);
  // main calculation part
#ifdef AMAJOR
  gemm_block_calc_A(ALPHA,BETA,A_blk,B_blk,C_blk,A_buf,B_buf,T_buf,C_buf,R_buf);
#else
  gemm_block_calc_B(ALPHA,BETA,A_blk,B_blk,C_blk,A_buf,B_buf,T_buf,C_buf,R_buf);
#endif

#if BUFTYPE == 1
  free(A_buf);
  free(B_buf);
  free(T_buf);
  free(C_buf);
  free(R_buf);
#endif
}

inline void gemm_block_calc_A(
    float ALPHA, float BETA, 
    BlockedMatrix *A_blk, 
    BlockedMatrix *B_blk, 
    BlockedMatrix *C_blk, 
    float *A_buf, float *B_buf, float *T_buf, 
    float *C_buf, float *R_buf) 
{
  int num_blk_M=A_blk->H/BLK_M;
  int num_blk_N=B_blk->W/BLK_N;
  int num_blk_K=A_blk->W/BLK_K;

  int i, j, k, d, p, t;
  int num_depth, num_pipes;

  for (j = 0; j < num_blk_N; j ++) {
    for (i = 0; i < num_blk_M; i += NUM_DEPTH) {
      num_depth = ((i+NUM_DEPTH) <= num_blk_M) ? NUM_DEPTH : num_blk_M-i;
      // initialize C_buf
      for (d = 0; d < num_depth; d ++)
        memcpy(&C_buf[d*BLK_SIZE_MN], 
               &C_blk->mat[((i+d)*num_blk_N+j)*BLK_SIZE_MN],
               sizeof(float)*BLK_SIZE_MN);
      for (t = 0; t < NUM_DEPTH*BLK_SIZE_MN; t ++)
        C_buf[t] *= BETA;

      // Here NUM_DEPTH number of C blocks will be computed
      for (k = 0; k < num_blk_K; k += NUM_PIPES) {
        num_pipes = ((k+NUM_PIPES) <= num_blk_K) ? NUM_PIPES : num_blk_K-k;
        // initialize A_buf
        for (d = 0; d < num_depth; d ++)
          memcpy(&A_buf[d*PIPE_SIZE_MK],
                 &A_blk->mat[((i+d)*num_blk_K+k)*BLK_SIZE_MK],
                 sizeof(float)*PIPE_SIZE_MK);
        for (t = 0; t < NUM_DEPTH*PIPE_SIZE_MK; t ++)
          A_buf[t] *= ALPHA;
        
        // initialize B_buf
        for (p = 0; p < num_pipes; p ++) 
          for (d = 0; d < num_depth; d ++) 
            memcpy(&B_buf[d*PIPE_SIZE_KN+p*BLK_SIZE_KN], 
                   &B_blk->mat[((k+p)*num_blk_N+j)*BLK_SIZE_KN],
                   sizeof(float)*BLK_SIZE_KN);

        // compute
        gemm_block_units_mmult(A_buf,B_buf,T_buf);
        gemm_block_units_mplus(T_buf,C_buf,R_buf);

        // copy R_buf to C_buf
        memcpy(C_buf, R_buf, sizeof(float)*NUM_DEPTH*BLK_SIZE_MN);
      }

      // final write back
      for (d = 0; d < num_depth; d ++)
        memcpy(&C_blk->mat[((i+d)*num_blk_N+j)*BLK_SIZE_MN],
               &C_buf[d*BLK_SIZE_MN],
               sizeof(float)*BLK_SIZE_MN);
    }
  }
}

inline void gemm_block_calc_B(
    float ALPHA, float BETA, 
    BlockedMatrix *A_blk, 
    BlockedMatrix *B_blk, 
    BlockedMatrix *C_blk, 
    float *A_buf, float *B_buf, float *T_buf, 
    float *C_buf, float *R_buf) 
{
  int num_blk_M=A_blk->H/BLK_M;
  int num_blk_N=B_blk->W/BLK_N;
  int num_blk_K=A_blk->W/BLK_K;

  int i, j, k, d, p, t;
  int num_depth, num_pipes;
  for (i = 0; i < num_blk_M; i ++) {
    for (j = 0; j < num_blk_N; j += NUM_DEPTH) {
      num_depth = ((j+NUM_DEPTH) <= num_blk_N) ? NUM_DEPTH : num_blk_N-j;

      // initialize C_buf
      memcpy(C_buf, 
             &C_blk->mat[(i*num_blk_N+j)*BLK_SIZE_MN],
             sizeof(float)*NUM_DEPTH*BLK_SIZE_MN);
      for (t = 0; t < NUM_DEPTH*BLK_SIZE_MN; t ++)
        C_buf[t] *= BETA;

      // Here NUM_DEPTH number of C blocks will be computed
      for (k = 0; k < num_blk_K; k += NUM_PIPES) {
        memset(A_buf, 0, sizeof(float)*NUM_DEPTH*PIPE_SIZE_MK);
        memset(B_buf, 0, sizeof(float)*NUM_DEPTH*PIPE_SIZE_KN);
        memset(T_buf, 0, sizeof(float)*NUM_DEPTH*BLK_SIZE_MN);
        memset(R_buf, 0, sizeof(float)*NUM_DEPTH*BLK_SIZE_MN);

        num_pipes = ((k+NUM_PIPES) <= num_blk_K) ? NUM_PIPES : num_blk_K-k;
        // printf("(%3d %3d %3d) depth %3d pipes %3d\n", i, j, k, num_depth, num_pipes);
        // initialize A_buf
        for (d = 0; d < num_depth; d ++)
          memcpy(&A_buf[d*PIPE_SIZE_MK],
                 &A_blk->mat[(i*num_blk_K+k)*BLK_SIZE_MK],
                 sizeof(float)*num_pipes*BLK_SIZE_MK);
        for (t = 0; t < NUM_DEPTH*PIPE_SIZE_MK; t ++)
          A_buf[t] *= ALPHA;
        
        // initialize B_buf
        for (d = 0; d < num_depth; d ++) 
          for (p = 0; p < num_pipes; p ++) 
            memcpy(&B_buf[d*PIPE_SIZE_KN+p*BLK_SIZE_KN], 
                   &B_blk->mat[((k+p)*num_blk_N+(j+d))*BLK_SIZE_KN],
                   sizeof(float)*BLK_SIZE_KN);

        // compute
        gemm_block_units_mmult(A_buf,B_buf,T_buf);
        gemm_block_units_mplus(T_buf,C_buf,R_buf);

        // copy R_buf to C_buf
        memcpy(C_buf, R_buf, sizeof(float)*NUM_DEPTH*BLK_SIZE_MN);
      }

      // final write back
      memcpy(&C_blk->mat[(i*num_blk_N+j)*BLK_SIZE_MN],
             C_buf,
             sizeof(float)*BLK_SIZE_MN*num_depth);
    }
  }
}
