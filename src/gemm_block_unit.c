
#include "gemm_block_unit.h"

void mmult_kernel(
    float A[BLK_M][BLK_K],
    float B[BLK_K][BLK_N],
    float T[BLK_M*BLK_N])
{
#pragma HLS INLINE self
#pragma HLS array_partition variable=A block factor=16 dim=2
#pragma HLS array_partition variable=B block factor=16 dim=1
  int i, j, k;
  for (i = 0; i < BLK_M; i ++) 
    for (j = 0; j < BLK_N; j ++) {
    #pragma HLS pipeline II=1
      float sum = 0.0;
      for (k = 0; k < BLK_K; k ++) {
        float temp = A[i][k] * B[k][j];
        sum += temp;
      }
      T[i*BLK_N+j] = sum;
    }
}

void gemm_block_units_mmult_kernel(
        float A[NUM_DEPTH][NUM_PIPES][BLK_M][BLK_K],
        float B[NUM_DEPTH][NUM_PIPES][BLK_K][BLK_N],
        float T[NUM_DEPTH*NUM_PIPES*BLK_M*BLK_N])
{
#pragma HLS INLINE self
  int d, p;
  for (d = 0; d < NUM_DEPTH; d ++) {
  #pragma HLS unroll
    for (p = 0; p < NUM_PIPES; p ++) {
    #pragma HLS unroll
      mmult_kernel(A[d][p],B[d][p],&T[d*PIPE_SIZE_MN+p*BLK_SIZE_MN]);
    }
  }
}

void gemm_block_units_mmult(
        float A[NUM_DEPTH*NUM_PIPES*BLK_M*BLK_K],
        float B[NUM_DEPTH*NUM_PIPES*BLK_K*BLK_N],
        float T[NUM_DEPTH*NUM_PIPES*BLK_M*BLK_N])
{
  float A_buf[NUM_DEPTH][NUM_PIPES][BLK_M][BLK_K];
  float B_buf[NUM_DEPTH][NUM_PIPES][BLK_K][BLK_N];

  int d, p, i, j;
  for (d = 0; d < NUM_DEPTH; d ++)
    for (p = 0; p < NUM_PIPES; p ++)
      for (i = 0; i < BLK_M; i ++)
        for (j = 0; j < BLK_K; j ++)
        #pragma HLS pipeline II=1
          A_buf[d][p][i][j] = A[j+i*BLK_K+p*BLK_SIZE_MK+d*PIPE_SIZE_MK];

  for (d = 0; d < NUM_DEPTH; d ++)
    for (p = 0; p < NUM_PIPES; p ++)
      for (i = 0; i < BLK_K; i ++)
        for (j = 0; j < BLK_N; j ++)
        #pragma HLS pipeline II=1
          B_buf[d][p][i][j] = B[d*PIPE_SIZE_KN+p*BLK_SIZE_KN+i*BLK_N+j];

  gemm_block_units_mmult_kernel(A_buf,B_buf,T);
}

void gemm_block_units_mplus_kernel(
        float T[NUM_DEPTH][NUM_PIPES][BLK_M][BLK_N],
        float C[NUM_DEPTH][BLK_M][BLK_N],
        float R[NUM_DEPTH*BLK_M*BLK_N])
{
#pragma HLS INLINE self
  // On board BRAM for adder tree
  float R_[NUM_DEPTH][BLK_M][BLK_N];

  int d, p, i, j;
  for (d = 0; d < NUM_DEPTH; d ++)
    for (i = 0; i < BLK_M; i ++)
      for (j = 0; j < BLK_N; j ++)
      #pragma HLS pipeline II=1
        R_[d][i][j] = C[d][i][j];

  // balanced automatically
  for (d = 0; d < NUM_DEPTH; d ++)
    for (p = 0; p < NUM_PIPES; p ++) 
      for (i = 0; i < BLK_M; i ++)
        for (j = 0; j < BLK_N; j ++) 
        #pragma HLS pipeline II=1
          R_[d][i][j] += T[d][p][i][j];
  
  for (d = 0; d < NUM_DEPTH; d ++)
    for (i = 0; i < BLK_M; i ++)
      for (j = 0; j < BLK_N; j ++)
      #pragma HLS pipeline II=1
        R[d*BLK_SIZE_MN+i*BLK_N+j] = R_[d][i][j];
}

void gemm_block_units_mplus(
        float T[NUM_DEPTH*NUM_PIPES*BLK_M*BLK_N],
        float C[NUM_DEPTH*BLK_M*BLK_N],
        float R[NUM_DEPTH*BLK_M*BLK_N])
{
  float T_buf[NUM_DEPTH][NUM_PIPES][BLK_M][BLK_N];
  float C_buf[NUM_DEPTH][BLK_M][BLK_N];

  int d, p, i, j;
  for (d = 0; d < NUM_DEPTH; d ++)
    for (p = 0; p < NUM_PIPES; p ++)
      for (i = 0; i < BLK_M; i ++)
        for (j = 0; j < BLK_N; j ++)
          T_buf[d][p][i][j] = T[d*PIPE_SIZE_MN+p*BLK_SIZE_MN+i*BLK_N+j];

  for (d = 0; d < NUM_DEPTH; d ++)
    for (i = 0; i < BLK_M; i ++)
      for (j = 0; j < BLK_N; j ++)
        C_buf[d][i][j] = C[d*BLK_SIZE_MN+i*BLK_N+j];

  gemm_block_units_mplus_kernel(T_buf,C_buf,R);
}
