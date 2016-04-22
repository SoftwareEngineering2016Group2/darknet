
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#include "gemm_trans.h"
#include "gemm_utils.h"

/**
 * New transform functions:
 * - flatten_matrix_to_blocked
 * - BlockedMatrix_to_flatten
 */
BlockedMatrix* flatten_matrix_to_blocked(int T, float *A, 
      int M, int N, int lda, int blk_m, int blk_n) {
  
  int M_align = get_blocked_width(M,blk_m);
  int N_align = get_blocked_width(N,blk_n);

  // generate a BlockedMatrix struct by given input parameters
  BlockedMatrix* blk_mat = (BlockedMatrix *) malloc(sizeof(BlockedMatrix));
  blk_mat->T   = T;
  // size params should be taken carefully
  blk_mat->H   = M_align;
  blk_mat->W   = N_align;
  blk_mat->bH  = blk_m;
  blk_mat->bW  = blk_n;
  blk_mat->oH  = (T == 0) ? M : N;
  blk_mat->oW  = (T == 0) ? N : M;
  blk_mat->ld  = lda;
  blk_mat->mat = (float *) malloc(sizeof(float)*blk_mat->H*blk_mat->W);
  
  if (!blk_mat->mat) {
    fprintf(stderr, "Can't allocate memory for blocked matrix\n");
    exit(1);
  }

  // useful info
  int blk_id;
  int blk_size = blk_m * blk_n;
  int num_blk_M = M_align / blk_m;
  int num_blk_N = N_align / blk_n;
  
  // This is basically a linear algebra trick
  // First, we create a outer map from origin block(src_blk) to transposed block(dst_blk)
  for (blk_id = 0; blk_id < num_blk_M*num_blk_N; blk_id ++) {
    int src_blk_i = (T == 0) ? blk_id / num_blk_N : blk_id % num_blk_N;
    int src_blk_j = (T == 0) ? blk_id % num_blk_N : blk_id / num_blk_N;
    
    int src_base_i = src_blk_i * ((T==0) ? blk_m : blk_n);
    int src_base_j = src_blk_j * ((T==0) ? blk_n : blk_m);
    int blk_i, blk_j, src_i, src_j, blk_idx;
    
    for (blk_idx = 0; blk_idx < blk_size; blk_idx ++) {
      blk_i = (T == 0) ? blk_idx / blk_n : blk_idx % blk_n;
      blk_j = (T == 0) ? blk_idx % blk_n : blk_idx / blk_n;
      src_i = src_base_i + blk_i;
      src_j = src_base_j + blk_j;

      float src = 
        ((T == 0 && src_i < M && src_j < N) ||
         (T == 1 && src_i < N && src_j < M)) 
        ? A[src_i*lda+src_j] : 0.0;
      // printf("i %d j %d src %lf\n", src_i, src_j, src);
      blk_mat->mat[blk_id*blk_size+blk_idx] = src;
    }
  }

  return blk_mat;
}

void blocked_matrix_to_flatten(BlockedMatrix* blk_mat, float *A) {
  int blk_id;
  int blk_size  = blk_mat->bH * blk_mat->bW;
  int num_blk_W = blk_mat->W / blk_mat->bW;
  int blk_n = blk_mat->bW;
  int blk_m = blk_mat->bH;

  int i, j;
  int T = blk_mat->T;
  /* original shape */
  int M = (T) ? blk_mat->oW : blk_mat->oH;
  int N = (T) ? blk_mat->oH : blk_mat->oW;
  int lda = blk_mat->ld;
  for (i = 0; i < blk_mat->W*blk_mat->H; i += blk_size) {
    blk_id = i / blk_size;
    int blk_id_m = blk_id/num_blk_W;
    int blk_id_n = blk_id%num_blk_W;

    for (j = 0; j < blk_size; j++) {
      int x = (T) ? blk_id_n*blk_n+j%blk_n : blk_id_m*blk_m+j/blk_n;
      int y = (T) ? blk_id_m*blk_m+j/blk_n : blk_id_n*blk_n+j%blk_n;
      if (( T && x < N && y < M) ||
          (!T && x < M && y < N)) {
        A[x*lda+y] = blk_mat->mat[i+j];          
      }
    }
  }
}


/* This is a very basic version of transformation:
 * lda
 */
float *trans_to_blocked(int T, float *A, int m, int n, int lda, int blk_m, int blk_n) {
  // sanity checks
  if (!A) {
    fprintf(stderr, "A should not be NULL\n");
    exit(1);
  }
  if ((!T && lda < n) || (T && lda < m)) {
    fprintf(stderr, "LDA(%d) should be larger than N(%d)\n", lda, n);
    exit(1);
  }

  // m_align and n_align decides the matrix size
  int m_align = get_blocked_width(m, blk_m);
  int n_align = get_blocked_width(n, blk_n);

  int blk_size = blk_m*blk_n;
  int blk_per_n = n_align/blk_n;

  float *A_block = (float *) MALLOC(sizeof(float)*m_align*n_align);
  if (!A_block) {
    fprintf(stderr, "Can't allocate memory for (%d, %d)\n", m_align, n_align);
    exit(1);
  }

  int i, j, x, y;
  for (i = 0; i < m_align; i += blk_m) {
    for (j = 0; j < n_align; j += blk_n) {
      int blk_i = i/blk_m;
      int blk_j = j/blk_n;
      int blk_id = blk_i*blk_per_n+blk_j;
      // inside block
      for (x = 0; x < blk_m && x + i < m; x ++) {
        for (y = 0; y < blk_n && y + j < n; y ++) {
          int dst_idx = blk_id*blk_size+x*blk_n+y;
          // Take transpose into account
          int src_idx = (T == 0) ? (i+x)*lda+j+y : (j+y)*lda+(i+x);
          A_block[dst_idx] = A[src_idx];
        }
      }
    }
  }

  return A_block;
}

void trans_from_blocked(int T, float *A, float *A_block, int M, int N, int lda, int blk_m, int blk_n) {
  int i, j;

  int M_align = get_blocked_width(M, blk_m);
  int N_align = get_blocked_width(N, blk_n);
  
  int blk_size = blk_m*blk_n;
  int num_blk_n = N_align/blk_n;

  for (i = 0; i < M_align*N_align; i += blk_size) {
    int blk_id = i/blk_size;
    int blk_id_m = blk_id/num_blk_n;
    int blk_id_n = blk_id%num_blk_n;

    for (j = 0; j < blk_size; j++) {
      int x = (T) ? blk_id_n*blk_n+j%blk_n : blk_id_m*blk_m+j/blk_n;
      int y = (!T) ? blk_id_n*blk_n+j%blk_n : blk_id_m*blk_m+j/blk_n ;
      if ((T  && x < N && y < M) ||
          (!T && x < M && y < N)) {
        A[x*lda+y] = A_block[i+j];          
      }
    }
  }
}
