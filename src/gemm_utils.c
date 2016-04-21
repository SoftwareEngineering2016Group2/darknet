
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <assert.h>

#include "gemm_utils.h"

int get_blocked_width(int orig, int blk_width) {
  return (int) ceil((double)orig/blk_width)*blk_width;
}

void free_blocked_matrix(BlockedMatrix *blk_mat) {
  free(blk_mat->mat);
  free(blk_mat);
}
#ifdef SDGEMM_ONLY
// copied
void print_matrix(float *A, int M, int N) {
  int i, j;
  for (i = 0; i < M; i++) {
    printf("[ ");
    for (j = 0; j < N; j++) {
      printf("%8.6f", A[i*N+j]);
      if (j != N-1)
        printf(", ");
    }
    printf(" ]\n");
  }
}

void print_blocked_matrix(float *A, int M, int N, int blk_m, int blk_n) {
  int M_align = (int) ceil((double)M/blk_m)*blk_m;
  int N_align = (int) ceil((double)N/blk_n)*blk_n;

  int i;
  int blk_size = blk_m*blk_n;
  for (i = 0; i < (M_align*N_align); i += blk_size) {
    printf("[%d]:\n", i/blk_size);
    print_matrix(A+i, blk_m, blk_n);
  }
}

void show_blocked_matrix(BlockedMatrix* blk_mat, int show_mat) {
  float *A = blk_mat->mat;
  int M = blk_mat->H, N = blk_mat->W;
  int blk_m = blk_mat->bH, blk_n = blk_mat->bW;

  printf("Blocked matrix:\n");
  printf("===\n");
  printf("%s\n", (blk_mat->T) ? "Transposed" : "Normal");
  printf("M X N:\t\t %d X %d\n", M, N);
  printf("Block:\t\t %d X %d\n", blk_m, blk_n);
  printf("Original:\t %d X %d\n", blk_mat->oH, blk_mat->oW);
  if (show_mat) {
    printf("Matrix:\n");
    print_blocked_matrix(A, M, N, blk_m, blk_n);
  }
  printf("===\n");
}

float *random_matrix(int M, int N) {
  int i;
  float *A = (float *) malloc(sizeof(float)*M*N);

  for (i = 0; i < M*N; i ++)
    A[i] = (float) rand()/RAND_MAX;

  return A;
}

float *random_general_matrix(int M, int N, int lda) {
  if (lda < N) {
    fprintf(stderr, "LDA(%d) should be leq to N(%d)\n", lda, N);
    exit(1);
  }

  int i, j;
  float *A = (float *) malloc(sizeof(float)*M*lda);

  for (i = 0; i < M; i++)
    for (j = 0; j < N; j++)
      A[i*lda+j] = (float) rand()/RAND_MAX;

  return A;
}

BlockedMatrix *random_blocked_matrix(int T, int M, int N, int lda, int blk_m, int blk_n) {

  int M_align = get_blocked_width(M,blk_m);
  int N_align = get_blocked_width(N,blk_n);

  BlockedMatrix* blk_mat = (BlockedMatrix *) malloc(sizeof(BlockedMatrix));
  blk_mat->T   = T;
  // size params should be taken carefully
  blk_mat->H   = M_align;
  blk_mat->W   = N_align;
  blk_mat->bH  = blk_m;
  blk_mat->bW  = blk_n;
  blk_mat->oH  = (T==0) ? M : N;
  blk_mat->oW  = (T==0) ? N : M;
  blk_mat->ld  = lda;
  blk_mat->mat = (float *) MALLOC(sizeof(float)*blk_mat->H*blk_mat->W);

  int i;
  for (i = 0; i < blk_mat->H*blk_mat->W; i++)
    blk_mat->mat[i] = (float) rand()/RAND_MAX;

  return blk_mat;
}

int is_matrix_equal(float *A, float *B, int M, int N, int lda) {
  int is_equal = 1;
  int i, j;
  for (i = 0; i < M; i++)
    for (j = 0; j < N; j++)
      is_equal &= (A[i*lda+j] == B[i*lda+j]);
  return is_equal;
}
#endif
