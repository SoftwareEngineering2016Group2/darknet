#ifndef GEMM_UTILS_H__
#define GEMM_UTILS_H__

#include <stdio.h>
#include <stdlib.h>

#include "gemm_types.h"

/* MEMORY */
// it seems that sds_alloc will fail in this case...
#ifdef __SDSCC__
  #include "sds_lib.h"
  #define MALLOC(x) sds_alloc((x))
  #define FREE(x) sds_free((x))
#else
  #define MALLOC malloc
  #define FREE free
#endif

/* printer */
#define CHECK_MARK "\xE2\x9C\x93"

#define PRINT_HIGHLIGHT(msg)  { printf("\x1b[33m%s\x1b[0m\n", (msg)); }
#define PRINT_TITLE(msg)  { printf("\x1b[1m\x1b[34m%s\x1b[0m\n", (msg)); }
#define PRINT_PASSED(msg) { printf("\x1b[1m\x1b[32m%s %s\x1b[0m\n", CHECK_MARK, (msg)); }
#define PRINT_FAILED(msg) { printf("\x1b[1m\x1b[31m%s\x1b[0m\n", (msg)); }

#ifdef SDGEMM_ONLY
void print_matrix(float *A, int M, int N);
void print_blocked_matrix(float *A, int M, int N, int blk_m, int blk_n);
void show_blocked_matrix(BlockedMatrix* blk_mat, int show_mat);

float *random_matrix(int M, int N);
float *random_general_matrix(int M, int N, int lda);
BlockedMatrix *random_blocked_matrix(int T, int M, int N, int lda, int blk_m, int blk_n);

int is_matrix_equal(float *A, float *B, int M, int N, int lda);
#endif

void free_blocked_matrix(BlockedMatrix *blk_mat);
int get_blocked_width(int orig, int blk_width);


#endif
