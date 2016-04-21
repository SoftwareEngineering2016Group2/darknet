#ifndef GEMM_CONSTS_H__
#define GEMM_CONSTS_H__

#define BLK_M     32
#define BLK_N     32
#define BLK_K     32

#define NUM_PIPES 4
#define NUM_DEPTH 2

#define BLK_SIZE_MN (BLK_M*BLK_N)
#define BLK_SIZE_KN (BLK_N*BLK_K)
#define BLK_SIZE_MK (BLK_M*BLK_K)

#define PIPE_SIZE_MN (BLK_SIZE_MN*NUM_PIPES)
#define PIPE_SIZE_KN (BLK_SIZE_KN*NUM_PIPES)
#define PIPE_SIZE_MK (BLK_SIZE_MK*NUM_PIPES)

#endif
