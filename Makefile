GPU=0
OPENCV=0
DEBUG=0
PROF=0
# SDS
SDS=0
HW=0
BLK=0

VPATH=./src/
EXEC=darknet
OBJDIR=./obj/

CC=gcc
NVCC=nvcc
OPTS=-Ofast
LDFLAGS= -lm -pthread -lstdc++ 
COMMON= 
CFLAGS=-Wall -Wfatal-errors -Wno-unknown-pragmas

ifeq ($(SDS), 1)
PLATFORM = zed
SDSFLAGS = -sds-pf $(PLATFORM)

CC=sdscc $(SDSFLAGS)
endif

CFLAGS = -Wall -Wfatal-errors -Ofast
CFLAGS += -MMD -MP -MF"$(@:%.o=%.d)"
LFLAGS = -O3

LDFLAGS= -lm -pthread -lstdc++ 
COMMON= 

ifeq ($(PROF), 1)
OPTS=-O0 -pg
endif

ifeq ($(OPENCV), 1) 
COMMON+= -DOPENCV
CFLAGS+= -DOPENCV
LDFLAGS+= `pkg-config --libs opencv` 
COMMON+= `pkg-config --cflags opencv` 
endif

ifeq ($(BLK), 1)
COMMON+= -DBLK
CFLAGS+= -DBLK
endif

OBJ=gemm.o gemm_sds.o gemm_utils.o gemm_trans.o gemm_block.o gemm_block_unit.o utils.o cuda.o deconvolutional_layer.o convolutional_layer.o list.o image.o activations.o im2col.o col2im.o blas.o crop_layer.o dropout_layer.o maxpool_layer.o softmax_layer.o data.o matrix.o network.o connected_layer.o cost_layer.o parser.o option_list.o darknet.o detection_layer.o imagenet.o captcha.o route_layer.o writing.o box.o nightmare.o normalization_layer.o avgpool_layer.o coco.o dice.o yolo.o layer.o compare.o classifier.o local_layer.o swag.o shortcut_layer.o activation_layer.o rnn_layer.o rnn.o rnn_vid.o crnn_layer.o coco_demo.o tag.o cifar.o yolo_demo.o go.o
ifeq ($(GPU), 1) 
OBJ+=convolutional_kernels.o deconvolutional_kernels.o activation_kernels.o im2col_kernels.o col2im_kernels.o blas_kernels.o crop_layer_kernels.o dropout_layer_kernels.o maxpool_layer_kernels.o softmax_layer_kernels.o network_kernels.o avgpool_layer_kernels.o
endif

OBJS = $(addprefix $(OBJDIR), $(OBJ))
DEPS = $(wildcard src/*.h) Makefile

all: obj results $(EXEC)

$(EXEC): $(OBJS)
	$(CC) $(COMMON) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(OBJDIR)%.o: %.c $(DEPS)
	$(CC) $(COMMON) $(CFLAGS) -c $< -o $@

obj:
	mkdir -p obj
results:
	mkdir -p results

.PHONY: clean

clean:
	rm -rf $(OBJS) $(EXEC)

