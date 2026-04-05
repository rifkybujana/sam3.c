/*
 * src/core/graph.h - Compute graph for inference
 *
 * Represents a DAG of tensor operations. The model builds a compute
 * graph during setup, then the backend evaluates it. This allows
 * backends to optimize execution order, fuse operations, and manage
 * GPU command buffers.
 *
 * Key types:  sam3_graph, sam3_node, sam3_op
 * Depends on: tensor.h
 * Used by:    model/ files, backend/ files
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#ifndef SAM3_CORE_GRAPH_H
#define SAM3_CORE_GRAPH_H

#include "tensor.h"

#define SAM3_GRAPH_MAX_NODES 16384
#define SAM3_NODE_MAX_INPUTS 8

/* Compute operation types. */
enum sam3_op {
	SAM3_OP_NONE,
	SAM3_OP_MATMUL,
	SAM3_OP_ADD,
	SAM3_OP_MUL,
	SAM3_OP_SOFTMAX,
	SAM3_OP_RELU,
	SAM3_OP_GELU,
	SAM3_OP_LAYERNORM,
	SAM3_OP_CONV2D,
	SAM3_OP_RESHAPE,
	SAM3_OP_TRANSPOSE,
	SAM3_OP_CAST,
	SAM3_OP_CONCAT,     /* Join tensors along axis. params[0]=axis */
	SAM3_OP_SLICE,      /* Extract sub-tensor.     params[0]=axis, [1]=start, [2]=end */
	SAM3_OP_EMBED,      /* Table lookup by index.  inputs: table, indices */
	SAM3_OP_SIGMOID,    /* 1/(1+exp(-x))           */
	SAM3_OP_UPSAMPLE,   /* Nearest-neighbor.       params[0]=scale */
	SAM3_OP_ROPE,       /* Rotary position embed.  params[0]=head_dim */
	SAM3_OP_SILU,            /* x * sigmoid(x)          */
	SAM3_OP_CONV_TRANSPOSE2D, /* params[0]=stride, [1]=padding */
	SAM3_OP_MAXPOOL2D,       /* params[0]=kernel, [1]=stride  */
	SAM3_OP_SDPA,            /* Fused scaled dot-product attention.
				  * inputs: Q[seq,hd], K[seq,hd], V[seq,hd],
				  *         mask[seq,seq] (optional).
				  * params[0]=head_dim (scale=1/sqrt(hd)).
				  * output: [seq, head_dim].              */
	SAM3_OP_BIAS_ADD,        /* NCHW bias add: x[N,C,H,W] + bias[C].
				  * inputs[0]=x, inputs[1]=bias.
				  * output: same shape as x.              */
	SAM3_OP_GROUPNORM,      /* GroupNorm on NCHW input.
				  * inputs[0]=x[N,C,H,W],
				  * inputs[1]=gamma[C], inputs[2]=beta[C].
				  * params[0]=num_groups.
				  * output: same shape as x.              */
	SAM3_OP_COUNT,  /* must be last */
};

/* A single node in the compute graph. */
struct sam3_node {
	enum sam3_op         op;
	struct sam3_tensor  *inputs[SAM3_NODE_MAX_INPUTS];
	int                  n_inputs;
	struct sam3_tensor  *output;
	int                  params[4]; /* Op-specific (e.g. conv2d stride/padding) */
};

/* A compute graph: ordered list of nodes. */
struct sam3_graph {
	struct sam3_node nodes[SAM3_GRAPH_MAX_NODES];
	int              n_nodes;
};

/* Initialize an empty compute graph. */
void sam3_graph_init(struct sam3_graph *g);

/* Add a node to the graph. Returns pointer to the output tensor, or NULL. */
struct sam3_tensor *sam3_graph_add_op(struct sam3_graph *g, enum sam3_op op,
				     struct sam3_tensor **inputs, int n_inputs,
				     struct sam3_tensor *output);

/* Return a short string name for the op ("MATMUL", "ADD", etc). */
const char *sam3_op_str(enum sam3_op op);

#endif /* SAM3_CORE_GRAPH_H */
