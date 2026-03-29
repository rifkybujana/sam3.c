/*
 * src/core/graph.c - Compute graph construction
 *
 * Builds the compute graph by appending nodes. The graph is a simple
 * linear array — topological ordering is the responsibility of the
 * caller (model code builds nodes in evaluation order).
 *
 * Key types:  sam3_graph, sam3_node
 * Depends on: graph.h
 * Used by:    model/ files
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#include <string.h>

#include "graph.h"

void sam3_graph_init(struct sam3_graph *g)
{
	memset(g, 0, sizeof(*g));
}

struct sam3_tensor *sam3_graph_add_op(struct sam3_graph *g, enum sam3_op op,
				     struct sam3_tensor **inputs, int n_inputs,
				     struct sam3_tensor *output)
{
	if (g->n_nodes >= SAM3_GRAPH_MAX_NODES)
		return NULL;

	if (n_inputs > SAM3_NODE_MAX_INPUTS)
		return NULL;

	struct sam3_node *node = &g->nodes[g->n_nodes++];
	node->op = op;
	node->n_inputs = n_inputs;
	node->output = output;

	for (int i = 0; i < n_inputs; i++)
		node->inputs[i] = inputs[i];

	return output;
}
