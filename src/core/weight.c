/*
 * src/core/weight.c - Native .sam3 weight file loader and writer
 *
 * Implements mmap-based loading of .sam3 weight files with FNV-1a hash
 * table for O(1) tensor lookup. Also provides the writer that serializes
 * tensors from any weight_reader into the .sam3 binary format.
 *
 * Key types:  sam3_weight_file
 * Depends on: core/weight.h, util/log.h
 * Used by:    sam3.c, tools/sam3_convert.c
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#include "weight.h"
