/*
 * tools/sam3_convert.c - Model weight conversion tool
 *
 * Converts SAM3 model weights from PyTorch/safetensors format to
 * sam3's native binary format. Handles tensor layout transposition,
 * dtype conversion, and weight name mapping.
 *
 * Usage: sam3_convert -i <input.pt> -o <output.sam3>
 *
 * Key types:  (standalone tool)
 * Depends on: sam3/sam3_types.h
 * Used by:    end users (pre-inference step)
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <stdlib.h>

#include "sam3/sam3_types.h"

int main(int argc, char **argv)
{
	(void)argc;
	(void)argv;

	printf("sam3 weight conversion tool\n");
	printf("usage: sam3_convert -i <input.pt> -o <output.sam3>\n");
	printf("(not yet implemented)\n");

	return 0;
}
