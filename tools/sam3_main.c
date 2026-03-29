/*
 * tools/sam3_main.c - SAM3 inference CLI
 *
 * Main command-line tool for running SAM3 segmentation. Takes an image
 * and prompt coordinates, outputs segmentation masks. This is the
 * primary user-facing binary.
 *
 * Usage: sam3 -m <model> -i <image> -p <x,y,label> [-o <output>]
 *
 * Key types:  (uses sam3_ctx from sam3.h)
 * Depends on: sam3/sam3.h
 * Used by:    end users
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <stdlib.h>

#include "sam3/sam3.h"
#include "util/error.h"

int main(int argc, char **argv)
{
	(void)argc;
	(void)argv;

	printf("sam3 inference tool v%s\n", sam3_version());
	printf("usage: sam3 -m <model> -i <image> -p <x,y,label>\n");
	printf("(not yet implemented)\n");

	return 0;
}
