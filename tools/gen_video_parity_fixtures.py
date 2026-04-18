#!/usr/bin/env python3
"""Generate parity fixtures from the Python reference predictor.

NOT run automatically — invoke by hand when reference outputs need
refresh. See tests/fixtures/video_kids/README.md for invocation.
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(
	0,
	os.path.abspath(os.path.join(
		os.path.dirname(__file__), "..", "reference", "sam3")))

from sam3.sam3_video_predictor import Sam3VideoPredictor


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--video", required=True)
	ap.add_argument("--out", required=True)
	ap.add_argument("--frames", type=int, default=30)
	args = ap.parse_args()

	os.makedirs(os.path.join(args.out, "frames"), exist_ok=True)

	predictor = Sam3VideoPredictor(
		checkpoint_path=os.environ["SAM3_CKPT"],
		bpe_path=os.environ["SAM3_BPE"],
	)
	state = predictor.init_state(video_path=args.video)

	# Canned prompts: two objects on frame 0. Adjust x,y to match the
	# actual foreground regions in kids.mp4.
	prompts = {
		"obj_1": {"frame": 0, "points": [[400, 250]], "labels": [1]},
		"obj_2": {"frame": 0, "points": [[600, 250]], "labels": [1]},
	}
	with open(os.path.join(args.out, "prompts.json"), "w") as f:
		json.dump(prompts, f, indent=2)

	for name, p in prompts.items():
		obj_id = int(name.split("_")[1])
		predictor.add_new_points_or_box(
			state,
			frame_idx=p["frame"],
			obj_id=obj_id,
			points=np.array(p["points"]),
			labels=np.array(p["labels"]),
		)

	for frame_idx, obj_ids, masks in predictor.propagate_in_video(state):
		if frame_idx >= args.frames:
			break
		for obj_id, m in zip(obj_ids, masks):
			arr = (m > 0).cpu().numpy().astype(np.uint8) * 255
			Image.fromarray(arr.squeeze()).save(os.path.join(
				args.out, "frames",
				f"frame_{frame_idx:04d}_obj_{obj_id}.png"))


if __name__ == "__main__":
	main()
