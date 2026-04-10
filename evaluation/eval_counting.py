"""
NUMINA Counting Accuracy Evaluation via GroundingDINO
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch


def extract_all_frames(video_path):
    """Extract ALL frames from a video file."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def count_objects_in_frame(model, image_rgb, noun, box_threshold, text_threshold):
    """Count instances of `noun` in an RGB image using GroundingDINO."""
    from groundingdino.util.inference import predict
    import groundingdino.datasets.transforms as T
    from PIL import Image

    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    pil_image = Image.fromarray(image_rgb)
    image_transformed, _ = transform(pil_image, None)

    caption = f"{noun} ."
    boxes, logits, phrases = predict(
        model=model,
        image=image_transformed,
        caption=caption,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )
    return len(boxes)


def find_video_for_prompt(video_dir, prompt_idx):
    """Find video file matching a prompt index."""
    prefix = f"{prompt_idx:03d}_"
    for f in sorted(os.listdir(video_dir)):
        if f.endswith(".mp4") and f.startswith(prefix):
            return os.path.join(video_dir, f)

    fallback = os.path.join(video_dir, f"video_{prompt_idx:03d}.mp4")
    if os.path.exists(fallback):
        return fallback

    mp4s = sorted([f for f in os.listdir(video_dir) if f.endswith(".mp4")])
    idx0 = prompt_idx - 1
    if 0 <= idx0 < len(mp4s):
        return os.path.join(video_dir, mp4s[idx0])
    return None


def evaluate(args):
    from groundingdino.util.inference import load_model
    print(f"Loading GroundingDINO from {args.gdino_weights}")
    model = load_model(args.gdino_config, args.gdino_weights)
    print("GroundingDINO loaded.\n")

    with open(args.noun_counts_file, "r") as f:
        all_noun_counts = [json.loads(line.strip()) for line in f if line.strip()]

    prompts = None
    if args.prompt_file and os.path.isfile(args.prompt_file):
        with open(args.prompt_file, "r") as f:
            prompts = [line.strip() for line in f if line.strip()]

    start0 = max(args.start_idx - 1, 0)
    end0 = args.end_idx if args.end_idx is not None else len(all_noun_counts)
    noun_counts_slice = all_noun_counts[start0:end0]
    total = len(noun_counts_slice)

    print(f"Evaluating {total} videos (indices {args.start_idx}-{start0 + total})")
    print("=" * 80)

    all_prompt_accs = []
    all_results = []

    for i, noun_counts in enumerate(noun_counts_slice):
        global_idx = args.start_idx + i
        video_path = find_video_for_prompt(args.video_dir, global_idx)

        if video_path is None or not os.path.exists(video_path):
            print(f"\n[{global_idx}] Video not found, skipping.")
            continue

        prompt_str = prompts[start0 + i] if prompts and (start0 + i) < len(prompts) else ""

        print(f"\n[{global_idx}/{start0 + total}] {prompt_str}")
        print(f"  Targets: {noun_counts}")
        print(f"  Video:   {os.path.basename(video_path)}")

        frames = extract_all_frames(video_path)
        if not frames:
            print(f"  ERROR: No frames extracted, skipping.")
            continue

        print(f"  Frames:  {len(frames)}")

        # Detect per noun per frame
        per_noun_frame_counts = {}
        for noun, target in noun_counts.items():
            detected_per_frame = []
            for frame in frames:
                count = count_objects_in_frame(
                    model, frame, noun,
                    args.box_threshold, args.text_threshold,
                )
                detected_per_frame.append(count)
            per_noun_frame_counts[noun] = detected_per_frame

        # Print per-frame results
        nouns = list(noun_counts.keys())
        n_nouns = len(nouns)
        header = f"  {'Frame':>7} | " + " | ".join(
            f"{n}({noun_counts[n]})" for n in nouns
        ) + " | Score"
        print(header)
        print("  " + "-" * (len(header) - 2))

        frame_scores = []
        for f_idx in range(len(frames)):
            detections = {n: per_noun_frame_counts[n][f_idx] for n in nouns}
            # Score = fraction of categories correct in this frame
            n_correct = sum(1 for n in nouns if detections[n] == noun_counts[n])
            score = n_correct / n_nouns

            frame_scores.append(score)

            cols = []
            for n in nouns:
                d = detections[n]
                t = noun_counts[n]
                mark = "✓" if d == t else "✗"
                cols.append(f"{d:>3}{mark}")
            score_str = f"{score:.2f}"
            print(f"  {f_idx+1:>5}   | " + " | ".join(
                f"{c:>{len(f'{n}({noun_counts[n]})')}}".rjust(len(f"{n}({noun_counts[n]})"))
                for c, n in zip(cols, nouns)
            ) + f" | {score_str}")

        prompt_acc = sum(frame_scores) / len(frames)
        print(f"  Accuracy: {prompt_acc:.4f}  ({n_nouns} categories)")

        all_prompt_accs.append(prompt_acc)
        all_results.append({
            "index": global_idx,
            "prompt": prompt_str,
            "video": os.path.basename(video_path),
            "targets": noun_counts,
            "num_frames": len(frames),
            "per_noun": {
                noun: per_noun_frame_counts[noun] for noun in nouns
            },
            "prompt_accuracy": prompt_acc,
        })

    # Overall
    print("\n" + "=" * 80)
    if all_prompt_accs:
        overall = np.mean(all_prompt_accs)
        print(f"Overall accuracy: {overall:.4f} ({overall*100:.1f}%)")
        print(f"Evaluated: {len(all_prompt_accs)} videos")
    else:
        overall = 0.0
        print("No videos evaluated.")

    # Save
    if args.save_results:
        summary = {
            "overall_accuracy": overall,
            "num_prompts": len(all_prompt_accs),
            "box_threshold": args.box_threshold,
            "text_threshold": args.text_threshold,
            "per_prompt": all_results,
        }
        with open(args.save_results, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"Results saved to {args.save_results}")


def main():
    p = argparse.ArgumentParser(description="NUMINA Counting Accuracy Evaluation")
    p.add_argument("--video_dir", type=str, required=True)
    p.add_argument("--noun_counts_file", type=str, required=True)
    p.add_argument("--prompt_file", type=str, default=None)
    p.add_argument("--gdino_config", type=str,
                   default="groundingdino/config/GroundingDINO_SwinT_OGC.py")
    p.add_argument("--gdino_weights", type=str,
                   default="weights/groundingdino_swint_ogc.pth")
    p.add_argument("--box_threshold", type=float, default=0.32)
    p.add_argument("--text_threshold", type=float, default=0.25)
    p.add_argument("--start_idx", type=int, default=1)
    p.add_argument("--end_idx", type=int, default=None)
    p.add_argument("--save_results", type=str, default="eval_results.json")
    args = p.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
