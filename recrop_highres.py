import os
import cv2
from tqdm import tqdm

def yolo_to_pixels(x_center, y_center, width, height, frame_w, frame_h):
    """Convert YOLO normalized bbox to pixel bbox (xmin, ymin, xmax, ymax)."""
    x_center_abs = x_center * frame_w
    y_center_abs = y_center * frame_h
    width_abs = width * frame_w
    height_abs = height * frame_h

    xmin = int(x_center_abs - width_abs / 2)
    ymin = int(y_center_abs - height_abs / 2)
    xmax = int(x_center_abs + width_abs / 2)
    ymax = int(y_center_abs + height_abs / 2)

    # Clamp coordinates
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(frame_w - 1, xmax)
    ymax = min(frame_h - 1, ymax)

    return xmin, ymin, xmax, ymax

def recrop(video_path, labels_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Processing video: {video_path}")
    print(f"Total frames: {frame_count}")
    print(f"Labels folder: {labels_dir}")

    for frame_idx in tqdm(range(frame_count), desc="Frames"):
        ret, frame = cap.read()
        if not ret:
            print(f"Frame {frame_idx} not read, stopping.")
            break

        # YOLO label filename matches frame number, zero padded? Check your folder:
        # Example: frame_000123.txt or 123.txt — adjust as needed
        label_filename = f"{frame_idx:06d}.txt"  # 6-digit zero padding
        label_path = os.path.join(labels_dir, label_filename)

        if not os.path.exists(label_path):
            continue  # no detections on this frame

        with open(label_path, 'r') as f:
            lines = f.readlines()

        for det_idx, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) != 5:
                continue  # malformed line

            cls_id, x_c, y_c, w, h = parts
            x_c, y_c, w, h = map(float, (x_c, y_c, w, h))

            xmin, ymin, xmax, ymax = yolo_to_pixels(x_c, y_c, w, h, frame.shape[1], frame.shape[0])

            crop = frame[ymin:ymax, xmin:xmax]

            if crop.size == 0:
                continue

            # Save crop, e.g. frame000123_det0.jpg
            crop_fname = f"frame{frame_idx:06d}_det{det_idx}.jpg"
            cv2.imwrite(os.path.join(output_dir, crop_fname), crop)

    cap.release()
    print(f"✅ Saved crops to {output_dir}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to original video (e.g., data/broadcast.mp4)")
    parser.add_argument("--labels", required=True, help="Path to YOLO labels folder (e.g., runs/detect/predict_broadcast/labels)")
    parser.add_argument("--output", required=True, help="Output folder for high-res crops")
    args = parser.parse_args()

    recrop(args.video, args.labels, args.output)
