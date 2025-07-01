# player_reid
# 🧠 Player Crop Feature Extraction and Re-Identification with ResNet50

This repository provides a lightweight and efficient pipeline to extract deep feature vectors from player image crops using a pre-trained ResNet50 model in PyTorch. The extracted features are then used to match players across two different camera views (`broadcast` and `tacticam`) via cosine similarity, enabling player re-identification.

---

## 📂 File Structure

```
player_reid/
├── extract_features.py            # Extracts ResNet50 features from player crops
├── match_features.py              # Matches players by cosine similarity between features
├── visualize_matches.py           # (Optional) Visualizes matched player crops side-by-side
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── features/
│   ├── player_features_broadcast.npz
│   └── player_features_tacticam.npz
├── results/
│   ├── matched_players.csv
│   └── player_mapping.json       # (Optional JSON match dictionary)
└── runs/
└── detect/
    ├── predict_broadcast/        # YOLOv11 detection outputs for broadcast crops
    └── predict_tacticam/         # YOLOv11 detection outputs for tacticam crops
```

---

## ⚙️ Installation and Setup

### 1. Clone This Repository

```bash
git clone https://github.com/anisha-singh-2004/player_reid.git
cd player_reid
```

### 2. Clone YOLOv7 (for player detection)

We use YOLOv11 to detect and crop players before feature extraction.

```bash
git clone https://github.com/WongKinYiu/yolov7.git
cd yolov11
pip install -r requirements.txt
cd ..
```

### 3. (Optional) Create and activate a virtual environment

```bash
python -m venv player_reid_venv
source player_reid_venv/bin/activate      # On Windows: player_reid_venv\Scripts\activate
```

### 4. Install Required Python Packages

```bash
pip install torch torchvision pillow numpy tqdm scikit-learn pandas matplotlib opencv-python
```

---

## 🚀 Usage

### Step 1: Detect Players and Extract Crops using YOLOv7

Run YOLOv11 detection separately on both videos to get cropped player images:

```bash
python yolov7/detect.py --weights yolov7.pt --conf 0.5 --source <broadcast_video_path> --save-crop --project runs/detect --name predict_broadcast --exist-ok
```

```bash
python yolov7/detect.py --weights yolov7.pt --conf 0.5 --source <tacticam_video_path> --save-crop --project runs/detect --name predict_tacticam --exist-ok
```

This will create cropped player images under:

* `runs/detect/predict_broadcast/crops/player/`
* `runs/detect/predict_tacticam/crops/player/`

---

### Step 2: Extract Deep Features from Player Crops

Run the feature extraction script on each crop folder:

```bash
python extract_features.py --crop_path runs/detect/predict_broadcast/crops/player --output_file features/player_features_broadcast.npz
python extract_features.py --crop_path runs/detect/predict_tacticam/crops/player --output_file features/player_features_tacticam.npz
```

---

### Step 3: Match Players Across Views using Cosine Similarity

Run the matching script:

```bash
python match_features.py
```

This script:

* Loads extracted `.npz` features for broadcast and tacticam players.
* Computes cosine similarity between all pairs.
* Finds the best tacticam match for each broadcast player.
* Saves results to `results/matched_players.csv` with similarity scores.
* Optionally saves JSON mapping in `results/player_mapping.json`.

---

### Step 4: Visualize Matches (Optional)

Use the visualization script to display side-by-side matched crops with similarity scores:

```bash
python visualize_matches.py
```

This helps you qualitatively verify match quality.

---

## 📄 Detailed Script Descriptions

### `extract_features.py`

* Uses a pre-trained ResNet50 model with the final classification layer removed.
* Transforms each crop to 224x224 RGB tensor normalized for ResNet50.
* Extracts 2048-dimensional features.
* Saves `filenames` and `features` arrays in a `.npz` file.

### `match_features.py`

* Loads the `.npz` feature files for both views.
* Computes cosine similarity matrix.
* Finds best matches and saves to CSV and JSON.

### `visualize_matches.py` (optional)

* Loads matches CSV.
* Displays broadcast and tacticam crops side-by-side with similarity scores.

---

## 📋 Requirements

Install dependencies via:

```bash
pip install -r requirements.txt
```

Sample `requirements.txt`:

```
torch
torchvision
numpy
pandas
scikit-learn
tqdm
pillow
matplotlib
opencv-python
```

---

## ⚠️ Notes

* Ensure you have a compatible GPU to speed up feature extraction.
* Adjust YOLOv7 detection confidence (`--conf`) as needed to balance detection quality.
* Crop image quality and consistency significantly impact matching accuracy.
* Scripts assume default crop folder structure from YOLOv7 detections.

---

## 📝 Example Commands Recap

```bash
# Detect players and save crops
python yolov7/detect.py --weights yolov7.pt --conf 0.5 --source broadcast.mp4 --save-crop --project runs/detect --name predict_broadcast --exist-ok
python yolov7/detect.py --weights yolov7.pt --conf 0.5 --source tacticam.mp4 --save-crop --project runs/detect --name predict_tacticam --exist-ok

# Extract features
python extract_features.py --crop_path runs/detect/predict_broadcast/crops/player --output_file features/player_features_broadcast.npz
python extract_features.py --crop_path runs/detect/predict_tacticam/crops/player --output_file features/player_features_tacticam.npz

# Match features and save CSV
python match_features.py

# Visualize matches
python visualize_matches.py
```

---

## 👤 Author

Made with 💻 by **Anisha Singh**  
[GitHub: anisha-singh-2004](https://github.com/anisha-singh-2004)

---

## 🛡 License

This project is licensed under the **MIT License** — feel free to use and modify as you wish.

---

Happy feature extracting and matching! 🚀
