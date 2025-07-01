import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse
import torch
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights

def extract_features(crop_path, output_file):
    # Convert to absolute path
    crop_path = os.path.abspath(crop_path)
    print(f"🔍 Current working directory: {os.getcwd()}")
    print(f"🔍 Absolute crop_path: {crop_path}")

    if not os.path.exists(crop_path):
        raise FileNotFoundError(f"❌ Crop path not found: {crop_path}")

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = torch.nn.Identity()
    model = model.to(device).eval()

    # Preprocessing
    transform = T.Compose([
        T.Resize((256, 128)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    features = []
    filenames = []

    for fname in tqdm(os.listdir(crop_path), desc="🔄 Extracting features"):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        path = os.path.join(crop_path, fname)
        img = Image.open(path).convert('RGB')
        img = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            feat = model(img).squeeze().cpu().numpy()

        filenames.append(fname)
        features.append(feat)

    feats = np.stack(features)
    np.savez(output_file, filenames=filenames, features=feats)
    print(f"✅ Saved {len(filenames)} features to {output_file}.npz")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--crop_path",
        required=True,
        help="Absolute path to YOLO crops folder"
    )
    parser.add_argument(
        "--output_file",
        required=True,
        help="Output .npz file name (e.g., player_features_broadcast)"
    )
    args = parser.parse_args()

    # Force .npz extension
    if not args.output_file.endswith(".npz"):
        args.output_file += ".npz"

    extract_features(args.crop_path, args.output_file)
