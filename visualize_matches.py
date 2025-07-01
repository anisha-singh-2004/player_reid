import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt

# Edit paths to your cropped image folders
BROADCAST_CROP_DIR = r"C:\Users\HP\player_reid\runs\detect\predict_broadcast\crops\player"
TACTICAM_CROP_DIR = r"C:\Users\HP\player_reid\runs\detect\predict_tacticam\crops\player"

def show_match(broadcast_img, tacticam_img, sim_score):
    img1 = cv2.imread(os.path.join(BROADCAST_CROP_DIR, broadcast_img))
    img2 = cv2.imread(os.path.join(TACTICAM_CROP_DIR, tacticam_img))

    if img1 is None or img2 is None:
        print(f"❌ Could not load images: {broadcast_img} / {tacticam_img}")
        return

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(img1)
    ax[0].set_title("Broadcast")
    ax[0].axis('off')

    ax[1].imshow(img2)
    ax[1].set_title(f"Tacticam\nScore: {sim_score:.2f}")
    ax[1].axis('off')

    plt.tight_layout()
    plt.show()

# Load matches
df = pd.read_csv("matched_players.csv")

# 🔢 How many to display?
N = 10
for i in range(N):
    row = df.iloc[i]
    show_match(row['broadcast_image'], row['tacticam_image'], row['cosine_similarity'])
