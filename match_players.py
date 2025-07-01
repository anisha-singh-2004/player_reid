import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Load extracted features
broadcast = np.load("player_features_broadcast.npz", allow_pickle=True)
tacticam = np.load("player_features_tacticam.npz", allow_pickle=True)

fnames_b = broadcast["filenames"]
feats_b = broadcast["features"]

fnames_t = tacticam["filenames"]
feats_t = tacticam["features"]

print(f"🎥 Broadcast players: {len(fnames_b)}")
print(f"📹 Tacticam players: {len(fnames_t)}")

# Cosine similarity
similarity_matrix = cosine_similarity(feats_b, feats_t)

# Best match for each broadcast crop
best_indices = similarity_matrix.argmax(axis=1)
best_scores = similarity_matrix.max(axis=1)

matches = []
for i, (idx, score) in enumerate(zip(best_indices, best_scores)):
    matches.append({
        "broadcast_image": fnames_b[i],
        "tacticam_image": fnames_t[idx],
        "cosine_similarity": float(score)
    })

# Save results to CSV
df = pd.DataFrame(matches)
df.to_csv("matched_players.csv", index=False)
print("✅ Saved matches to matched_players.csv")

# Preview top 5
print("\n🔍 Sample Matches:")
print(df.head(5))
