import os
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from tqdm import tqdm

import vertexai
from vertexai.preview.language_models import TextEmbeddingModel, TextEmbeddingInput

from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

load_dotenv()
PROJECT_ID = os.getenv("PROJECT_ID")
LOCATION = os.getenv("LOCATION")
INPUT_CSV = "clustering_dataset.csv"  
#INPUT_CSV = "clustering_medChat.csv"  

TEXT_COL = "clean_text"
BATCH_SIZE = 64
K_CANDIDATES = [8, 10, 12, 15, 18]

print("Set VertexAI")
vertexai.init(project=PROJECT_ID, location=LOCATION)

print("Load Dataset")
df = pd.read_csv(INPUT_CSV)
df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str).str.strip()
#df = df[df[TEXT_COL].str.len() <= 250].copy()

print("Start Text Embedding")
model = TextEmbeddingModel.from_pretrained("text-multilingual-embedding-002")

all_embeddings = []
for start in tqdm(range(0, len(df), BATCH_SIZE), desc="Embedding"):
    batch_texts = df.iloc[start:start + BATCH_SIZE][TEXT_COL].tolist()

    inputs = [
        TextEmbeddingInput(task_type="CLUSTERING", text=t)
        for t in batch_texts
    ]

    batch_embeddings = model.get_embeddings(inputs)
    all_embeddings.extend([e.values for e in batch_embeddings])

df["embedding"] = all_embeddings

# Normalization 
X = np.vstack(df["embedding"].values)
X = normalize(X) 

print("\n[Silhouette scores]")
sil_scores = []

for k in K_CANDIDATES:
    km = KMeans(n_clusters=k, n_init="auto", random_state=42)
    labels = km.fit_predict(X)
    score = silhouette_score(X, labels)
    sil_scores.append(score)

plt.figure()
plt.plot(K_CANDIDATES, sil_scores, marker="o")
plt.xlabel("k (number of clusters)")
plt.ylabel("Silhouette score")
plt.title("Silhouette score vs k")
plt.show()


print("K-means")
#best_k=int(input("Choose K"))
FINAL_K = 15 #best_k

print("Start K-means")
km = KMeans(n_clusters=FINAL_K, n_init="auto", random_state=42)
df["cluster"] = km.fit_predict(X)
print("End K-means")
print(f"\nK-means done. FINAL_K={FINAL_K}, clusters={df['cluster'].nunique()}")


print("Clustering Results")
centroids = km.cluster_centers_ 

def top_representatives_for_cluster(cluster_id, top_n=5):
    idxs = np.where(df["cluster"].values == cluster_id)[0]
    if len(idxs) == 0:
        return pd.DataFrame()

    sims = X[idxs] @ centroids[cluster_id]
    top_local = np.argsort(-sims)[:top_n]
    picked = idxs[top_local]

    out = df.loc[picked, [ "cluster", TEXT_COL]].copy()
    out["similarity_to_centroid"] = sims[top_local]
    return out.sort_values("similarity_to_centroid", ascending=False)

rep_rows = []
for c in sorted(df["cluster"].unique()):
    reps = top_representatives_for_cluster(c, top_n=5)
    rep_rows.append(reps)

reps_df = pd.concat(rep_rows, ignore_index=True)

df.to_csv("clustered_with_embeddings.csv", index=False, encoding="utf-8-sig")
reps_df.to_csv("cluster_representatives_top5.csv", index=False, encoding="utf-8-sig")

print("\nSaved CSV Files")
