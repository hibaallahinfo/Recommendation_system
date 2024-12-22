import os
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras import Sequential

# Dataset path
DATASET_PATH = "./static/data/"
#print(os.listdir(DATASET_PATH))

# Load dataset
df = pd.read_csv(DATASET_PATH + "styles.csv", on_bad_lines='skip')


# Helper functions


def img_path(img):
    return os.path.join(DATASET_PATH, "images", img)

def load_image(img, resized_fac=1):
    path = img_path(img)
    img = cv2.imread(path)
    if img is None:
        print(f"Image {path} could not be loaded.")
        return np.zeros((100, 100, 3), dtype=np.uint8)
    w, h, _ = img.shape
    return cv2.resize(img, (int(h * resized_fac), int(w * resized_fac)), interpolation=cv2.INTER_CUBIC)




# Image embeddings with pre-trained ResNet50
img_width, img_height = 224, 224
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalMaxPooling2D()
])

def get_embedding(model, img_name):
    path = img_path(img_name)
    img = image.load_img(path, target_size=(img_width, img_height))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return model.predict(x, verbose=0).reshape(-1)

# Compute embeddings
df_sample = df.sample(10)
df_embs = df_sample['image'].apply(lambda img: get_embedding(model, img))
df_embs = pd.DataFrame(df_embs.tolist())

from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn.functional as F
import torch.nn as nn
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# 1. Construire le graphe basé sur les similarités entre les embeddings
def build_graph(df_embs, threshold=0.7):
    """
    Construction du graphe avec sélection des k voisins les plus proches
    """
    G = nx.Graph()
    num_items = len(df_embs)
    similarities = cosine_similarity(df_embs)
    
    for i in range(num_items):
        G.add_node(i)
        # Sélectionner les 10 voisins les plus similaires
        top_k_neighbors = np.argsort(similarities[i])[::-1][1:11]
        
        for j in top_k_neighbors:
            if similarities[i, j] > threshold:
                G.add_edge(i, j, weight=similarities[i, j])
    
    return G
    
# 2. Calculer les poids d'attention pour les voisins dans le graphe
def compute_attention(graph, df_embs ):
    """
    Calcule les poids d'attention pour chaque voisin d'un nœud dans le graphe.
    """
    attention_weights = {}
    for node in graph.nodes():
        neighbors = list(graph.neighbors(node))
        if not neighbors:
            continue
        node_embedding = df_embs[node].reshape(1, -1)
        neighbor_embeddings = df_embs[neighbors]
        similarities = cosine_similarity(node_embedding, neighbor_embeddings).flatten()
        exp_similarities = np.exp(similarities)
        attention_weights[node] = exp_similarities / np.sum(exp_similarities)
    return attention_weights

# 3. Propagation des informations à travers le graphe
def propagate_information(graph, df_embs, attention_weights, num_iterations=4):
    updated_embeddings = df_embs.copy()
    for _ in range(num_iterations):
        new_embeddings = updated_embeddings.copy()
        for node in graph.nodes():
            neighbors = list(graph.neighbors(node))
            if not neighbors:
                continue
            
            # Add a normalization factor
            aggregated = np.zeros_like(df_embs[node])
            total_weight = 0
            
            for i, neighbor in enumerate(neighbors):
                weight = attention_weights[node][i]
                aggregated += weight * updated_embeddings[neighbor]
                total_weight += weight
            
            # Normalize the aggregation
            new_embeddings[node] = aggregated / total_weight if total_weight > 0 else df_embs[node]
        
        updated_embeddings = new_embeddings
    return updated_embeddings

# 4. Recommandation : Trouver les articles les plus similaires
def find_top_similar_items(target_idx, updated_embeddings, top_n=10):
    """
    Trouve les top_n articles les plus similaires à un article cible et calcule leurs poids.
    """
    similarities = cosine_similarity(updated_embeddings[target_idx].reshape(1, -1), updated_embeddings).flatten()
    similar_indices = np.argsort(similarities)[::-1]  # Tri décroissant
    similar_indices = [idx for idx in similar_indices if idx != target_idx]  # Exclure l'article cible
    similar_items = similar_indices[:top_n]
    weights = similarities[similar_items]
    return similar_items, weights

# Pipeline intégré
# Construire le graphe
graph = build_graph(df_embs.values, threshold=0.7)

# Calculer les poids d'attention
attention_weights = compute_attention(graph, df_embs.values)

# Propager les informations dans le graphe
updated_embeddings = propagate_information(graph, df_embs.values, attention_weights, num_iterations=4)
np.save("static/data/updated_embeddings.npy", updated_embeddings)