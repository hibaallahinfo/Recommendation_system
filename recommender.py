import os
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras import Sequential

# Dataset path
DATASET_PATH = "./static/data/"

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

# Enhanced graph and recommendation functions
def compute_multi_metric_similarity(embeddings_a, embeddings_b):
    cosine_sim = cosine_similarity(embeddings_a, embeddings_b)
    euclidean_dist = euclidean_distances(embeddings_a, embeddings_b)
    scaler = MinMaxScaler()
    euclidean_sim = 1 - scaler.fit_transform(euclidean_dist)
    combined_sim = 0.8 * cosine_sim + 0.2 * euclidean_sim
    return combined_sim

def build_enhanced_graph(df_embs, min_threshold=0.9, max_neighbors=10):
    G = nx.Graph()
    num_items = len(df_embs)
    similarities = compute_multi_metric_similarity(df_embs, df_embs)
    similarities = np.power(similarities, 2)
    
    for i in range(num_items):
        G.add_node(i)
        sim_scores = similarities[i]
        potential_neighbors = np.where(sim_scores > min_threshold)[0]
        potential_neighbors = potential_neighbors[potential_neighbors != i]
        if len(potential_neighbors) > max_neighbors:
            top_neighbors = potential_neighbors[np.argsort(sim_scores[potential_neighbors])[-max_neighbors:]]
        else:
            top_neighbors = potential_neighbors
            
        for j in top_neighbors:
            G.add_edge(i, j, weight=similarities[i, j])
    
    return G, similarities

def compute_enhanced_attention(graph, embeddings, similarities, temperature=0.5):
    attention_weights = {}
    for node in graph.nodes():
        neighbors = list(graph.neighbors(node))
        if not neighbors:
            continue
        neighbor_similarities = similarities[node, neighbors]
        attention_scores = np.exp(neighbor_similarities / temperature)
        attention_weights[node] = attention_scores / np.sum(attention_scores)
    return attention_weights

def propagate_enhanced_information(graph, embeddings, attention_weights, num_iterations=3, decay_factor=0.9):
    updated_embeddings = embeddings.copy()
    for iteration in range(num_iterations):
        new_embeddings = updated_embeddings.copy()
        current_decay = decay_factor ** iteration
        for node in graph.nodes():
            neighbors = list(graph.neighbors(node))
            if not neighbors:
                continue
            aggregated = np.zeros_like(embeddings[node])
            total_weight = 0
            for i, neighbor in enumerate(neighbors):
                weight = attention_weights[node][i] * current_decay
                aggregated += weight * updated_embeddings[neighbor]
                total_weight += weight
            if total_weight > 0:
                new_embeddings[node] = (0.8 * embeddings[node] + 0.2 * (aggregated / total_weight))
        updated_embeddings = new_embeddings
    return updated_embeddings

def find_enhanced_similar_items(target_idx, embeddings, similarities, top_n=8, similarity_threshold=0.85):
    initial_similarities = similarities[target_idx]
    valid_candidates = np.where(initial_similarities > similarity_threshold)[0]
    valid_candidates = valid_candidates[valid_candidates != target_idx]
    if len(valid_candidates) == 0:
        return np.array([]), np.array([])
    candidate_similarities = similarities[valid_candidates][:, valid_candidates]
    coherence_scores = np.mean(candidate_similarities, axis=1)
    final_scores = 0.7 * initial_similarities[valid_candidates] + 0.3 * coherence_scores
    top_indices = np.argsort(final_scores)[::-1][:top_n]
    selected_items = valid_candidates[top_indices]
    selected_weights = initial_similarities[selected_items]
    return selected_items, selected_weights

# Run enhanced pipeline
#graph, similarities = build_enhanced_graph(df_embs.values)
#attention_weights = compute_enhanced_attention(graph, df_embs.values, similarities)
#updated_embeddings = propagate_enhanced_information(graph, df_embs.values, attention_weights)
