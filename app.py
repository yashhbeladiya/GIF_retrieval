
import streamlit as st
import pandas as pd
import numpy as np
import torch
import clip
from PIL import Image
import requests
from io import BytesIO
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import os
from datetime import datetime

# Initialize models
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)

# Paths
EMBEDDINGS_DIR = "embeddings"
VISUAL_INDEX = os.path.join(EMBEDDINGS_DIR, "index_visual.faiss")
TEXT_INDEX = os.path.join(EMBEDDINGS_DIR, "index_text.faiss")
IMAGE_EMBEDDINGS = os.path.join(EMBEDDINGS_DIR, "image_embeddings.npy")
TEXT_EMBEDDINGS = os.path.join(EMBEDDINGS_DIR, "text_embeddings.npy")
SENTENCE_EMBEDDINGS = os.path.join(EMBEDDINGS_DIR, "sentence_embeddings.npy")
PATHS = os.path.join(EMBEDDINGS_DIR, "paths.npy")
CAPTIONS = os.path.join(EMBEDDINGS_DIR, "captions.npy")
EMOTIONS = os.path.join(EMBEDDINGS_DIR, "emotions.npy")
FEEDBACK_FILE = os.path.join(EMBEDDINGS_DIR, "feedback.csv")

# Load data
def load_data_st():
    image_embs = np.load(IMAGE_EMBEDDINGS)
    text_embs = np.load(TEXT_EMBEDDINGS)
    sentence_embs = np.load(SENTENCE_EMBEDDINGS)
    paths = np.load(PATHS, allow_pickle=True)
    captions = np.load(CAPTIONS, allow_pickle=True)
    emotions = np.load(EMOTIONS, allow_pickle=True)
    return image_embs, text_embs, sentence_embs, paths, captions, emotions

# Search functions
def search_text(query, sentence_embs, paths, captions, emotions, top_k=5):
    query_emotion_scores = emotion_classifier(query)[0]
    query_emotion = max(query_emotion_scores, key=lambda x: x['score'])['label']

    text_input = clip.tokenize([query[:77]]).to(device)
    with torch.no_grad():
        query_emb = clip_model.encode_text(text_input).cpu().numpy()

    query_sentence_emb = sentence_model.encode(query)

    index_text = faiss.read_index(TEXT_INDEX)
    distances, indices = index_text.search(query_emb, top_k * 2)

    results = []
    for idx in indices[0]:
        emo_score = 1.0 if emotions[idx] == query_emotion else 0.5
        sem_score = np.dot(query_sentence_emb, sentence_embs[idx]) / (np.linalg.norm(query_sentence_emb) * np.linalg.norm(sentence_embs[idx]))
        score = 0.5 * emo_score + 0.5 * sem_score
        results.append((idx, score))

    results = sorted(results, key=lambda x: x[1], reverse=True)[:top_k]
    return [(paths[idx], captions[idx], emotions[idx]) for idx, _ in results]

def search_image(image, image_embs, paths, captions, emotions, top_k=5):
    img_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        img_emb = clip_model.encode_image(img_input).cpu().numpy()

    index_visual = faiss.read_index(VISUAL_INDEX)
    distances, indices = index_visual.search(img_emb, top_k)

    return [(paths[idx], captions[idx], emotions[idx]) for idx in indices[0]]

def get_query_suggestions(query, captions, sentence_embs, emotions):
    query_emb = sentence_model.encode(query)
    scores = np.dot(sentence_embs, query_emb) / (np.linalg.norm(sentence_embs, axis=1) * np.linalg.norm(query_emb))
    top_indices = np.argsort(scores)[-5:][::-1]
    return list(set([emotions[idx] for idx in top_indices]))

def get_mood_results(mood, paths, captions, emotions, top_k=5):
    indices = np.where(emotions == mood)[0][:top_k]
    return [(paths[idx], captions[idx], emotions[idx]) for idx in indices]

def get_contextual_suggestions(sentence_embs, paths, captions, emotions, top_k=5):
    hour = datetime.now().hour
    target_emotion = "joy" if 18 <= hour <= 23 else "surprise" if 0 <= hour <= 5 else "neutral"
    indices = np.where(emotions == target_emotion)[0][:top_k]
    return [(paths[idx], captions[idx], emotions[idx]) for idx in indices]

def save_feedback(gif_url, rating):
    feedback = pd.DataFrame([[gif_url, rating]], columns=['gif_url', 'rating'])
    if os.path.exists(FEEDBACK_FILE):
        feedback.to_csv(FEEDBACK_FILE, mode='a', header=False, index=False)
    else:
        feedback.to_csv(FEEDBACK_FILE, index=False)
    print(f"Feedback saved: {gif_url} - {rating}")

def main():
    st.title("Meme Retrieval System")
    with st.spinner("Loading data ..."):
        image_embs, text_embs, sentence_embs, paths, captions, emotions = load_data_st()

    tab1, tab2, tab3 = st.tabs(["Search by Text", "Search by Image", "Mood Browser"])

    with tab1:
        st.header("Search by Text")
        query = st.text_input("Enter your query (e.g., 'feeling excited'):")
        if query:
            results = search_text(query, sentence_embs, paths, captions, emotions)
            st.write("### Results:")
            for url, caption, emotion in results:
                st.image(url, caption=f"{caption} ({emotion})", use_container_width=True)
                rating = st.radio(f"Rate: {caption}", ["✅ On Point", "🤔 relatable", "❌ meh"], key=url, horizontal=True)
                submit_button = st.button("Submit Feedback", key=f"submit_{url}")
                if submit_button and rating:
                    save_feedback(url, rating)
                    st.success("Feedback submitted!")

                suggestions = get_query_suggestions(query, captions, sentence_embs, emotions)

                st.write("### Try these emotions:", ", ".join(suggestions))

    with tab2:
        st.header("Search by Image")
        uploaded_file = st.file_uploader("Upload an image", type=["gif", "jpeg", "png", "jpg"])
        if uploaded_file:
            image = Image.open(uploaded_file)
            results = search_image(image, image_embs, paths, captions, emotions)
            st.write("### Similar Memes:")
            for url, caption, emotion in results:
                st.image(url, caption=f"{caption} ({emotion})", use_container_width=True)
                rating = st.radio(f"Rate: {caption}", ["✅ On Point", "🤔 relatable", "❌ meh"], key=url, horizontal=True)
                submit_button = st.button("Submit Feedback", key=f"submit_{url}")
                if submit_button and rating:
                    save_feedback(url, rating)
                    st.success("Feedback submitted!")

    with tab3:
        st.header("Mood Browser")
        mood = st.selectbox("Pick a mood", ["joy", "sadness", "anger", "fear", "surprise", "disgust"])
        
        if mood:
            results = get_mood_results(mood, paths, captions, emotions)
            st.write("### Memes for your mood:")
            for url, caption, emotion in results:
                st.image(url, caption=f"{caption} ({emotion})", use_container_width=True)
                rating = st.radio(f"Rate: {caption}", ["✅ On Point", "🤔 relatable", "❌ meh"], key=url, horizontal=True)
                submit_button = st.button("Submit Feedback", key=f"submit_{url}")
                if submit_button and rating:
                    save_feedback(url, rating)
                    st.success("Feedback submitted!")

    st.sidebar.write("### Trending Now")
    contextual_results = get_contextual_suggestions(sentence_embs, paths, captions, emotions)
    for url, caption, emotion in contextual_results:
        st.sidebar.image(url, caption=f"{caption} ({emotion})", use_container_width=True)
        
if __name__ == "__main__":
    main()
