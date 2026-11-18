
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "dataset" / "cleaned_dataset.csv"
OUTPUT_PATH = BASE_DIR / "dataset" / "features_dataset.csv"


def extract_features():
    print("Loading cleaned dataset...")
    data = pd.read_csv(DATASET_PATH)

    student_text = data["clean_student_answer"].fillna("").tolist()
    reference_text = data["clean_reference_answer"].fillna("").tolist()

    # TF-IDF feature extraction
    print("Calculating TF-IDF similarity...")

    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english"
    )

    all_text = student_text + reference_text
    vectors = vectorizer.fit_transform(all_text)

    n = len(data)
    student_vectors = vectors[:n]
    reference_vectors = vectors[n:]

    # Row-wise cosine similarity
    tfidf_similarity = np.asarray(
        student_vectors.multiply(reference_vectors).sum(axis=1)
    ).ravel()

    # Vectors are L2 normalized by default
    data["tfidf_similarity"] = tfidf_similarity

    # Semantic similarity
    print("Loading Sentence Transformer model...")

    semantic_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Generating semantic embeddings...")

    student_embeddings = semantic_model.encode(
        student_text,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    reference_embeddings = semantic_model.encode(
        reference_text,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    data["semantic_similarity"] = np.sum(
        student_embeddings * reference_embeddings,
        axis=1
    )

    data.to_csv(OUTPUT_PATH, index=False)

    print("\nFeature extraction completed!")
    print("Dataset shape:", data.shape)

    print("\nSample features:")
    print(
        data[
            [
                "tfidf_similarity",
                "semantic_similarity",
                "score_avg"
            ]
        ].head()
    )

    print("\nSaved at:", OUTPUT_PATH)


if __name__ == "__main__":
    extract_features()
