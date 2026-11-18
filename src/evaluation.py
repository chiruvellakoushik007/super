
import os
import re
import joblib
import numpy as np

from pathlib import Path

from scipy.sparse import hstack, csr_matrix

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Disable symlinks warning and prefer offline hub mode when cache exists
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")


# =====================================================
# 1. PATH CONFIGURATION
# =====================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "answer_evaluation_model.pkl"
VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.pkl"


# =====================================================
# 2. LAZY MODEL LOADERS
# =====================================================

_model = None
_vectorizer = None
_semantic_model = None


def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Trained ML model not found at {MODEL_PATH}. "
                "Run 'python src/model_training.py' first."
            )
        _model = joblib.load(MODEL_PATH)
    return _model


def get_vectorizer():
    global _vectorizer
    if _vectorizer is None:
        if not VECTORIZER_PATH.exists():
            raise FileNotFoundError(
                f"TF-IDF vectorizer not found at {VECTORIZER_PATH}. "
                "Train the model first."
            )
        _vectorizer = joblib.load(VECTORIZER_PATH)
    return _vectorizer


def get_semantic_model():
    global _semantic_model
    if _semantic_model is None:
        from sentence_transformers import SentenceTransformer
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        try:
            # Attempt loading from local cache first to prevent 30-45s network retry timeouts
            _semantic_model = SentenceTransformer(model_name, local_files_only=True)
        except Exception:
            _semantic_model = SentenceTransformer(model_name)
    return _semantic_model


# Backwards compatibility module-level proxies
class _LazyProxy:
    def __init__(self, loader_func):
        self._loader = loader_func

    def __getattr__(self, name):
        return getattr(self._loader(), name)


model = _LazyProxy(get_model)
vectorizer = _LazyProxy(get_vectorizer)
semantic_model = _LazyProxy(get_semantic_model)


# =====================================================
# 4. TEXT PREPROCESSING
# =====================================================

def clean_text(text):

    if text is None:
        return ""

    text = str(text).lower()

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Keep alphabetic and numeric characters
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# =====================================================
# 5. CONCEPT EXTRACTION
# =====================================================

def extract_concepts(text):

    words = set(text.split())

    concepts = {
        word for word in words
        if word not in ENGLISH_STOP_WORDS
        and len(word) > 2
        and not word.isdigit()
    }

    return concepts


# =====================================================
# 6. SEMANTIC SIMILARITY
# =====================================================

def calculate_semantic_similarity(
    student_answer,
    reference_answer
):

    sem_model = get_semantic_model()
    embeddings = sem_model.encode(
        [
            student_answer,
            reference_answer
        ],
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    similarity = float(
        np.dot(
            embeddings[0],
            embeddings[1]
        )
    )

    return float(np.clip(similarity, -1, 1))


# =====================================================
# 7. TF-IDF SIMILARITY
# =====================================================

def calculate_tfidf_similarity(
    student_answer,
    reference_answer
):

    vec = get_vectorizer()
    vectors = vec.transform([
        student_answer,
        reference_answer
    ])

    similarity = cosine_similarity(
        vectors[0],
        vectors[1]
    )[0, 0]

    return float(similarity)


# =====================================================
# 8. SCORE ADJUSTMENT
# =====================================================

def adjust_score(
    model_score,
    tfidf_similarity,
    semantic_similarity,
    concept_coverage
):

    score = float(
        np.clip(model_score, 0, 5)
    )

    reason = None

    # Detect completely unrelated answers
    if (
        semantic_similarity < 0.15
        and tfidf_similarity < 0.08
    ) or (semantic_similarity < 0.20 and concept_coverage == 0.0):

        score = min(score, 0.5)

        reason = (
            "The student answer appears unrelated "
            "to the instructor's reference answer."
        )

    # Detect answers with very weak relevance
    elif (
        semantic_similarity < 0.25
        and tfidf_similarity < 0.08
    ):

        score = min(score, 1.0)

        reason = (
            "Very little meaningful similarity was "
            "found between the answers."
        )

    # Detect answers with limited relevance
    elif (
        semantic_similarity < 0.35
        and tfidf_similarity < 0.12
    ):

        score = min(score, 1.5)

        reason = (
            "The answer contains limited relevant "
            "information."
        )

    # Detect partial / incomplete answers that got an inflated ML score
    elif concept_coverage < 25.0 and semantic_similarity < 0.55:

        max_allowed = 2.5
        if score > max_allowed:
            score = max_allowed
            reason = (
                "The answer contains only basic or partial aspects "
                "of the expected explanation."
            )

    # Calibrate moderate concept coverage with excessively high scores
    elif concept_coverage < 50.0 and semantic_similarity < 0.70 and score > 3.8:

        score = min(score, 3.5)

    return round(float(np.clip(score, 0, 5)), 2), reason


# =====================================================
# 9. GENERATE DETAILED FEEDBACK
# =====================================================

def generate_feedback(
    score,
    covered,
    missing,
    tfidf_similarity,
    semantic_similarity,
    relevance_warning
):

    feedback = []

    # Irrelevant or heavily adjusted answer
    if relevance_warning:

        feedback.append(
            "Needs improvement. "
            + relevance_warning
        )

        feedback.append(
            "Review the question carefully and ensure your explanation "
            "directly addresses the core required concepts."
        )

    # Strong answer
    elif score >= 4:

        feedback.append(
            "Strong answer! Your response demonstrates "
            "good alignment with the reference answer."
        )

        if missing:
            feedback.append(
                "To improve further, include these "
                "additional reference concepts: "
                + ", ".join(missing)
            )

        else:
            feedback.append(
                "The important reference keywords "
                "are covered."
            )

    # Partial answer
    elif score >= 2.5:

        feedback.append(
            "Partially complete answer. "
            "Some important information is present, "
            "but the explanation needs improvement."
        )

        if missing:
            feedback.append(
                "Consider including: "
                + ", ".join(missing)
            )

        feedback.append(
            "Explain the important concepts more clearly."
        )

    # Weak answer
    else:

        feedback.append(
            "Your answer needs improvement."
        )

        feedback.append(
            "Include more relevant information "
            "from the instructor's expected answer."
        )

        if missing:
            feedback.append(
                "Important concepts to review: "
                + ", ".join(missing)
            )

    # Similarity explanation
    if semantic_similarity >= 0.65:

        feedback.append(
            "Semantic analysis indicates strong "
            "meaning-level similarity."
        )

    elif semantic_similarity >= 0.35:

        feedback.append(
            "Semantic analysis indicates moderate "
            "meaning-level similarity."
        )

    else:

        feedback.append(
            "Semantic analysis indicates weak "
            "meaning-level similarity."
        )

    return "\n\n".join(feedback)


# =====================================================
# 10. MAIN ANSWER EVALUATION FUNCTION
# =====================================================

def evaluate_answer(
    question,
    reference_answer,
    student_answer
):

    # ---------------------------------------------
    # Input validation
    # ---------------------------------------------

    if not str(question).strip():

        raise ValueError(
            "Please enter a valid question."
        )

    if not str(reference_answer).strip():

        raise ValueError(
            "Please enter an instructor reference answer."
        )

    if not str(student_answer).strip():

        raise ValueError(
            "Please enter a student answer."
        )

    # ---------------------------------------------
    # Clean input text
    # ---------------------------------------------

    clean_q = clean_text(question)

    reference = clean_text(reference_answer)

    student = clean_text(student_answer)

    if not clean_q:
        raise ValueError("Please enter a question containing valid alphanumeric words.")

    if not reference:
        raise ValueError("Please enter a reference answer containing valid alphanumeric words.")

    reference_concepts = extract_concepts(reference)

    # ---------------------------------------------
    # Empty / Non-alphanumeric student answer check
    # ---------------------------------------------

    if not student:
        return {
            "question": clean_q,
            "model_score": 0.0,
            "predicted_score": 0.0,
            "tfidf_similarity": 0.0,
            "semantic_similarity": 0.0,
            "concept_coverage": 0.0,
            "covered_concepts": [],
            "potentially_missing_concepts": sorted(reference_concepts),
            "relevance_warning": "The student answer contains no meaningful text or valid words.",
            "feedback": (
                "Needs improvement. The student answer contains no meaningful text or valid words.\n\n"
                "Please provide a complete descriptive answer addressing the question."
            )
        }

    # ---------------------------------------------
    # Concept coverage analysis
    # ---------------------------------------------

    student_concepts = extract_concepts(student)

    covered_concepts = sorted(
        reference_concepts & student_concepts
    )

    missing_concepts = sorted(
        reference_concepts - student_concepts
    )

    if reference_concepts:
        concept_coverage = (
            len(covered_concepts)
            / len(reference_concepts)
        ) * 100
    else:
        concept_coverage = 0.0

    # ---------------------------------------------
    # Calculate similarities
    # ---------------------------------------------

    tfidf_similarity = calculate_tfidf_similarity(
        student,
        reference
    )

    semantic_similarity = calculate_semantic_similarity(
        student,
        reference
    )

    # ---------------------------------------------
    # Calculate answer lengths
    # ---------------------------------------------

    student_length = len(student.split())

    reference_length = len(reference.split())

    # ---------------------------------------------
    # Extract TF-IDF features
    # ---------------------------------------------

    vec = get_vectorizer()

    student_vector = vec.transform([
        student
    ])

    # ---------------------------------------------
    # Numerical features
    # IMPORTANT:
    # Keep the same order as model_training.py
    # ---------------------------------------------

    numerical_features = csr_matrix([[
        tfidf_similarity,
        semantic_similarity,
        student_length,
        reference_length
    ]])

    # ---------------------------------------------
    # Combine all features
    # ---------------------------------------------

    final_features = hstack([
        student_vector,
        numerical_features
    ]).tocsr()

    # ---------------------------------------------
    # Machine Learning prediction
    # ---------------------------------------------

    mdl = get_model()

    model_score = float(
        mdl.predict(final_features)[0]
    )

    model_score = float(
        np.clip(model_score, 0, 5)
    )

    # ---------------------------------------------
    # Relevance-based score adjustment
    # ---------------------------------------------

    predicted_score, relevance_warning = adjust_score(
        model_score,
        tfidf_similarity,
        semantic_similarity,
        concept_coverage
    )

    # ---------------------------------------------
    # Generate feedback
    # ---------------------------------------------

    feedback = generate_feedback(
        predicted_score,
        covered_concepts,
        missing_concepts,
        tfidf_similarity,
        semantic_similarity,
        relevance_warning
    )

    # ---------------------------------------------
    # Return evaluation results
    # ---------------------------------------------

    return {

        "question": clean_q,

        "model_score": round(model_score, 2),

        "predicted_score": predicted_score,

        "tfidf_similarity": round(
            tfidf_similarity, 3
        ),

        "semantic_similarity": round(
            semantic_similarity, 3
        ),

        "concept_coverage": round(
            concept_coverage, 2
        ),

        "covered_concepts": covered_concepts,

        "potentially_missing_concepts": missing_concepts,

        "relevance_warning": relevance_warning,

        "feedback": feedback

    }


# =====================================================
# 11. TEST THE EVALUATION SYSTEM
# =====================================================

if __name__ == "__main__":

    question = (
        "What is the role of a prototype program "
        "in problem solving?"
    )

    reference = (
        "To simulate the behaviour of portions "
        "of the desired software product."
    )

    test_answers = {

        "A - Strong": (
            "A prototype simulates portions of the "
            "desired software product before its "
            "final development."
        ),

        "B - Partial": (
            "A prototype is an early version of "
            "a software program."
        ),

        "C - Irrelevant": (
            "The weather is sunny today and "
            "I enjoy playing cricket."
        )

    }

    print("\n" + "=" * 55)

    print("AI ANSWER EVALUATION SYSTEM")

    print("=" * 55)

    for test_name, student_answer in test_answers.items():

        print("\n", test_name)

        print("-" * 45)

        result = evaluate_answer(
            question,
            reference,
            student_answer
        )

        for key, value in result.items():

            print(f"{key}: {value}")

    print("\n" + "=" * 55)

    print("ALL TEST CASES COMPLETED")

    print("=" * 55)
