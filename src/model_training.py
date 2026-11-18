
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from scipy.sparse import hstack, csr_matrix

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# =====================================================
# 1. PATH CONFIGURATION
# =====================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "dataset" / "features_dataset.csv"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20

TEXT_COLUMN = "clean_student_answer"
REFERENCE_COLUMN = "clean_reference_answer"
GROUP_COLUMN = "question"
TARGET_COLUMN = "score_avg"

NUMERIC_COLS = [
    "tfidf_similarity",
    "semantic_similarity",
    "student_answer_length",
    "reference_answer_length"
]

print("=" * 65)
print("AI-POWERED DESCRIPTIVE ANSWER EVALUATION")
print("IMPROVED MODEL TRAINING")
print("=" * 65)


# =====================================================
# 2. LOAD DATASET
# =====================================================

print("\n[1/7] Loading dataset...")

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}"
    )

data = pd.read_csv(DATA_PATH)

required_cols = [
    GROUP_COLUMN,
    TEXT_COLUMN,
    REFERENCE_COLUMN,
    TARGET_COLUMN,
    "tfidf_similarity",
    "semantic_similarity"
]

missing_cols = [
    col for col in required_cols
    if col not in data.columns
]

if missing_cols:
    raise ValueError(
        f"Missing required columns: {missing_cols}"
    )

data = data.copy()

data[GROUP_COLUMN] = (
    data[GROUP_COLUMN].fillna("").astype(str)
)

data[TEXT_COLUMN] = (
    data[TEXT_COLUMN].fillna("").astype(str)
)

data[REFERENCE_COLUMN] = (
    data[REFERENCE_COLUMN].fillna("").astype(str)
)

data[TARGET_COLUMN] = pd.to_numeric(
    data[TARGET_COLUMN],
    errors="coerce"
)

data = data.dropna(
    subset=[TARGET_COLUMN, GROUP_COLUMN]
).copy()

data[TARGET_COLUMN] = data[TARGET_COLUMN].clip(0, 5)

# Calculate answer lengths
data["student_answer_length"] = (
    data[TEXT_COLUMN].str.split().str.len()
)

data["reference_answer_length"] = (
    data[REFERENCE_COLUMN].str.split().str.len()
)

# Convert numeric features
for col in [
    "tfidf_similarity",
    "semantic_similarity"
]:
    data[col] = pd.to_numeric(
        data[col],
        errors="coerce"
    )

data[NUMERIC_COLS] = (
    data[NUMERIC_COLS]
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0.0)
)

# Remove empty student answers
data = data[
    data[TEXT_COLUMN].str.strip().ne("")
].copy()

if data[GROUP_COLUMN].nunique() < 3:
    raise ValueError(
        "At least 3 unique questions are required."
    )

print("Dataset shape:", data.shape)
print(
    "Unique questions:",
    data[GROUP_COLUMN].nunique()
)

print(
    "Average score:",
    round(data[TARGET_COLUMN].mean(), 3)
)

print("\nScore distribution:")
print(
    data[TARGET_COLUMN]
    .value_counts()
    .sort_index()
)


# =====================================================
# 3. GROUPED TRAIN TEST SPLIT
# =====================================================

print("\n[2/7] Splitting dataset by question...")

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

train_idx, test_idx = next(
    splitter.split(
        data,
        groups=data[GROUP_COLUMN]
    )
)

train_data = data.iloc[train_idx].copy()
test_data = data.iloc[test_idx].copy()

print("Training rows:", len(train_data))
print("Testing rows:", len(test_data))

print(
    "Training questions:",
    train_data[GROUP_COLUMN].nunique()
)

print(
    "Testing questions:",
    test_data[GROUP_COLUMN].nunique()
)

# Check question leakage
overlap = (
    set(train_data[GROUP_COLUMN])
    & set(test_data[GROUP_COLUMN])
)

if overlap:
    raise RuntimeError(
        "Question leakage detected!"
    )


# =====================================================
# 4. TF-IDF FEATURE EXTRACTION
# =====================================================

def make_vectorizer():

    return TfidfVectorizer(
        max_features=1500,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True
    )


def build_features(
    frame,
    vectorizer,
    fit_vectorizer=False
):

    texts = (
        frame[TEXT_COLUMN]
        .fillna("")
        .astype(str)
    )

    if fit_vectorizer:
        all_train_texts = (
            texts.tolist()
            + frame[REFERENCE_COLUMN].fillna("").astype(str).tolist()
        )
        vectorizer.fit(all_train_texts)
        text_matrix = vectorizer.transform(texts)
    else:
        text_matrix = vectorizer.transform(texts)

    numeric_matrix = csr_matrix(
        frame[NUMERIC_COLS].to_numpy(
            dtype=float
        )
    )

    return hstack([
        text_matrix,
        numeric_matrix
    ]).tocsr()


# =====================================================
# 5. DEFINE MACHINE LEARNING MODELS
# =====================================================

def make_models():

    return {

        "Ridge Regression": Pipeline([
            (
                "scaler",
                StandardScaler(with_mean=False)
            ),
            (
                "model",
                Ridge(alpha=10.0)
            )
        ]),

        "Random Forest": RandomForestRegressor(
            n_estimators=250,
            max_depth=20,
            min_samples_leaf=2,
            max_features=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),

        "Extra Trees": ExtraTreesRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            max_features=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

    }


# =====================================================
# 6. GROUPED CROSS VALIDATION
# =====================================================

print("\n[3/7] Comparing machine learning models...")

number_of_groups = (
    train_data[GROUP_COLUMN].nunique()
)

n_splits = min(5, number_of_groups)

if n_splits < 2:
    raise ValueError(
        "Not enough questions for cross-validation."
    )

cv = GroupKFold(
    n_splits=n_splits
)

model_results = []

for model_name in make_models().keys():

    print(
        f"\nEvaluating: {model_name}"
    )

    fold_mae = []
    fold_rmse = []
    fold_r2 = []

    for fold_number, (
        fold_train_idx,
        fold_valid_idx
    ) in enumerate(
        cv.split(
            train_data,
            groups=train_data[GROUP_COLUMN]
        ),
        start=1
    ):

        fold_train = train_data.iloc[
            fold_train_idx
        ].copy()

        fold_valid = train_data.iloc[
            fold_valid_idx
        ].copy()

        # Fit a new vectorizer for each fold
        fold_vectorizer = make_vectorizer()

        X_fold_train = build_features(
            fold_train,
            fold_vectorizer,
            fit_vectorizer=True
        )

        X_fold_valid = build_features(
            fold_valid,
            fold_vectorizer,
            fit_vectorizer=False
        )

        y_fold_train = (
            fold_train[TARGET_COLUMN].to_numpy()
        )

        y_fold_valid = (
            fold_valid[TARGET_COLUMN].to_numpy()
        )

        estimator = make_models()[model_name]

        estimator.fit(
            X_fold_train,
            y_fold_train
        )

        fold_predictions = np.clip(
            estimator.predict(X_fold_valid),
            0,
            5
        )

        fold_mae.append(
            mean_absolute_error(
                y_fold_valid,
                fold_predictions
            )
        )

        fold_rmse.append(
            np.sqrt(
                mean_squared_error(
                    y_fold_valid,
                    fold_predictions
                )
            )
        )

        if len(np.unique(y_fold_valid)) > 1:

            fold_r2.append(
                r2_score(
                    y_fold_valid,
                    fold_predictions
                )
            )

        print(
            f"Fold {fold_number}: "
            f"MAE={fold_mae[-1]:.4f}, "
            f"RMSE={fold_rmse[-1]:.4f}"
        )

    mean_mae = float(np.mean(fold_mae))
    mean_rmse = float(np.mean(fold_rmse))

    mean_r2 = (
        float(np.mean(fold_r2))
        if fold_r2 else float("nan")
    )

    model_results.append({

        "Model": model_name,
        "CV_MAE": mean_mae,
        "CV_RMSE": mean_rmse,
        "CV_R2": mean_r2

    })

    print(
        "Average MAE:",
        round(mean_mae, 4)
    )

    print(
        "Average RMSE:",
        round(mean_rmse, 4)
    )

    print(
        "Average R2:",
        round(mean_r2, 4)
    )


# =====================================================
# 7. SELECT BEST MODEL
# =====================================================

print("\n[4/7] Selecting best model...")

results_df = pd.DataFrame(
    model_results
).sort_values("CV_MAE")

print("\nMODEL COMPARISON")

print(
    results_df.to_string(index=False)
)

best_model_name = str(
    results_df.iloc[0]["Model"]
)

best_cv_mae = float(
    results_df.iloc[0]["CV_MAE"]
)

print(
    "\nSelected model:",
    best_model_name
)


# =====================================================
# 8. TRAIN SELECTED MODEL
# =====================================================

print("\n[5/7] Preparing final training features...")

final_vectorizer = make_vectorizer()

X_train = build_features(
    train_data,
    final_vectorizer,
    fit_vectorizer=True
)

X_test = build_features(
    test_data,
    final_vectorizer,
    fit_vectorizer=False
)

y_train = train_data[
    TARGET_COLUMN
].to_numpy(dtype=float)

y_test = test_data[
    TARGET_COLUMN
].to_numpy(dtype=float)

print(
    "TF-IDF vocabulary size:",
    len(final_vectorizer.vocabulary_)
)

print(
    "Training feature shape:",
    X_train.shape
)

print(
    "Testing feature shape:",
    X_test.shape
)

print("\nTraining selected model...")

final_model = make_models()[best_model_name]

final_model.fit(
    X_train,
    y_train
)


# =====================================================
# 9. MODEL EVALUATION
# =====================================================

print("\n[6/7] Evaluating final model...")

predictions = np.clip(
    final_model.predict(X_test),
    0,
    5
)

mae = float(
    mean_absolute_error(
        y_test,
        predictions
    )
)

rmse = float(
    np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )
)

r2 = (
    float(r2_score(y_test, predictions))
    if len(np.unique(y_test)) > 1
    else float("nan")
)

print("\n" + "=" * 55)
print("FINAL MODEL EVALUATION RESULTS")
print("=" * 55)

print("Selected Model:", best_model_name)
print("MAE:", round(mae, 4))
print("RMSE:", round(rmse, 4))
print("R2 Score:", round(r2, 4))

print("\nSample predictions:")

comparison = pd.DataFrame({

    "Actual Score": y_test[:15],
    "Predicted Score": predictions[:15]

})

print(
    comparison.to_string(index=False)
)


# =====================================================
# 10. SAVE TRAINED MODEL
# =====================================================

print("\n[7/7] Saving trained model...")

joblib.dump(
    final_model,
    MODEL_DIR / "answer_evaluation_model.pkl"
)

joblib.dump(
    final_vectorizer,
    MODEL_DIR / "tfidf_vectorizer.pkl"
)


# =====================================================
# 11. SAVE MODEL METRICS
# =====================================================

metrics = {

    "selected_model": best_model_name,

    "MAE": mae,
    "RMSE": rmse,
    "R2": r2,

    "CV_MAE": best_cv_mae,

    "training_samples": len(train_data),
    "testing_samples": len(test_data),

    "training_questions": int(
        train_data[GROUP_COLUMN].nunique()
    ),

    "testing_questions": int(
        test_data[GROUP_COLUMN].nunique()
    ),

    "model_comparison": model_results,

    "score_range": [0, 5],

    "split_method": "GroupShuffleSplit by question",

    "cv_method": (
        f"GroupKFold with {n_splits} folds"
    )

}

joblib.dump(
    metrics,
    MODEL_DIR / "model_metrics.pkl"
)


# =====================================================
# 12. COMPLETION
# =====================================================

print("\n" + "=" * 65)
print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 65)

print("\nSaved files:")

print(
    MODEL_DIR / "answer_evaluation_model.pkl"
)

print(
    MODEL_DIR / "tfidf_vectorizer.pkl"
)

print(
    MODEL_DIR / "model_metrics.pkl"
)

print("\nNext command:")

print("python src/evaluation.py")
