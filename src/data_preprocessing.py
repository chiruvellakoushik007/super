
import pandas as pd
import re
from pathlib import Path


# Project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Dataset path
DATASET_PATH = BASE_DIR / "dataset" / "mohler_asag.csv"

# Output path
OUTPUT_PATH = BASE_DIR / "dataset" / "cleaned_dataset.csv"


def clean_text(text):
    """Clean student and instructor answers."""

    text = str(text).lower()

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove unnecessary special characters
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def preprocess_dataset():

    print("Loading dataset...")

    data = pd.read_csv(DATASET_PATH)

    print("Original dataset shape:", data.shape)

    # Create cleaned student answers
    data["clean_student_answer"] = (
        data["student_answer"].apply(clean_text)
    )

    # Create cleaned instructor answers
    data["clean_reference_answer"] = (
        data["instructor_answer"].apply(clean_text)
    )

    # Calculate answer lengths
    data["student_answer_length"] = (
        data["clean_student_answer"].str.split().str.len()
    )

    data["reference_answer_length"] = (
        data["clean_reference_answer"].str.split().str.len()
    )

    # Save processed dataset
    data.to_csv(OUTPUT_PATH, index=False)

    print("\nPreprocessing completed!")
    print("Processed dataset shape:", data.shape)

    print("\nNew columns:")
    print([
        "clean_student_answer",
        "clean_reference_answer",
        "student_answer_length",
        "reference_answer_length"
    ])

    print("\nProcessed dataset saved at:")
    print(OUTPUT_PATH)

    return data


if __name__ == "__main__":
    processed_data = preprocess_dataset()

    print("\nFirst five processed records:")

    print(
        processed_data[
            [
                "clean_student_answer",
                "clean_reference_answer",
                "student_answer_length",
                "reference_answer_length",
                "score_avg"
            ]
        ].head()
    )
