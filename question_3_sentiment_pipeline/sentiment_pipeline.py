"""Mini sentiment-classification pipeline using TF-IDF and Logistic Regression."""

import json
import re
from pathlib import Path
from urllib.request import urlretrieve

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

DATASET_URL = "https://lief-assets-storage.sgp1.cdn.digitaloceanspaces.com/Test/dataset.txt"
STOPWORDS = {"is", "the", "and", "a", "an", "of"}


def preprocess(text: str) -> str:
    """Lowercase, remove punctuation and specified stopwords, then tokenize."""
    lowercase_text = text.lower()
    without_punctuation = re.sub(r"[^a-z0-9\s]", " ", lowercase_text)
    tokens = without_punctuation.split()
    return " ".join(token for token in tokens if token not in STOPWORDS)


def load_dataset(dataset_path: Path) -> tuple[list[str], list[int]]:
    if not dataset_path.exists():
        print(f"Downloading dataset from {DATASET_URL}")
        urlretrieve(DATASET_URL, dataset_path)

    records = json.loads(dataset_path.read_text(encoding="utf-8"))
    texts = [preprocess(record["text"]) for record in records]
    labels = [int(record["label"]) for record in records]
    return texts, labels


def main() -> None:
    dataset_path = Path(__file__).with_name("dataset.txt")
    texts, labels = load_dataset(dataset_path)
    x_train, x_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    model = Pipeline(
        [
            ("vectorizer", TfidfVectorizer()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    accuracy = accuracy_score(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    metrics = {
        "dataset_size": len(texts),
        "train_size": len(x_train),
        "test_size": len(x_test),
        "accuracy": accuracy,
        "labels": [0, 1],
        "confusion_matrix": matrix.tolist(),
    }
    metrics_path = Path(__file__).with_name("metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    print(f"Dataset size: {metrics['dataset_size']}")
    print(f"Train / test: {metrics['train_size']} / {metrics['test_size']}")
    print(f"Accuracy: {accuracy:.4f}")
    print("Confusion matrix (rows=actual, columns=predicted; labels=[0, 1]):")
    print(matrix)
    print(f"Created {metrics_path.name}")


if __name__ == "__main__":
    main()
