"""Cosine similarity using NumPy."""

from typing import List

import numpy as np


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must have the same length.")

    a = np.array(vec1, dtype=float)
    b = np.array(vec2, dtype=float)

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        raise ValueError("Cosine similarity is undefined for a zero vector.")

    return float(np.dot(a, b) / (norm_a * norm_b))


if __name__ == "__main__":
    vec1 = [1, 2, 3]
    vec2 = [2, 3, 4]
    result = f"ANS => {cosine_similarity(vec1, vec2):.4f}"
    output_path = __file__.replace("cosine_similarity.py", "output.txt")
    with open(output_path, "w", encoding="utf-8") as output_file:
        output_file.write(result + "\n")
    print(result)
    print("Created output.txt")
