"""Cosine similarity without external libraries."""

from math import sqrt
from typing import List


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must have the same length.")

    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = sqrt(sum(value * value for value in vec1))
    magnitude2 = sqrt(sum(value * value for value in vec2))
    if magnitude1 == 0 or magnitude2 == 0:
        raise ValueError("Cosine similarity is undefined for a zero vector.")

    return dot_product / (magnitude1 * magnitude2)


if __name__ == "__main__":
    vec1 = [1, 2, 3]
    vec2 = [2, 3, 4]
    result = f"ANS => {cosine_similarity(vec1, vec2):.4f}"
    output_path = __file__.replace("cosine_similarity.py", "output.txt")
    with open(output_path, "w", encoding="utf-8") as output_file:
        output_file.write(result + "\n")
    print(result)
    print("Created output.txt")
