"""Cosine similarity using NumPy.

โปรแกรมนี้คำนวณค่าความคล้ายคลึงเชิงโคไซน์ (cosine similarity)
ระหว่างเวกเตอร์สองตัว โดยใช้ numpy สำหรับการคำนวณ dot product และ norm
"""

from typing import List

import numpy as np


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    # เวกเตอร์ทั้งสองต้องมีความยาวเท่ากัน มิฉะนั้นจะไม่สามารถคำนวณ dot product ได้
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must have the same length.")

    a = np.array(vec1, dtype=float)
    b = np.array(vec2, dtype=float)

    # คำนวณความยาว (L2 norm) ของแต่ละเวกเตอร์
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    # หากเวกเตอร์ใดเวกเตอร์หนึ่งเป็นเวกเตอร์ศูนย์ ค่า cosine similarity จะไม่นิยาม
    if norm_a == 0 or norm_b == 0:
        raise ValueError("Cosine similarity is undefined for a zero vector.")

    # cosine similarity = (a · b) / (||a|| * ||b||)
    return float(np.dot(a, b) / (norm_a * norm_b))


if __name__ == "__main__":
    # ตัวอย่างเวกเตอร์สำหรับทดสอบ
    vec1 = [1, 2, 3]
    vec2 = [2, 3, 4]
    result = f"ANS => {cosine_similarity(vec1, vec2):.4f}"

    # เขียนผลลัพธ์ลงไฟล์ output.txt ในโฟลเดอร์เดียวกับไฟล์นี้
    output_path = __file__.replace("cosine_similarity.py", "output.txt")
    with open(output_path, "w", encoding="utf-8") as output_file:
        output_file.write(result + "\n")
    print(result)
    print("Created output.txt")
