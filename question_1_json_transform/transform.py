"""Transform Received JSON into the required Result JSON.

โปรแกรมนี้แปลงข้อมูล JSON ดิบจากหลายตาราง (gender, cars, rgbCode, countries, students)
ให้กลายเป็น JSON รูปแบบที่ต้องการ โดยใช้ pandas จับคู่ (map/lookup) ข้อมูลระหว่างตาราง
"""

import json
from pathlib import Path
from typing import Any

import pandas as pd


def transform(data: dict[str, Any]) -> list[dict[str, Any]]:
    # สร้าง lookup table จากตารางย่อย เพื่อให้สามารถดึงค่าตาม id/label ได้เร็ว
    genders = pd.DataFrame(data["gender"]).set_index("gender_id")["label"]
    cars = pd.DataFrame(data["cars"]).set_index("car_id")
    colors = pd.DataFrame(data["rgbCode"]).set_index("label")

    # สร้างตาราง mapping user_id -> country จากรายการประเทศและผู้ใช้ในประเทศนั้น
    countries = pd.DataFrame(
        [
            {"user_id": int(user["userId"]), "country": country["label"]}
            for country in data["countries"]
            for user in country["user_ids"]
        ]
    ).set_index("user_id")["country"]

    # แปลงตารางนักเรียนให้ตรงตามโครงสร้าง output ที่ต้องการ
    students = pd.DataFrame(data["students"])
    students["gender"] = students["gender_id"].map(genders)  # แปลง gender_id -> ชื่อ gender
    students["adore_car"] = students["adore_car"].map(
        # ดึงข้อมูลรถคันโปรดของนักเรียน โดยระบุทั้งยี่ห้อและประเทศผู้ผลิต
        lambda car_id: {
            "car_brand": cars.at[car_id, "car_brand"],
            "brand_from": cars.at[car_id, "car_make"],
        }
    )
    students["car_brand"] = students["car_brand"].map(
        # แปลงรายการ car_id ของรถที่ชอบทั้งหมด เป็นรายชื่อยี่ห้อ
        lambda car_ids: [cars.at[car_id, "car_brand"] for car_id in car_ids]
    )
    students["colors"] = students["color"].map(
        # แปลงรายการชื่อสีเป็นรายการ object ที่มีทั้งชื่อและค่า RGB
        lambda labels: [colors.loc[label].to_dict() for label in labels]
    )
    students["countries"] = students["user_id"].map(countries)  # ดึงประเทศของผู้ใช้

    return (
        students[
            [
                "user_id",
                "first_name",
                "last_name",
                "gender",
                "adore_car",
                "car_brand",
                "countries",
                "colors",
            ]
        ]
        .to_dict(orient="records")
    )


if __name__ == "__main__":
    # อ่าน input.json จากโฟลเดอร์เดียวกับไฟล์นี้ แล้วเขียนผลลัพธ์ลง result.json
    source = Path(__file__).with_name("input.json")
    output = Path(__file__).with_name("result.json")
    results_json = json.dumps(
        transform(json.loads(source.read_text(encoding="utf-8"))), indent=2
    )
    output.write_text(results_json + "\n", encoding="utf-8")
    print(f"Created {output.name}")
