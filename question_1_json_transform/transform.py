"""Transform Received JSON into the required Result JSON."""

import json
from pathlib import Path
from typing import Any


def transform(data: dict[str, Any]) -> list[dict[str, Any]]:
    genders = {item["gender_id"]: item["label"] for item in data["gender"]}
    cars = {item["car_id"]: item for item in data["cars"]}
    colors = {item["label"]: item for item in data["rgbCode"]}
    countries = {
        int(user["userId"]): country["label"]
        for country in data["countries"]
        for user in country["user_ids"]
    }

    results = []
    for student in data["students"]:
        adored = cars[student["adore_car"]]
        results.append(
            {
                "user_id": student["user_id"],
                "first_name": student["first_name"],
                "last_name": student["last_name"],
                "gender": genders[student["gender_id"]],
                "adore_car": {
                    "car_brand": adored["car_brand"],
                    "brand_from": adored["car_make"],
                },
                "car_brand": [cars[car_id]["car_brand"] for car_id in student["car_brand"]],
                "countries": countries[student["user_id"]],
                "colors": [colors[color] for color in student["color"]],
            }
        )
    return results


if __name__ == "__main__":
    source = Path(__file__).with_name("input.json")
    output = Path(__file__).with_name("result.json")
    results_json = json.dumps(
        transform(json.loads(source.read_text(encoding="utf-8"))), indent=2
    )
    output.write_text(results_json + "\n", encoding="utf-8")
    print(f"Created {output.name}")
