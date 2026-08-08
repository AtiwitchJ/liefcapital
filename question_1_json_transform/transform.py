"""Transform Received JSON into the required Result JSON."""

import json
from pathlib import Path
from typing import Any

import pandas as pd


def transform(data: dict[str, Any]) -> list[dict[str, Any]]:
    genders = pd.DataFrame(data["gender"]).set_index("gender_id")["label"]
    cars = pd.DataFrame(data["cars"]).set_index("car_id")
    colors = pd.DataFrame(data["rgbCode"]).set_index("label")
    countries = pd.DataFrame(
        [
            {"user_id": int(user["userId"]), "country": country["label"]}
            for country in data["countries"]
            for user in country["user_ids"]
        ]
    ).set_index("user_id")["country"]

    students = pd.DataFrame(data["students"])
    students["gender"] = students["gender_id"].map(genders)
    students["adore_car"] = students["adore_car"].map(
        lambda car_id: {
            "car_brand": cars.at[car_id, "car_brand"],
            "brand_from": cars.at[car_id, "car_make"],
        }
    )
    students["car_brand"] = students["car_brand"].map(
        lambda car_ids: [cars.at[car_id, "car_brand"] for car_id in car_ids]
    )
    students["colors"] = students["color"].map(
        lambda labels: [colors.loc[label].to_dict() for label in labels]
    )
    students["countries"] = students["user_id"].map(countries)

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
    source = Path(__file__).with_name("input.json")
    output = Path(__file__).with_name("result.json")
    results_json = json.dumps(
        transform(json.loads(source.read_text(encoding="utf-8"))), indent=2
    )
    output.write_text(results_json + "\n", encoding="utf-8")
    print(f"Created {output.name}")
