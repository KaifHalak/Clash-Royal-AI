# Download all Clash Royale card images from the official API and save them to a local folder.

import os
import requests

API_KEY = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiIsImtpZCI6IjI4YTMxOGY3LTAwMDAtYTFlYi03ZmExLTJjNzQzM2M2Y2NhNSJ9.eyJpc3MiOiJzdXBlcmNlbGwiLCJhdWQiOiJzdXBlcmNlbGw6Z2FtZWFwaSIsImp0aSI6IjkxOWEyYmRiLTI2M2YtNDY4Mi04ZjkyLTc1M2E2ZDFjMzhkOSIsImlhdCI6MTc4OTkyMzMxNiwic3ViIjoiZGV2ZWxvcGVyLzFiNTA5N2Q3LTc3ZmMtNDE0NC05ZTM2LWJlMWY3MjI4MjFlNCIsInNjb3BlcyI6WyJyb3lhbGUiXSwibGltaXRzIjpbeyJ0aWVyIjoiZGV2ZWxvcGVyL3NpbHZlciIsInR5cGUiOiJ0aHJvdHRsaW5nIn0seyJjaWRycyI6WyIxMjEuMTIyLjI0Ni4yNDAiXSwidHlwZSI6ImNsaWVudCJ9XX0.U9lCjvPvrJ3N4etiqA-zJmtZghIBgpKQeXalDjcBb7Ocz_Ye3hKcYnsQsd-_toR-ghHIgLJFNNE19ub79bbTHg"  

OUT_DIR = "C:/MyStuff/ClashRoyal/all-cards"

os.makedirs(OUT_DIR, exist_ok=True)

resp = requests.get(
    "https://api.clashroyale.com/v1/cards",
    headers={"Authorization": f"Bearer {API_KEY}"},
)

resp.raise_for_status()
cards = resp.json()["items"]

for card in cards:
    name = card["name"].lower().replace(" ", "_").replace(".", "")
    url = card["iconUrls"]["medium"]

    img_data = requests.get(url).content
    with open(os.path.join(OUT_DIR, f"{name}.png"), "wb") as f:
        f.write(img_data)

    print(f"Saved {name}.png")

print(f"Done. {len(cards)} cards saved to {OUT_DIR}/")