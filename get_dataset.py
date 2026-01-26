from datasets import load_dataset
import pandas as pd

ds = load_dataset("squarelike/ko_medical_chat")
dataset=ds['train']['conversations']
rows = []
for d in dataset:
    for id, content in enumerate(d):
        rows.append({
            "text": content["value"]
        })

df = pd.DataFrame(rows)
print(df.head())

df.to_csv("medicalChat_dataset.csv", index=False, encoding="utf-8-sig")
