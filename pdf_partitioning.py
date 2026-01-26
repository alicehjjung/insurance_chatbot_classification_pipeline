from unstructured.partition.pdf import partition_pdf
import pandas as pd

pdf_path = "dataset.pdf"

elements = partition_pdf(
    filename=pdf_path,
    strategy="fast",
    chunking_strategy="by_title",
    languages=["kor"],
    max_characters=1200,
    new_after_n_chars=900,
    combine_text_under_n_chars=300,
    infer_table_structure=False,
)

rows = []
for i, e in enumerate(elements):
    text = (e.text or "").strip()
    if len(text) < 120:
        continue

    md = getattr(e, "metadata", None)
    page = getattr(md, "page_number", None) if md else None

    rows.append({
        "id": i,
        "text": text,
        "label": "",
        "page": page,
        "element_type": e.__class__.__name__,
    })

df = pd.DataFrame(rows)
df.to_csv("./dataset/insurance_dataset.csv", index=False, encoding="utf-8-sig")
print("saved:", "insurance_dataset.csv", "rows:", len(df))