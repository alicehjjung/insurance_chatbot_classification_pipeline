import re
import pandas as pd

#path = "./dataset/insurance_dataset.csv"
path="./medicalChat_dataset.csv"
df = pd.read_csv(path)

def clean_text1(t: str) -> str:
    if not isinstance(t, str):
        return ""
    t = t.strip()

    # 페이지 표기 제거: "2 / 166" 등
    t = re.sub(r"\b\d+\s*/\s*\d+\b", " ", t)

    # 조항/관/항/호 표기 제거
    t = re.sub(r"제\s*\d+\s*관", " ", t)
    t = re.sub(r"제\s*\d+\s*조", " ", t)
    t = re.sub(r"제\s*\d+\s*항", " ", t)
    t = re.sub(r"[\u2460-\u2473]", " ", t)

    # STEP 절차 블록 제거
    t = re.sub(r"(STEP\s*.*?)(?=STEP|$)", " ", t)

    # 의미 숫자 정규화
    t = re.sub(r"\d{1,3}(,\d{3})+만원", " [AMOUNT] ", t)
    t = re.sub(r"\d+만원", " [AMOUNT] ", t)
    t = re.sub(r"\d+\s*년", " [YEAR] ", t)
    t = re.sub(r"\d+\s*개월", " [MONTH] ", t)
    t = re.sub(r"\d+\s*일", " [DAY] ", t)
    t = re.sub(r"\d+(\.\d+)?\s*%", " [PERCENT] ", t)

    # 남은 순수 숫자는 제거
    t = re.sub(r"\d+", " ", t)
    t = re.sub(r"\b(I{1,3}|IV|V|VI{0,3}|IX|X)\s*~\s*\1\b", " ", t)

    # 회사명/문서명 같은 반복 헤더 후보
    t = re.sub(r"(한화생명|보험약관|가이드북)", " ", t)

    # () 또는 (   ) 제거
    t = re.sub(r"\(\s*\)", " ", t)

    # . 이 연속해서 두번 이상 나오는 경우
    t = re.sub(r"\.\s+\.", ".", t)

    # C ~C, I ~I
    t = re.sub(r"\b([A-Z])\s*~\s*\1\b", " ", t)

    # 공백 정리
    t = re.sub(r"\s+", " ", t).strip()

    return t

def clean_text2(t: str) -> str:
    if not isinstance(t, str):
        return ""
    t = t.strip()
    t = re.sub(r'[\"“”″＂]', '', t)

    return t

#df["clean_text"] = df["text"].apply(clean_text1)
df["clean_text"] = df["text"].apply(clean_text2)

# 짧은 조각 제거
#min_len = 120
#df = df[df["clean_text"].str.len() >= min_len].copy()

# 중복 제거
df = df.drop_duplicates(subset=["clean_text"]).reset_index(drop=True)

#out = df[["page", "clean_text"]]
#out.to_csv("clustering_dataset.csv", index=False, encoding="utf-8-sig")
out = df[["clean_text"]]
out.to_csv("clustering_medChat.csv", index=False, encoding="utf-8-sig")
print(out.head())
print(out.shape)
