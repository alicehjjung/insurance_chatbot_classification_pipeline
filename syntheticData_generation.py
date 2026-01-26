import os
import pandas as pd
from google import genai
from dotenv import load_dotenv
from google import genai

# Gemini Model
MODEL_NAME = "gemini-2.5-pro" 

def get_client():
    #Set Model
    load_dotenv()
    PROJECT_ID = os.getenv("PROJECT_ID")
    LOCATION = os.getenv("LOCATION")
    return genai.Client(vertexai=True,project=PROJECT_ID,location=LOCATION)

def get_sample_dataset(df,cluster):
    lines = [f"[EXAMPLES for cluster {cluster}]"]
    for i, text in enumerate(df["text"].tolist(), 1):
        lines.append(f"{i}. {text}")
    return "\n".join(lines)

def generate_dataset_gemini(labels,sample,n_per_label):

    client = get_client()

    rows = []

    for lab in labels:
        print(f"{lab} Start")
        if sample is None:
            prompt = f"""
            암보험에 가입하려는 고객을 상담하는 챗봇을 생각해보자. 고객이 챗봇에 물어볼 수 있는 질문들을 유형을 분류(Classify)하려고해. 
            따라서, 고객이 물어볼만한 내용을 라벨에 맞게 지정해준 생성 개수만큼 생성해줘.

            [라벨]
            {lab}

            [생성 개수]
            {n_per_label}개

            [규칙]
            - 한국어로 생성
            - 1~3문장 길이
            - 유사한 문장 반복 금지
            - 실제 암보험 상담/약관/FAQ를 통해 질문할 수 있는 내용으로 생성
            - 각 항목은 '한 줄'로 출력 (줄바꿈으로 구분)
            - 번호/불릿/따옴표 없이 출력
            """.strip()
        else:
            samples=get_sample_dataset(sample,lab)
            prompt = f"""
            암보험에 가입하려는 고객을 상담하는 챗봇을 생각해보자. 고객이 챗봇에 물어볼 수 있는 질문들을 유형을 분류(Classify)하려고해. 
            따라서, 고객이 물어볼만한 내용을 아래 Sample을 이용해서 라벨에 맞게 지정해준 생성 개수만큼 생성해줘.

            [라벨]
            {lab}

            [생성 개수]
            {n_per_label}개

            [Sample]
            {samples}

            [규칙]
            - 한국어로 생성
            - 1~3문장 길이
            - 유사한 문장 반복 금지
            - 실제 암보험 상담/약관/FAQ를 통해 질문할 수 있는 내용으로 생성
            - 각 항목은 '한 줄'로 출력 (줄바꿈으로 구분)
            - 번호/불릿/따옴표 없이 출력
            """.strip()

        resp = client.models.generate_content(
            model="gemini-2.5-pro" ,
            contents=prompt,
        )
        print("Extracting")

        lines = [x.strip() for x in (resp.text or "").splitlines() if x.strip()]

        cleaned = []
        for s in lines:
            s2 = s.lstrip("0123456789. )-•\t")
            s2 = s2.strip()
            if s2:
                cleaned.append(s2)

        cleaned = cleaned[:n_per_label]

        if len(cleaned) < n_per_label:
            print(f"[WARN] label='{lab}': generated {len(cleaned)}/{n_per_label}")

        for t in cleaned:
            rows.append({"label": lab, "text": t})
        print(f"{lab} Finished")

    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    LABELS1 = ["암 보험금 청구 및 지급 절차","보험료 납입 면제 및 계약 유지","암 보험 해지 및 환급","암 보험 보장 범위 및 지급 사유"]
    LABELS2 = ["질문 및 요청","다음 단계 안내","마무리/감사 인사","인사/대화 시작","확인","추가 상담 및 조언"]
    LABELS3 = ["보험 종류 소개","고객 정보","보험료"]

    print("Generate1")
    df1 = generate_dataset_gemini(
        labels=LABELS1,
        n_per_label=10,
        sample=pd.read_csv("insurance_sampling.csv"),
    )

    print("Generate2")
    df2 = generate_dataset_gemini(
        labels=LABELS2,
        n_per_label=10,
        sample=pd.read_csv("medChat_sampling.csv"),
    )

    print("Generate3")
    df3 = generate_dataset_gemini(
        labels=LABELS3,
        n_per_label=10,
        sample=None
    )

    result = pd.concat([df1, df2, df3])

    print(result.head())
    result.to_csv("testset.csv")
    print("Done")
