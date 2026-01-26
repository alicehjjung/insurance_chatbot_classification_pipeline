import os
import pandas as pd
from google import genai
from dotenv import load_dotenv
from google import genai
import pandas as pd
from sklearn.metrics import classification_report
from vertexai.generative_models import GenerationConfig, GenerativeModel
from tqdm import tqdm
import re

# Gemini Model
MODEL_NAME = "gemini-2.5-pro" 
#MODEL_NAME = "gemini-2.5-flash" 
#MODEL_NAME = "gemini-2.5-flash-lite"

def get_client():
    #Set Model
    load_dotenv()
    PROJECT_ID = os.getenv("PROJECT_ID")
    LOCATION = os.getenv("LOCATION")
    return genai.Client(vertexai=True,project=PROJECT_ID,location=LOCATION)

generation_model = GenerativeModel(MODEL_NAME)
generation_config = GenerationConfig(temperature=0.1, max_output_tokens=256)

def classification(row):
    prompt = f"""
        너는 암보험 상담 챗봇의 유형분류기야.
        아래에 주어진 고객의 질문을 읽고, 반드시 아래 Label 중 하나만 선택해서 출력해.
        설명, 이유, 추가 문장은 절대 쓰지 말고 Label 이름만 그대로 출력해.
        출력에는 한국어 Label만 포함해야하고, 영어, 괄호, 설명을 쓰지마.

        [Label 목록]
        - 암 보험금 청구 및 지급 절차
        - 보험료 납입 면제 및 계약 유지
        - 암 보험 해지 및 환급
        - 암 보험 보장 범위 및 지급 사유
        - 질문 및 요청
        - 다음 단계 안내
        - 마무리 및 감사 인사
        - 인사 및 대화 시작
        - 확인
        - 추가 상담 및 조언
        - 보험 종류 소개
        - 고객 정보
        - 보험료

        [고객 질문]
        {row}

        [출력 형식]
        Label
        """

    response = generation_model.generate_content(
        contents=prompt, generation_config=generation_config
    ).text
    response = re.sub(
                r"[^가-힣\s]",
                "",
                re.split(r"\s*\(|\s*[-–—]\s*", response.strip())[0]
            ).strip()
    return response

#Get Dataset
print("Data Load")
df=pd.read_csv("testset.csv")
print("Prediction start\n")

#df["label_prediction"] = df["text"].progress_apply(classification)

df["label_prediction"] = [
    classification(text)
    for text in tqdm(df["text"], desc="Classifying")
]

print(df.head())
print("Prediction Finished\n")

report = classification_report(
    df["label"], df["label_prediction"]
)
print(report)
with open("report.txt", "w") as file:
    file.write(report)
df.to_csv("output.csv")