# insurance_chatbot_classification_pipeline
## 암 보험 상담 챗봇을 위한 LLM 기반 고객 질의 유형 분류 파이프라인

### Clustering
- **pdf_partitioning.py**  
  PDF 파일을 의미 있는 단위의 텍스트 청크로 분할하는 파일

- **get_dataset.py**  
  Hugging Face로부터 외부 데이터셋을 로드하는 파일

- **data_preprocessing.py**  
  텍스트 정제, 정규화 등 전처리를 수행하는 파일

- **Clustering.py**  
  K-Means 알고리즘을 이용해 텍스트 데이터를 군집화하는 파일

- **data_sampling.py**  
  각 군집(cluster)에서 10개씩 샘플 데이터를 추출하는 파일


### Synthetic Data Generation
- **SyntheticData_generation.py**  
  LLM을 활용해 군집별 합성 데이터를 생성하는 파일


### Text Classification
- **Classifier.py**  
  LLM 기반 zero-shot 방식의 유형 분류기
