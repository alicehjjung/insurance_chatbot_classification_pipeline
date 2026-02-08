# LLM-based user query classification pipeline
Recent advances in Large Language Models (LLMs) have enabled chatbots to be used not only for general conversations but also as domain-specific consultation tools. In sensitive domains such as insurance, where accuracy and careful information delivery are critical, correctly identifying user query types is essential.   

This project proposes an LLM-based user query classification pipeline for a cancer insurance consultation chatbot. Assuming a label-scarce setting where predefined query categories do not exist, we applied K-Means clustering to insurance policy summaries and medical dialogue datasets to derive meaningful query clusters. Based on these clusters, query categories were defined, and a synthetic dataset was generated using LLMs.   

Using the generated dataset, we built a zero-shot LLM-based query classifier. Experimental results show that categories with clear linguistic patterns achieve relatively strong classification performance, whereas semantically overlapping categories degrade performance.   

This work demonstrates a practical approach to defining query categories and building classification systems using LLMs, even in the absence of clearly labeled datasets.

### Clustering

-**pdf_partitioning.py**
Splits PDF documents into meaningful text chunks for downstream processing.

-**get_dataset.py**
Loads external datasets from Hugging Face.

-**data_preprocessing.py**
Performs text preprocessing tasks such as cleaning and normalization.

-**clustering.py**
Applies the K-Means algorithm to cluster text data.

-**data_sampling.py**
Samples 10 data points from each cluster for analysis and dataset construction.

### Synthetic Data Generation

-**SyntheticData_generation.py**
Generates synthetic data for each cluster using an LLM.

### Text Classification

-**Classifier.py**
Implements an LLM-based zero-shot query classifier.

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
