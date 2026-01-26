import pandas as pd

def data_sampling(df,clusters,labels):
    sampled_rows=[]

    for c in clusters:
        sampled=df[df['cluster']==c].sample(n=10,random_state=42)

        for _, row in sampled.iterrows():
            sampled_rows.append({
                "cluster":row["cluster"],
                "label":labels[f"{c}"],
                "text":row["clean_text"]
            })

    new_df=pd.DataFrame(sampled_rows)
    return new_df

print("Data Load")
df1=pd.read_csv("./after_clustering_ds/insurance_embeddings.csv")
df2=pd.read_csv("./after_clustering_ds/medChat_with_embeddings.csv")
label1={"0":"암 보험금 청구 및 지급 절차","1":"보험료 납입 면제 및 계약 유지","4":"암 보험 해지 및 환급","7":"암 보험 보장 범위 및 지급 사유"}
label2={"0":"질문 및 요청","3":"다음 단계 안내","4":"마무리/감사 인사","5":"인사/대화 시작","9":"확인","10":"추가 상담 및 조언"}

print("Sampling Start")
new_df1=data_sampling(df1,[0,1,4,7],label1)
new_df2=data_sampling(df1,[0,3,4,5,9,10],label2)
print("Sampling Finished")

print("Save csv files")
new_df1.to_csv("./insurance_sampling.csv")
new_df2.to_csv("./medChat_sampling.csv")
print("Saved csv files")
