import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from .util import CONSTANTS
import re
def clean_code_str(code: str) -> str:
    # 1. 去掉首尾空白
    code = code.strip()
    # 2. 去掉多余空格和制表符，合并多行
    code = re.sub(r'\s+', ' ', code)
    return code



def construct_vector(batch_size=1):
    knowledge_path = CONSTANTS.repository_dir + '/knowledge.csv'
    df = pd.read_csv(knowledge_path)

    texts = df['slicing'].tolist()
    cleaned_slicing = [clean_code_str(text) for text in texts]
    metadatas = df[['diff', 'desc']].to_dict(orient='records')

    embeddings = HuggingFaceEmbeddings(
        model_name="jinaai/jina-embeddings-v2-base-code",
        model_kwargs={"device": "cpu","trust_remote_code": True}
    )

    vectorstore = None
    for i in range(0, len(cleaned_slicing), batch_size):
        batch_texts = cleaned_slicing[i:i + batch_size]
        batch_metas = metadatas[i:i + batch_size]

        # 生成向量（返回 list of list）
        batch_vectors = embeddings.embed_documents(batch_texts)
        text_vector_tuples = list(zip(batch_texts, batch_vectors))
        if vectorstore is None:
            # 第一次创建 FAISS
            vectorstore = FAISS.from_embeddings(text_vector_tuples,embeddings, batch_metas)
        else:
            # 后续批次追加
            vectorstore.add_embeddings(text_vector_tuples, batch_metas)

        print(f"Processed batch {i // batch_size + 1}/{(len(cleaned_slicing) + batch_size - 1) // batch_size}")

    vectorstore.save_local("slicing_index")
    print("Vectorstore saved successfully.")

def construct_clean_vector(batch_size=1):
    knowledge_path = CONSTANTS.repository_dir + '/knowledge_clean.csv'
    df = pd.read_csv(knowledge_path)

    texts = df['slicing'].tolist()
    cleaned_slicing = [clean_code_str(text) for text in texts]
    metadatas = df[['diff']].to_dict(orient='records')

    embeddings = HuggingFaceEmbeddings(
        model_name="jinaai/jina-embeddings-v2-base-code",
        model_kwargs={"device": "cpu","trust_remote_code": True}
    )

    vectorstore = None
    for i in range(0, len(cleaned_slicing), batch_size):
        batch_texts = cleaned_slicing[i:i + batch_size]
        batch_metas = metadatas[i:i + batch_size]

        # 生成向量（返回 list of list）
        batch_vectors = embeddings.embed_documents(batch_texts)
        text_vector_tuples = list(zip(batch_texts, batch_vectors))
        if vectorstore is None:
            # 第一次创建 FAISS
            vectorstore = FAISS.from_embeddings(text_vector_tuples,embeddings, batch_metas)
        else:
            # 后续批次追加
            vectorstore.add_embeddings(text_vector_tuples, batch_metas)

        print(f"Processed batch {i // batch_size + 1}/{(len(cleaned_slicing) + batch_size - 1) // batch_size}")

    vectorstore.save_local("clean_slicing_index")
    print("Vectorstore saved successfully.")

if __name__=='__main__':
    construct_clean_vector()