# query_rag.py
import asyncio
import re
import traceback
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from tqdm import tqdm
from util.construc_vector import clean_code_str
from util.process_commit import parse_divergent_data, prepare_meta,prepare_meta_not_divergent,prepare_meta_not_slice,writegraph
from langchain_huggingface import HuggingFaceEmbeddings
import ast
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import os
import openai
import numpy as np
from dotenv import load_dotenv
from tqdm.asyncio import tqdm_asyncio
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from util.util import CONSTANTS
import random
import logging

logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    filename='err.log',  # 日志写入文件（不输出到控制台）
    filemode='a'         # 追加模式
)

load_dotenv()
model = SentenceTransformer("BAAI/bge-base-en-v1.5")
def clean_input(text):
    text = re.sub(r'\s+', ' ', text)
    # 去掉敏感 token-like 内容
    text = re.sub(r'([A-Za-z0-9]{20,})', '[REDACTED]', text)
    return text
async def query_rag(datas):
    prompt = ChatPromptTemplate.from_messages([("system","""You are an expert Java engineer and static code reviewer. 
You will be given:
1)A program slice (PROGRAM_SLICE) related to the diff, including only the newly added code and any surrounding context necessary to reason about the change.**The diff lines are start with "+"**.
2)A similar example bug diff (BUG_EXAMPLE_DIFF) that previously introduced a bug, 
3)A short human-provided explanation (EXAMPLE_REASON) describing why that BUG_EXAMPLE_DIFF caused a bug.
4)A clean example diff(CLEAN_EXAMPLE_DIFF) that does not introduce any bugs.
                                                
Task:
- Understand the diff in the program slice.
- Analyze whether the NEW diff **introduces a bug** in the program slice or the broader code context implied by the slice.
- Focus only on Java-specific bugs.
- Use the BUG_EXAMPLE_DIFF , EXAMPLE_REASON,CLEAN_EXAMPLE_DIFF as a hint for patterns to look for,**but do not assume the same root cause unless evidence supports it**.

---**Important Note**---
- The PROGRAM_SLICE may be incomplete or truncated; missing definitions should **not** be treated as errors.
- Your judgment of “introduces_bug” should be based on **the logic of the diff itself**, **not on missing context** from the slice.
- The removed code in the diff are hidden.

---Required OUTPUTS (produce valid JSON and do not include any other markdown, comments, or formatting.)---

{{
  "introduces_bug": "yes" | "no",
  "bug_summary": "<one-sentence high-level summary if yes, else empty string>",
  "evidence": [
    {{
      "diff_code": "<exact line(s) from the diff that introduce or suggest the bug>",
      "reason": "<why this snippet supports the root cause>"
    }}
  ],
  "confidence": "<low|medium|high>",
}}

Formatting rules and guidance:
- ALWAYS return **only** the **JSON STRING** above (no extra explanation).
- If you answer "no" for introduces_bug, still fill the JSON fields reasonably.
    """),("human","""
Now analyze the inputs below.

---INPUTS---
{DATAS}

---END---
    """)])
    datas=clean_input(datas)
    llm = ChatOpenAI(
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        model="qwen-max",
        temperature=0.2
    )
    result = await (prompt | llm).ainvoke({"DATAS":datas})
    return result.content


async def query_not_rag(datas):
    prompt = ChatPromptTemplate.from_messages([("system","""You are an expert Java engineer and static code reviewer. 
You will be given:
1)A program slice (PROGRAM_SLICE) related to the diff, including only the newly added code and any surrounding context necessary to reason about the change.**The diff lines are start with "+"**.

Task:
- Understand the change, and relate it to the program slice.
- Analyze whether the NEW diff **introduces a bug** in the program slice or the broader code context implied by the slice.
- Focus only on Java-specific bugs.            
---Important Note---
- The PROGRAM_SLICE may be incomplete or truncated; missing definitions should **not** be treated as errors.
- Your judgment of “introduces_bug” should be based on the logic of the diff itself, not on missing context from the slice.
- The removed code in the diff are hidden.

------Required OUTPUTS (produce valid JSON and do not include any other markdown, comments, or formatting.)------

{{
  "introduces_bug": "yes" | "no",
  "bug_summary": "<one-sentence high-level summary if yes, else empty string>",
  "evidence": [
    {{
      "diff_code": "<exact line(s) from the diff that introduce or suggest the bug>",
      "reason": "<why this snippet supports the root cause>"
    }}
  ],
  "confidence": "<low|medium|high>",
}}

Formatting rules and guidance:
- ALWAYS return **only** the **JSON STRING** above (no extra explanation).
- If you answer "no" for introduces_bug, still fill the JSON fields reasonably.
"""),("human","""
Now analyze the inputs below.

---INPUTS---
{DATAS}

---END---
    """)])
    print(datas)
    llm = ChatOpenAI(
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        model="qwen-max",
        temperature=0
    )
    result = await (prompt | llm).ainvoke({"DATAS":datas})
    return result.content

async def query_not_rag_slice(datas):
    prompt = ChatPromptTemplate.from_messages([("system","""You are an expert Java engineer and static code reviewer. 
You will be given:
1) A Java commit diff (DIFF) that should be analyzed.

Task:
- Understand the change.
- Analyze whether the NEW diff **introduces a bug** .
- Focus only on Java-specific bugs.          
------Required OUTPUTS (produce valid JSON and do not include any other markdown, comments, or formatting.)------

{{
  "introduces_bug": "yes" | "no",
  "bug_summary": "<one-sentence high-level summary if yes, else empty string>",
  "evidence": [
    {{
      "diff_code": "<exact line(s) from the diff that introduce or suggest the bug>",
      "reason": "<why this snippet supports the root cause>"
    }}
  ],
  "confidence": "<low|medium|high>"
}}

Formatting rules and guidance:
- ALWAYS return **only** the **JSON STRING** above (no extra explanation).
- If you answer "no" for introduces_bug, still fill the JSON fields reasonably.
"""),("human","""
Now analyze the inputs below.

---INPUTS---
{DATAS}

---END---
    """)])

    llm = ChatOpenAI(
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        model="qwen-max",
        temperature=0.2
    )
    result = await (prompt | llm).ainvoke({"DATAS":datas})
    return result.content

def similar_slice(meta_datas):
    embeddings = HuggingFaceEmbeddings(
        model_name="jinaai/jina-embeddings-v2-base-code",
        model_kwargs={"device": "cpu","trust_remote_code": True}
    )
    vectorstore = FAISS.load_local("slicing_index", embeddings,allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":20})
    
    clean_vectorstore = FAISS.load_local("clean_slicing_index", embeddings,allow_dangerous_deserialization=True)
    clean_retriever = clean_vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":20})

    all_datas=[]
    for meta_data_group in meta_datas:
        group_datas=[]
        for meta_data in meta_data_group:
          #只有添加代码才会引入bug
          if(len(meta_data["add_codes"]))==0:
              continue
          PATH=meta_data["file_path"]
          PROGRAM_SLICE=meta_data["slices"]
          clean_slicing=clean_code_str(PROGRAM_SLICE)
          PROGRAM_SLICE=PROGRAM_SLICE.replace("\\n","\n")

          BUG_EXAMPLE_DIFF="There are not EXAMPLE DIFF"
          EXAMPLE_REASON="There are not EXAMPLE REASON"
          CLEAN_EXAMPLE_DIFF="There are not CLEAN EXAMPLE DIFF"
          if len(clean_slicing)<=15000:
            # 检索相似 slicing
            retrieved_docs = retriever.invoke(clean_slicing)
            add_code=clean_code_str("".join(s[1:] for s in meta_data["add_codes"]))
            max_similar=0
            for similar_doc in retrieved_docs:
                reference_diff=clean_code_str("".join(s[1:] for s in ast.literal_eval(similar_doc.metadata["diff"])))
                if len(reference_diff)>5000:
                    continue
                similar_score=calculate_similar(reference_diff,add_code)
                if similar_score>max_similar:
                    max_similar=similar_score
                    BUG_EXAMPLE_DIFF="".join(ast.literal_eval(similar_doc.metadata["diff"]))
                    EXAMPLE_REASON=similar_doc.metadata["desc"]
            clean_retrieved_docs = clean_retriever.invoke(clean_slicing)
            max_similar=0
            for similar_doc in clean_retrieved_docs:
                reference_diff=clean_code_str("".join(s[1:] for s in ast.literal_eval(similar_doc.metadata["diff"])))
                if len(reference_diff)>5000:
                    continue
                similar_score=calculate_similar(reference_diff,add_code)
                if similar_score>max_similar:
                    max_similar=similar_score
                    CLEAN_EXAMPLE_DIFF="".join(ast.literal_eval(similar_doc.metadata["diff"]))
                    
          DATA=f'''PATH:{PATH},
            PROGRAM_SLICE:\n"{PROGRAM_SLICE}"
            BUG_EXAMPLE_DIFF:\n"{BUG_EXAMPLE_DIFF}"
            EXAMPLE_REASON:\n"{EXAMPLE_REASON}"
            CLEAN_EXAMPLE_DIFF:\n"{CLEAN_EXAMPLE_DIFF}"
              '''
          group_datas.append(DATA)
        datas="".join(group_datas)
        all_datas.append(datas)

    return all_datas

def calculate_similar(text1,text2):
    """
    计算两个文本的相似度，综合 jaccard 与 Embedding 语义相似度
    返回值范围约为 [0, 1]
    """

    txt1=set(text1.split())
    txt2=set(text2.split())
    jaccard_score=len(txt1&txt2)/len(txt1|txt2)

    emb1 = model.encode(text1, normalize_embeddings=True)
    emb2 = model.encode(text2, normalize_embeddings=True)
    cosine_sim = float(util.cos_sim(emb1, emb2))
  
    alpha = 0.8
    final_score = alpha * cosine_sim + (1 - alpha) * jaccard_score
    #print(final_score)
    return final_score

async def predict_buggy(project,parent_hash,commit_hash,data_get_func=similar_slice,query_func=query_rag,meta_function=prepare_meta):
    meta_datas = meta_function(project, parent_hash, commit_hash)
    all_datas = data_get_func(meta_datas)
    sem = asyncio.Semaphore(5)  
    async def run_single(datas):
        async with sem:
            res = await query_func(datas)
            match = re.search(r'"introduces_bug"\s*:\s*"([^"]+)"', res)
            if match:
                introduces_bug = match.group(1)
                if "yes" in introduces_bug:
                    return res
            else:
                logging.error(f"{res} 未找到 introduces_bug 字段")
            return None

    # 并发执行所有任务
    tasks = [asyncio.create_task(run_single(datas)) for datas in all_datas]
    results = await asyncio.gather(*tasks)

    # 筛选出有效结果
    valid_results = [r for r in results if r is not None]
    if valid_results:
        return True, "|".join(valid_results)
    return False, ""

SEM = asyncio.Semaphore(1)
@retry(
    stop=stop_after_attempt(5),                # 最多重试5次
    wait=wait_exponential(multiplier=1, min=2, max=30),  # 指数退避
    retry=retry_if_exception_type(openai.RateLimitError)
)
async def safe_generate_prediction(project, parent_hash, commit_hash,data_get_func=similar_slice,query_func=query_rag,meta_function=prepare_meta):
    async with SEM:  # 控制并发
        buggy, res = await predict_buggy(project, parent_hash, commit_hash,data_get_func,query_func,meta_function)
        await asyncio.sleep(random.uniform(0.5, 1.5))  # 在请求间随机等待，减少撞限
        return buggy, res
            
async def quickly_start():
    res_path=CONSTANTS.repository_dir+'/predictions.csv'
    df=pd.read_csv(res_path)
    df['predicted_result'] = df['predicted_result'].astype(object)
    df['output'] = df['output'].astype(object)
    async def process_one(index, row):
        project = row['project']
        parent_hash = row['parent_hashes']
        commit_hash = row['commit_hash']
        buggy=row['predicted_result']
        reason=row['output']
        ground_truth=row["is_buggy_commit"]==1.0
        if pd.notna(buggy) and str(buggy).strip() != "":
            return index, buggy, reason

        try:
            buggy, reason = await safe_generate_prediction(project, parent_hash, commit_hash)
            return index, buggy, reason
        except Exception as e:
            traceback.print_exc()
            logging.error(e)
            return index, None, None
        
    try:
      tasks = [process_one(index, row) for index, row in df.iterrows()]
      for coro in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc="Generating ..."):
        index, buggy, reason = await coro
        if buggy is not None:
            df.at[index, 'predicted_result'] = 1.0 if buggy else 0.0
            df.at[index, 'output'] = str(reason).replace("\n","").replace("\r","") if reason else ""
        if index%10==0:
            df.to_csv(res_path, index=False)

    finally:
        df.to_csv(res_path, index=False)

def generate_divergent_output():
    res_path=CONSTANTS.repository_dir+'/predictions.csv'
    df=pd.read_csv(res_path)
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        project = row['project']
        parent_hash = row['parent_hashes']
        commit_hash = row['commit_hash']
        parse_divergent_data(project,parent_hash,commit_hash)

def process_not_rag(meta_datas):
    all_datas=[]
    for meta_data_group in meta_datas:
        group_datas=[]
        for meta_data in meta_data_group:
          #只有添加代码才会引入bug
          if(len(meta_data["add_codes"]))==0:
              continue
          PATH=meta_data["file_path"]
          DIFF='\n'.join(meta_data["remove_codes"])+'\n'.join(meta_data["add_codes"])
          PROGRAM_SLICE=meta_data["slices"].replace("\\n","\n")
          if len('\n'.join(meta_data["add_codes"]))>=len(PROGRAM_SLICE)*0.9:
              PROGRAM_SLICE="There are not PROGRAM_SLICE. DIFF is enought"

          DATA=f'''*PATH*:{PATH},
            *DIFF*:\n"{DIFF}"
            PROGRAM_SLICE:\n"{PROGRAM_SLICE}"
            '''
          group_datas.append(DATA)
        datas="".join(group_datas)
        all_datas.append(datas)

    return all_datas

async def quickly_start_not_rag():
    res_path=CONSTANTS.repository_dir+'/predictions.csv'
    df=pd.read_csv(res_path)
    df['result_not_rag'] = df['result_not_rag'].astype(object)
    df['not_rag_output'] = df['not_rag_output'].astype(object)
    async def process_one(index, row):
        project = row['project']
        parent_hash = row['parent_hashes']
        commit_hash = row['commit_hash']
        buggy=row['result_not_rag']
        reason=row['not_rag_output']
        if pd.notna(buggy) and str(buggy).strip() != "":
            return index, buggy, reason
        try:
            buggy, reason = await safe_generate_prediction(project, parent_hash, commit_hash,data_get_func=process_not_rag,query_func=query_not_rag)
            return index, buggy, reason
        except Exception as e:
            traceback.print_exc()
            logging.error(e)
            return index, None, None
        
    try:
      tasks = [process_one(index, row) for index, row in df.iterrows()]
      for coro in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc="Generating Description..."):
        index, buggy, reason = await coro
        if buggy is not None:
            df.at[index, 'result_not_rag'] = 1.0 if buggy else 0.0
            df.at[index, 'not_rag_output'] = str(reason).replace("\n","").replace("\r","") if reason else ""
    finally:
        df.to_csv(res_path, index=False)

def process_not_rag_slice(meta_datas):
    all_datas=[]
    for meta_data_group in meta_datas:
        group_datas=[]
        for meta_data in meta_data_group:
          #只有添加代码才会引入bug
          if(len(meta_data["add_codes"]))==0:
              continue
          PATH=meta_data["file_path"]
          DIFF='\n'.join(meta_data["remove_codes"])+'\n'.join(meta_data["add_codes"])

          DATA=f'''PATH:{PATH},
            *DIFF*:\n"{DIFF}"
            '''
          group_datas.append(DATA)
        datas="".join(group_datas)
        all_datas.append(datas)

    return all_datas

async def quickly_start_not_rag_slice():
    res_path=CONSTANTS.repository_dir+'/predictions.csv'
    df=pd.read_csv(res_path)
    df['result_not_rag_slice'] = df['result_not_rag_slice'].astype(object)
    df['not_rag_sclice_output'] = df['not_rag_sclice_output'].astype(object)
    async def process_one(index, row):
        project = row['project']
        parent_hash = row['parent_hashes']
        commit_hash = row['commit_hash']
        buggy=row['result_not_rag_slice']
        reason=row['not_rag_sclice_output']
        
        if pd.notna(buggy) and str(buggy).strip() != "":
            return index, buggy, reason

        try:
            buggy, reason = await safe_generate_prediction(project, parent_hash, commit_hash,data_get_func=process_not_rag_slice,query_func=query_not_rag_slice,meta_function=prepare_meta_not_slice)
            return index, buggy, reason
        except Exception as e:
            traceback.print_exc()
            logging.error(e)
            return index, None, None
        
    try:
      tasks = [process_one(index, row) for index, row in df.iterrows()]
      for coro in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc="Generating Description..."):
        index, buggy, reason = await coro
        if buggy is not None:
            df.at[index, 'result_not_rag_slice'] = 1.0 if buggy else 0.0
            df.at[index, 'not_rag_sclice_output'] = str(reason).replace("\n","").replace("\r","") if reason else ""
    finally:
        df.to_csv(res_path, index=False)


async def quickly_start_not_divergent():
    res_path=CONSTANTS.repository_dir+'/predictions.csv'
    df=pd.read_csv(res_path)
    df['result_not_divergent'] = df['result_not_divergent'].astype(object)
    df['not_divergent_output'] = df['not_divergent_output'].astype(object)
    async def process_one(index, row):
        project = row['project']
        parent_hash = row['parent_hashes']
        commit_hash = row['commit_hash']
        buggy=row['result_not_divergent']
        reason=row['not_divergent_output']
        ground_truth=row["is_buggy_commit"]==1.0
        if pd.notna(buggy) and str(buggy).strip() != "":
            return index, buggy, reason
        try:
            buggy, reason = await safe_generate_prediction(project, parent_hash, commit_hash,meta_function=prepare_meta_not_divergent)
            return index, buggy, reason
        except Exception as e:
            traceback.print_exc()
            logging.error(e)
            return index, None, None
        
    try:
      tasks = [process_one(index, row) for index, row in df.iterrows()]
      for coro in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc="Generating Description..."):
        index, buggy, reason = await coro
        if buggy is not None:
            df.at[index, 'result_not_divergent'] = 1.0 if buggy else 0.0
            df.at[index, 'not_divergent_output'] = str(reason).replace("\n","").replace("\r","") if reason else ""
    finally:
        df.to_csv(res_path, index=False)


def clean_prediction():
    res_path=CONSTANTS.repository_dir+'/predictions.csv'
    df=pd.read_csv(res_path)
    df['predicted_result']=""
    df["output"]=""
    df["result_not_rag"]=""
    df['not_rag_output']=""
    df['result_not_rag_slice']=""
    df['not_rag_sclice_output']=""
    df['result_not_divergent']=""
    df['not_divergent_output']=""
    df.to_csv(res_path, index=False)

from util.process_commit import clean_line_construc
from util.construc_vector import construct_clean_vector
if __name__ == "__main__":
    writegraph()

    #clean_prediction()
    # asyncio.run(quickly_start())
    # generate_divergent_output()
    #a,res=asyncio.run(predict_buggy('commons-math','3491906594b069a042f38e7526ca67f46efac2f4','bb014c3ebfe20bf7a7fce5e7a8124121de695350',query_func=query_not_rag_slice,data_get_func=process_not_rag_slice))
    # a,res=asyncio.run(predict_buggy('commons-jcs','d4ed250a0f508e8e3477342d00b090ee3858a40b','67f0a2dca2686b5b740273fb9f20d32ecdd1e5c3'))
    # print(a)
    # print(res)

    # meta_datas = prepare_meta('commons-jcs','d4ed250a0f508e8e3477342d00b090ee3858a40b','67f0a2dca2686b5b740273fb9f20d32ecdd1e5c3')
