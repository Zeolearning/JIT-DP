from collections import defaultdict
import json
import threading
import time
import pandas as pd
from .util import CONSTANTS
import re
import subprocess
from tqdm import tqdm
import csv
import traceback
from .CCG_build import create_graph
from .make_slicing import CCGBuilder, sort_filter_blank
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import asyncio
from tqdm.asyncio import tqdm_asyncio
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import random
import openai
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
failcount=[[],[]]
def get_git_diff(repo_path, ref1='HEAD', ref2=None):
    """
    获取指定仓库路径下的 git diff 输出。
    
    :param repo_path: 仓库的本地路径
    :param ref1: 比较的起始引用（如 'HEAD~1'、'main' 等）
    :param ref2: 比较的目标引用（如 'HEAD'），若为 None，则表示工作区与 ref1 的差异
    :return: diff 的字符串输出
    """
    cmd = ['git', 'diff']
    if ref2 is not None:
        cmd.extend([str(ref1), str(ref2)])
    else:
        cmd.append(str(ref1))

    result = subprocess.run(
        cmd,
        cwd=repo_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding='utf-8',        
        errors='replace' 
    )
    
    if result.returncode != 0:
        failcount[0].append(repo_path)
        failcount[1].append(ref2)
    
    return result.stdout

def parse_cmd_diff(cmd_output,file_all_bug):
    """
    解析 git diff 输出，提取引入bug的行号。
    
    :param cmd_output: git diff 的字符串输出
    :param file_all_bug: 引入bug的代码行

    :return: 引入bug的行号列表{file,line,code}
    """
    text=cmd_output.strip()
    result=dict(dict())
    line_list_dict=defaultdict(list)

    diff_blocks = re.split(r'diff --git', text)[1:]


    for block in diff_blocks:
        lines = block.split('\n')#每块diff分行
        match=re.match(r'^a/(.*?) b/(.*?)$',lines[0].strip())
        if (match):
            file_path = match.group(2)
        else:
            continue
        if file_path.endswith(".java")==False:
            continue

        line_list=[]
        current_add_line=0
        new_start_b=0
        for line in lines[1:]:
            if line.startswith('@@'):
                # 解析新文件行号
                addline=re.search(r'\+(\d+),?', line)
                if addline :
                    new_start_b=int(addline.group(1))
                else :
                    continue
                current_add_line = new_start_b - 1  # 补偿初始自增
            elif line.startswith('+') and not line.startswith("+++"):
                current_add_line += 1
                clean_line=line[1:].strip()

                if clean_line in file_all_bug.get(file_path,set()):
                  line_list.append(current_add_line)
                  code_and_line=dict()
                  code_and_line[clean_line]=current_add_line
                  result[file_path]=code_and_line

            elif line.startswith(' ') :
                current_add_line +=1
        if len(line_list)>0:
          line_list_dict[file_path]=line_list
    return result,line_list_dict


def line_construc():
    buggy_repos = None
    with open(CONSTANTS.repository_dir+'/buggy_repos.json', 'r', encoding='utf-8') as f:
        buggy_repos = json.load(f)
    jsondata=dict()
    for i in range(2):
        if i==0:
            dataset_file=CONSTANTS.repository_dir + '/features_train.csv'
        else:
            dataset_file=CONSTANTS.repository_dir + '/features_valid.csv'
        with open(dataset_file, 'r', encoding='utf-8') as f:
            df = pd.read_csv(f)
            for index, row in tqdm(df.iterrows(),total=len(df),desc="Constructing Repository..."):
                project = row['project']
                commit_hash = row['commit_hash']
                parent_hash = row['parent_hashes']
                if_bug_commit=row['is_buggy_commit']==1.0
                if not if_bug_commit:
                    continue
                
                project_path = Path(CONSTANTS.projects_dir)/project
                if not os.path.exists(project_path):
                    print(f"Project path does not exist: {project_path}")
                    continue
                
                cmd_output = get_git_diff(project_path, parent_hash, commit_hash)

                file_all_bug = dict()
                for filePath,changes in buggy_repos[project][commit_hash]["added_buggy_level"].items():
                    all_bugs = set()
                    for buggy_code in changes["added_buggy"]:
                        buggy_code=buggy_code.strip()
                        pattern = re.compile(r'}\s*(?:else|catch|finally)\s*{')
                        if len(buggy_code)>1 and not pattern.fullmatch(buggy_code):
                            #剔除简单的“}”等
                            all_bugs.add(buggy_code)
                    file_all_bug[filePath]=all_bugs

                buggy_line_dict,line_list_dict=parse_cmd_diff(cmd_output,file_all_bug)
                this_hash_list=[]
                for file_name,file_line in line_list_dict.items():
                    muti_data={
                        "file_path":file_name,
                        "buggy_lines":str(file_line)
                    }
                    this_hash_list.append(muti_data)
                jsondata[project+":"+commit_hash]=this_hash_list
        print("\n")
    jsondata = dict(sorted(jsondata.items()))
    with open(CONSTANTS.repository_dir+"/repository.json",'w') as js:
        json.dump(jsondata, js, indent=4)
    print("failcount:",len(failcount))
    print("successcount:",len(jsondata))

def buggy_line_construc():
    buggy_repos = None
    with open(CONSTANTS.repository_dir+'/buggy_repos.json', 'r', encoding='utf-8') as f:
        buggy_repos = json.load(f)
    jsondata=dict()

    dataset_files=[CONSTANTS.repository_dir + '/features_train.csv',
                   CONSTANTS.repository_dir + '/features_valid.csv',
                   CONSTANTS.repository_dir + '/features_test.csv',   
    ]
    for dataset_file in dataset_files:
        with open(dataset_file, 'r', encoding='utf-8') as f:
            df = pd.read_csv(f)
            for _, row in tqdm(df.iterrows(),total=len(df),desc="Constructing Repository..."):
                project = row['project']
                commit_hash = row['commit_hash']
                parent_hash = row['parent_hashes']
                if_bug_commit=row['is_buggy_commit']==1.0
                if not if_bug_commit:
                    continue

                project_path = Path(CONSTANTS.projects_dir)/project
                if not os.path.exists(project_path):
                    print(f"Project path does not exist: {project_path}")
                    continue

                cmd_output = get_git_diff(project_path, parent_hash, commit_hash)

                file_all_bug = dict()
                for filePath,changes in buggy_repos[project][commit_hash]["added_buggy_level"].items():
                    all_bugs = set()
                    for buggy_code in changes["added_buggy"]:
                        buggy_code=buggy_code.strip()
                        pattern = re.compile(r'}\s*(?:else|catch|finally)\s*{')
                        if len(buggy_code)>1 and not pattern.fullmatch(buggy_code):
                            #剔除简单的“}”等
                            all_bugs.add(buggy_code)
                    file_all_bug[filePath]=all_bugs

                _,line_list_dict=parse_cmd_diff(cmd_output,file_all_bug)
                this_hash_list=dict()
                for file_name,file_line in line_list_dict.items():
                    this_hash_list[file_name]=str(file_line)
                jsondata[project+":"+commit_hash]=this_hash_list
    jsondata = dict(sorted(jsondata.items()))
    with open(CONSTANTS.repository_dir+"/repository_lines.json",'w') as js:
        json.dump(jsondata, js, indent=4)
    print("failcount:",len(failcount))
    print("successcount:",len(jsondata))

def getSlice(project,inner_file_path,commit_hash,target_lines):
    '''
        传入文件名称，构造lines相关的切片
    '''
    project_path = Path(CONSTANTS.projects_dir)/project
    if not os.path.exists(project_path):
        raise ValueError(file_path,commit_hash,f"Project path does not exist: {project_path}")
    
    checkout_hash(project_path, commit_hash)
    graph_db_builder = CCGBuilder()
    line_set=list()
    merge=list()
    file_path=Path(project_path)/inner_file_path
    if not os.path.exists(file_path) or len(target_lines)==0:
        return ""
    with open(file_path, 'r',encoding='latin-1') as f:
        src_lines = f.readlines()

        try:
            ccg = create_graph(src_lines)
        except Exception as e:
            print(f"Error creating CCG for {file_path}:{commit_hash} {e}")
            traceback.print_exc()
            exit(1)
        for file_line in target_lines:
            result,_,_=graph_db_builder.build_slicing_graph(int(file_line)-1,line_set,ccg)
            if(len(result)>0):
                merge+=result["key_forward_context"]+result["key_backward_context"]

        sorted_context = sort_filter_blank(merge)

        slicing = ''.join(
            ("+" + value if str(int(key+1)) in target_lines or int(key)+1 in line_set else value)
            for item in sorted_context
            for key, value in item.items())
        slicing = re.sub(r'/\*.*?\*/', '', slicing, flags=re.DOTALL)
        slicing = re.sub(r'//(?!www).*', '', slicing)
        slicing = slicing.replace('\r', '').replace('\n', '\\n')
    return slicing

PROJECT_LOCKS = {}

def get_project_lock(project):
    if project not in PROJECT_LOCKS:
        PROJECT_LOCKS[project] = threading.Lock()
    return PROJECT_LOCKS[project]

def run_getSlice(project, inner_file_path, commit_hash, target_lines):
    lock = get_project_lock(project)
    with lock:  # 在调用时控制互斥
        return getSlice(project, inner_file_path, commit_hash, target_lines)
def checkout_hash(repo_path, commit_hash):
    """
    切换到指定的 git 提交。
    
    :param repo_path: 仓库的本地路径
    :param commit_hash: 目标提交的哈希值
    :return: None
    """
    
    cmd = ['git', 'checkout','-f', commit_hash]
    head_lock = Path(repo_path)/ ".git"/ "HEAD.lock"
    retry=10
    try :
        for attempt in range(retry):
            if not os.path.exists(head_lock):
                break
            print(f"HEAD.lock 存在，等待释放... (尝试 {attempt+1}/{retry})")
            time.sleep(0.5)
        if os.path.exists(head_lock):
            raise RuntimeError("HEAD.lock 长时间未释放，放弃切换。")
        result = subprocess.run(
            cmd,
            cwd=repo_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True, 
            text=True,
            encoding='utf-8',        
            errors='replace' 
        )
    except subprocess.CalledProcessError :
        time.sleep(2)
        return checkout_hash(repo_path, commit_hash)
    
    if result.returncode != 0:
        failcount[0].append(repo_path)
        failcount[1].append(commit_hash)
        raise RuntimeError(f"Error checking out {commit_hash} in {repo_path}: {result.stderr}")

# def process_srclines(src_lines):
#     for i in range(len(src_lines)):
#         line = src_lines[i]
#         semicolon_pos = line.find(';')
#         if semicolon_pos != -1:
#             # 在分号之后查找 /*
#             comment_start = line.find('/*', semicolon_pos)
#             if comment_start != -1:
#                   src_lines[i] = line[:comment_start] + '\n' if line.endswith('\n') else line[:comment_start]
#     return src_lines

def construc_slicing_from_hash():
    fail_count=0
    with open(CONSTANTS.repository_dir+'/knowledge.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['repo','hash', 'slicing', 'diff', 'desc'])
    with open(CONSTANTS.repository_dir+"/repository.json",'r') as js:
        repo_data = json.load(js)
        for file_and_sha,datas in tqdm(repo_data.items(),total=len(repo_data),desc="Constructing Slicing..."):
            project,commit_hash=file_and_sha.split(":")
            # if commit_hash!="0430af408da0da9c1577ff36bb43260bb1577e16":
            #     continue
            project_path = Path(CONSTANTS.projects_dir)/project
            if not os.path.exists(project_path):
                print(f"Project path does not exist: {project_path}")
                fail_count+=1
                continue
            checkout_hash(project_path, commit_hash)

            for data in datas:
                buggy_lines=eval(data["buggy_lines"])
                file_path=Path(project_path)/data["file_path"]

                if not os.path.exists(file_path):
                    print(f"File path does not exist: {file_path}")
                    fail_count+=1
                    continue

                graph_db_builder = CCGBuilder()
                line_set=list()
                merge=list()

                with open(file_path, 'r',encoding='latin-1') as f:
                    src_lines = f.readlines()


                try:
                    ccg = create_graph(src_lines)
                except Exception as e:
                    print(f"Error creating CCG for {project}:{file_path}:{commit_hash} {e}")
                    traceback.print_exc()
                    exit(1)
                src_code=[]

                for line_number in buggy_lines:
                    if "+"+src_lines[line_number-1] not in src_code:
                        src_code.append("+"+src_lines[line_number-1])
                    
                for file_line in buggy_lines:
                    result,_,_=graph_db_builder.build_slicing_graph(file_line-1,line_set,ccg)
                    if(len(result)>0):
                        merge+=result["key_forward_context"]+result["key_backward_context"]
                sorted_context = sort_filter_blank(merge)

                slicing = ''.join(
                    ("+" + value if int(key+1) in buggy_lines else value)
                    for item in sorted_context
                    for key, value in item.items()
                    )
   
                slicing = re.sub(r'/\*.*?\*/', '', slicing, flags=re.DOTALL)
                slicing = re.sub(r'//(?!www).*', '', slicing)
                slicing = slicing.replace('\r', '').replace('\n', '\\n')
                with open(CONSTANTS.repository_dir+'/knowledge.csv', 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([project,commit_hash, slicing, src_code, ''])


def testdemo():
    buggy_repos = None
    with open(CONSTANTS.repository_dir+'/buggy_repos.json', 'r', encoding='utf-8') as f:
        buggy_repos = json.load(f)
    dataset_file=CONSTANTS.repository_dir + '/features_train.csv'
    with open(dataset_file, 'r', encoding='utf-8') as f:
        df = pd.read_csv(f)
        for index, row in tqdm(df.iterrows(),total=len(df),desc="Constructing Repository..."):
            project = row['project']
            commit_hash = row['commit_hash']
            if(commit_hash!="09499726b740d11acfdd16d1658ba98beb808d71"):
                continue
            parent_hash = row['parent_hashes']
            if_bug_commit=row['is_buggy_commit']==1.0
            if not if_bug_commit:
                continue
            
            project_path = Path(CONSTANTS.projects_dir)/project
            if not os.path.exists(project_path):
                print(f"Project path does not exist: {project_path}")
                continue
            
            cmd_output = get_git_diff(project_path, parent_hash, commit_hash)
            file_all_bug = dict()
            for filePath,changes in buggy_repos[project][commit_hash]["added_buggy_level"].items():
                all_bugs = set()
                for buggy_code in changes["added_buggy"]:
                    buggy_code=buggy_code.strip()
                    pattern = re.compile(r'}\s*(?:else|catch|finally)\s*{')
                    if len(buggy_code)>1 and not pattern.fullmatch(buggy_code):
                        #剔除简单的“}”等
                        all_bugs.add(buggy_code)
                file_all_bug[filePath]=all_bugs
            print(file_all_bug)
            buggy_line_dict,line_list_dict=parse_cmd_diff(cmd_output,file_all_bug)
            this_hash_list=[]
            for file_name,file_line in line_list_dict.items():
                muti_data={
                    "file_path":file_name,
                    "buggy_lines":str(file_line)
                }
                this_hash_list.append(muti_data)
            print( this_hash_list)



async def generate_desc(slicing):
    llm = ChatOpenAI(
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        model="qwen-max",
        temperature=0
    )
    prompt = ChatPromptTemplate.from_template("""
You are an experienced software engineer and code reviewer.
The newly added diff lines have already introduced an issue. 
Your task is to focus on the code slice and analyze **why these added lines cause the issue** in the context of the updated code slice.
               
Focus on Java-specific issues such as:
- NullPointerException or uninitialized variables  
- Deprecated API usage  
- Improper exception handling  
- Resource leaks or unsafe I/O operations  
- Logical mistakes that could lead to incorrect behavior  

Important rules:
- The resulting code are **syntactically correct** and **compile successfully**.
- The updated code slice **already contains** the newly added diff lines.
- The removed code in the diff are hidden
- The slice may be incomplete, so check for errors in the added code, using the slice context only as a reference.
Inputs:
**Updated code slice (After applying the diff, already contains the newly added diff lines):**
{slicing}

Your output:
- Provide a concise (1-5 sentences) explanation of the root cause or risky pattern introduced by the newly added lines.
- Focus only on the issue caused by the diff; **DO NOT** discuss formatting, style, or syntax.

""")

    result = await (prompt | llm).ainvoke({"slicing": slicing})
    return result.content


SEM = asyncio.Semaphore(10)
@retry(
    stop=stop_after_attempt(5),                # 最多重试5次
    wait=wait_exponential(multiplier=1, min=2, max=30),  # 指数退避
    retry=retry_if_exception_type(openai.RateLimitError)
)
async def safe_generate_desc(slicing):
    async with SEM:  # 控制并发
        desc = await generate_desc(slicing)
        await asyncio.sleep(random.uniform(0.5, 1.5))  # 在请求间随机等待，减少撞限
        return desc.replace("\n", " ")

async def add_desc():
    knowledge_path = CONSTANTS.repository_dir + '/knowledge.csv'
    df = pd.read_csv(knowledge_path)
    df['desc'] = df['desc'].astype(object)

    async def process_row(index, row):
        slicing = row['slicing']
        diffs = row['diff']
        desc = row['desc']

        if pd.notna(desc) and str(desc).strip() != "":
            return index, desc
        
        new_desc = await safe_generate_desc(slicing)
        return index, new_desc

    try:
        tasks = [process_row(index, row) for index, row in df.iterrows()]

        # 使用异步tqdm批量执行，但受Semaphore保护，不会瞬间超限
        for coro in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc="Generating Description..."):
            idx, new_desc = await coro
            df.at[idx, 'desc'] = new_desc

    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        df.to_csv(knowledge_path, index=False)


# construc_slicing_from_hash()

if __name__ == "__main__":
    buggy_line_construc()

