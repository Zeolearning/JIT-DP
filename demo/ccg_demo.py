import sys
from pathlib import Path

# 将项目根目录加入 sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from util.util import  CONSTANTS, set_default
from util.CCG_build  import create_graph
from pathlib import Path
from networkx.readwrite import json_graph
import json
import re

def process_srclines(src_lines):
    for i in range(len(src_lines)):
        line = src_lines[i]
        semicolon_pos = line.find(';')
        if semicolon_pos != -1:
            # 在分号之后查找 /*
            comment_start = line.find('/*', semicolon_pos)
            if comment_start != -1:
                  src_lines[i] = line[:comment_start] + '\n' if line.endswith('\n') else line[:comment_start]
    return src_lines
projects_dir=CONSTANTS.projects_dir
file='demo/demo.java'
with open(file, 'r',encoding='latin1') as f:
    src_lines = f.readlines()
    #src_lines=process_srclines(src_lines)
    ccg = create_graph(src_lines)

    
    json_str = json.dumps(json_graph.node_link_data(ccg), default=set_default, indent=4)
    with open("demo/demo.json", 'w',encoding='utf-8') as f:
      f.write(json_str)
