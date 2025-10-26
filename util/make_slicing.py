from .CCG_build import create_graph
from .slicing import Slicing
from networkx.readwrite import json_graph
from .util import set_default, make_needed_dir, dump_jsonl, graph_to_json,CONSTANTS
import json
import os
from pathlib import Path
import networkx as nx
class CCGBuilder:
    def __init__(self, projects_dir=CONSTANTS.projects_dir,
        graph_database_save_dir=CONSTANTS.graph_database_save_dir):
        self.projects_dir = projects_dir
        self.graph_database_save_dir = graph_database_save_dir

        return

    def build_full_graph(self, file_name):
        # code_files = iterate_repository_file(self.projects_dir, repo_name)
        code_file=Path(self.projects_dir)/file_name
        file_num = 0
        make_needed_dir(Path(self.graph_database_save_dir)/file_name)
        with open(code_file, 'r',encoding='utf-8') as f:
            src_lines = f.readlines()
        ccg = create_graph(src_lines, file_name)
        if ccg is None:

            return
        save_path = Path(self.graph_database_save_dir)/file_name/f"{file_num}.json"
        file_num += 1
        make_needed_dir(save_path.parent)
          
        with open(save_path, 'w',encoding='utf-8') as f:
            f.write(json.dumps(json_graph.node_link_data(ccg), default=set_default))
              
        return
    
    def build_slicing_graph(self, file_line,line_set,ccg):
        slicer = Slicing()
        all_statement = set()
        # get graph
        if ccg is None:
            return None,None,None
        filter_dict=dict()
        visit_set=set()
        # slicing for each statement
        for v in ccg.nodes:
            
            if not (ccg.nodes[v]['startRow'] <= file_line and ccg.nodes[v]['endRow'] >= file_line): 
                continue
            visit_set.add(v)
            startline=ccg.nodes[v]['startRow']
            

            _,forward_filter_ctx, _, forward_statement = slicer.forward_dependency_slicing(v, ccg,
                                                                        line_set,contain_node=True)
            _, backward_filter_ctx,_,backward_statement= slicer.backward_dependency_slicing(v, ccg,
                                                                        line_set,contain_node=False)

            all_statement = all_statement.union(forward_statement).union(backward_statement)

            if "key_forward_context" not in filter_dict:
                filter_dict["key_forward_context"]=forward_filter_ctx
            else :
                for ctx in forward_filter_ctx:
                    if ctx not in filter_dict["key_forward_context"]:
                        filter_dict["key_forward_context"].append(ctx)
            if "key_backward_context" not in filter_dict: 
                filter_dict['key_backward_context']=backward_filter_ctx
            else:
                for ctx in backward_filter_ctx:
                    if ctx not in filter_dict["key_backward_context"]:
                        filter_dict["key_backward_context"].append(ctx)
            

            if not ccg.nodes[v]['sourceLines'][file_line-startline].startswith("+"):
                ccg.nodes[v]['sourceLines'][file_line-startline]="+"+ccg.nodes[v]['sourceLines'][file_line-startline]

        line_set.sort()
       
        return filter_dict,all_statement,visit_set





# graph_db_builder = CCGBuilder()
# file_path="groovy/subprojects/groovy-json/src/main/java/org/apache/groovy/json/internal/MapItemValue.java"
# graph_db_builder.build_full_graph(file_path)      
            
    
def sort_filter_blank(context):
   temp= [d for d in context if not all(v.startswith('\n') for v in d.values())]
   result=sorted(temp, key=lambda x: list(x.keys())[0])
   return result

if __name__ == '__main__':
    file_path="demo/demo.java"
    graph_db_builder = CCGBuilder()
    line_set=list()
    name_line_set=dict()
    name_line_set[file_path]=line_set
    merge=list()
    buggy_lines=[172,173,174]
    all_statement=set()
    file_name=file_path
    with open(file_name, 'r',encoding='utf-8') as f:
        src_lines = f.readlines()
    ccg = create_graph(src_lines)
    for file_line in buggy_lines:
        
        print(f'Processing repo {file_path} at line {file_line}...')
        result,that_statement,_=graph_db_builder.build_slicing_graph(file_line-1,name_line_set[file_path],ccg)
        all_statement=all_statement.union(that_statement)
        
        if(len(result)>0):
            merge+=result["key_forward_context"]+result["key_backward_context"]
        
    sorted_context = sort_filter_blank(merge)
    slicing = ''.join(
                    (str(int(key)+1)+":"+"+" + value if int(key+1) in buggy_lines else str(int(key)+1)+":"+value)
                    for item in sorted_context
                    for key, value in item.items()
                    )
    print(slicing)

   