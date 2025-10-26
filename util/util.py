import os
import json
from networkx.readwrite import json_graph
import networkx as nx

class CONSTANTS:
    projects_dir = f"./Dataset"
    cfg_max_hop = 1
    ddg_max_hop = 1
    max_statement = 10
    graph_database_save_dir = "./graph_database"
    repository_dir='./repository'
def set_default(obj):
    if isinstance(obj, set):
        return list(obj)
    raise TypeError

def make_needed_dir(file_path):
    dir_path = os.path.dirname(file_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return

def dump_jsonl(obj, fname):
    with open(fname, 'w', encoding='utf8') as f:
        for item in obj:
            f.write(json.dumps(item, default=set_default) + '\n')



def graph_to_json(obj: nx.MultiDiGraph):
    return json.dumps(json_graph.node_link_data(obj, link="edges"), default=set_default)


