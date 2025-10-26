import queue
import networkx as nx
from .util import CONSTANTS


class Slicing:

    def __init__(self, cfg_max_hop=CONSTANTS.cfg_max_hop, max_statement=CONSTANTS.max_statement):
        self.cfg_max_hop = cfg_max_hop
        self.max_statement = max_statement

    def forward_dependency_slicing(self, node, graph: nx.MultiDiGraph,line_set=None,contain_node=True):
        line_ctx = dict()
        visited = set()
        n_nodes = len(graph.nodes)

        q = queue.Queue()
        q.put((node, 0))

        def cdg_view(v, u, t):
            return t == 'CDG'

        def cfg_view(v, u, t):
            return t == 'CFG'

        def ddg_view(v, u, t):
            return t == 'DDG'

        cdg = nx.subgraph_view(graph, filter_edge=cdg_view)
        cfg = nx.subgraph_view(graph, filter_edge=cfg_view)
        ddg = nx.subgraph_view(graph, filter_edge=ddg_view)

        visited = set()
        n_statement = set()
        while len(n_statement) < self.max_statement :
            if len(n_statement) == n_nodes or q.empty():
                break
            curr_v, hop = q.get()
            start_line = graph.nodes[curr_v]['startRow']
            end_line = graph.nodes[curr_v]['endRow']
            for i in range(start_line, end_line + 1):
                line_ctx[i] = graph.nodes[curr_v]['sourceLines'][i - start_line]
            n_statement.add(curr_v)
            if len(n_statement) >= self.max_statement:
                break
            p = curr_v
            if p in cdg.nodes:
                stack=[p]
                #向上找控制依赖，直到遇到方法定义、类定义。
                while len(stack)>0 and len(n_statement) < self.max_statement:
                    p=stack.pop()
                    if p in visited :
                        continue
                    visited.add(p)
                    if p!=curr_v:
                        start_line = graph.nodes[p]['startRow']
                        end_line = graph.nodes[p]['endRow']
                        for i in range(start_line, end_line + 1):
                            line_ctx[i] = graph.nodes[p]['sourceLines'][i - start_line]
                        n_statement.add(p)
                        if len(n_statement) >= self.max_statement:
                          break
                        if graph.nodes[p]['nodeType'] in ['class_declaration','method_declaration','interface_declaration','constructor_declaration','enum_declaration']:
                          break
                    for pred in list(cdg.predecessors(p)):
                        if pred not in visited:
                            stack.append(pred)

            #向上寻找数据依赖，ddg_max_hop步之内的
            stack=[curr_v]
            ddg_hop=CONSTANTS.ddg_max_hop
            while ddg_hop>0 and len(n_statement) < self.max_statement:
              temp_stack=[]
              while len(stack)>0 and len(n_statement) < self.max_statement :
                now= stack.pop()
                for u in ddg.predecessors(now):
                    p = u
                    start_line = graph.nodes[p]['startRow']
                    end_line = graph.nodes[p]['endRow']
                    for i in range(start_line, end_line + 1):
                        line_ctx[i] = graph.nodes[p]['sourceLines'][i - start_line]
                    n_statement.add(p)
                    if len(n_statement) >= self.max_statement:
                        break
                    if p in cdg.nodes:
                        if len(list(cdg.predecessors(p))) != 0:
                            p = list(cdg.predecessors(p))[0]
                            if p not in visited and len(n_statement) < self.max_statement:
                                start_line = graph.nodes[p]['startRow']
                                end_line = graph.nodes[p]['endRow']
                                for i in range(start_line, end_line + 1):
                                    if i not in line_ctx:
                                      line_ctx[i] = graph.nodes[p]['sourceLines'][i - start_line]
                                      n_statement.add(p)
                    if p in ddg.nodes:
                        for u in ddg.predecessors(p):
                          temp_stack.append(u)
              stack=temp_stack
              ddg_hop-=1     

            if hop+1 > self.cfg_max_hop:
                continue
            else:
                for u in cfg.predecessors(curr_v):
                    if u not in visited and 'definition' not in graph.nodes[u]['nodeType']:
                        q.put((u, hop+1))
            visited.add(curr_v)

        if not contain_node:
            n_statement.remove(node)
            start_line = graph.nodes[node]['startRow']
            end_line = graph.nodes[node]['endRow']
            for i in range(start_line, end_line + 1):
                line_ctx.pop(i)

        line_list = list(line_ctx.keys())
        line_list.sort()
        filter_ctx = []
        ctx=[]
        for i in range(0, len(line_list)):
            if not line_list[i] in line_set:
                ctx.append(line_ctx[line_list[i]])
                filter_ctx.append({line_list[i]:line_ctx[line_list[i]]})
                line_set.append(line_list[i])
        # subgraph = nx.subgraph(graph, n_statement)

        return "".join(ctx), filter_ctx, line_list, n_statement
    
    def backward_dependency_slicing(self, node, graph: nx.MultiDiGraph,line_set=None,contain_node=False):
        line_ctx = dict()
        visited = set()

        n_nodes = len(graph.nodes)

        q = queue.Queue()
        q.put((node, 0))

        def cdg_view(v, u, t):
            return t == 'CDG'
        
        def cfg_view(v, u, t):
            return t == 'CFG'

        def ddg_view(v, u, t):
            return t == 'DDG'

        cdg = nx.subgraph_view(graph, filter_edge=cdg_view)
        cfg = nx.subgraph_view(graph, filter_edge=cfg_view)
        ddg = nx.subgraph_view(graph, filter_edge=ddg_view)

        n_statement = set()

        while len(n_statement) < self.max_statement :
            if len(n_statement) == n_nodes or q.empty():
                break
            curr_v, hop = q.get()
            start_line = graph.nodes[curr_v]['startRow']
            end_line = graph.nodes[curr_v]['endRow']
            for i in range(start_line, end_line + 1):
                line_ctx[i] = graph.nodes[curr_v]['sourceLines'][i - start_line]
            n_statement.add(curr_v)
            if len(n_statement) >= self.max_statement:
                break
            p = curr_v

            #向下寻找数据依赖，ddg_max_hop步之内的
            stack=[curr_v]
            ddg_hop=CONSTANTS.ddg_max_hop
            while ddg_hop>0 and len(n_statement) < self.max_statement:
              temp_stack=[]
              while len(stack)>0 and len(n_statement) < self.max_statement :
                now= stack.pop()
                for u in ddg.successors(now):
                    p = u
                    start_line = graph.nodes[p]['startRow']
                    end_line = graph.nodes[p]['endRow']
                    for i in range(start_line, end_line + 1):
                        line_ctx[i] = graph.nodes[p]['sourceLines'][i - start_line]
                    n_statement.add(p)
                    if len(n_statement) >= self.max_statement:
                        break
                    if p in cdg.nodes:
                        if len(list(cdg.predecessors(p))) != 0:
                            for p in list(cdg.predecessors(p)):
                              if p not in visited and len(n_statement) < self.max_statement:
                                start_line = graph.nodes[p]['startRow']
                                end_line = graph.nodes[p]['endRow']
                                for i in range(start_line, end_line + 1):
                                    if i not in line_ctx:
                                      line_ctx[i] = graph.nodes[p]['sourceLines'][i - start_line]
                                      n_statement.add(p)
                    if p in ddg.nodes:
                        for u in ddg.successors(p):
                          temp_stack.append(u)
              stack=temp_stack
              ddg_hop-=1      


            if hop+1 > self.cfg_max_hop:
                continue
            else:
                for u in cfg.successors(curr_v):
                    if u not in visited and 'definition' not in graph.nodes[u]['nodeType']:
                        q.put((u, hop+1))
            visited.add(curr_v)
        
        if not contain_node:
            n_statement.remove(node)
            start_line = graph.nodes[node]['startRow']
            end_line = graph.nodes[node]['endRow']
            for i in range(start_line, end_line + 1):
                line_ctx.pop(i)

        line_list = list(line_ctx.keys())
        line_list.sort()
        filter_ctx = []
        ctx=[]
        for i in range(0, len(line_list)):
            if not line_list[i] in line_set:
                ctx.append(line_ctx[line_list[i]])
                filter_ctx.append({line_list[i]:line_ctx[line_list[i]]})
                line_set.append(line_list[i])
        # subgraph = nx.subgraph(graph, n_statement)

        return "".join(ctx), filter_ctx, line_list, n_statement
