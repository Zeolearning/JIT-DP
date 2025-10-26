import networkx as nx
from tree_sitter import Language, Parser
from collections import defaultdict, deque
import re

def java_control_dependence_graph(root_node, CCG, src_lines, parent):
    node_id = len(CCG.nodes)

    if root_node.type in['import_declaration','try_statement']:
        start_row = root_node.start_point[0]
        end_row = root_node.start_point[0]

        if parent is None:
            CCG.add_node(node_id, nodeType=root_node.type,
                         startRow=start_row, endRow=end_row,
                         sourceLines=src_lines[start_row:end_row + 1],
                         defSet=set(),
                         useSet=set())
            parent = node_id
        else:
            if CCG.nodes[parent]['startRow'] <= start_row and CCG.nodes[parent]['endRow'] >= end_row:
                pass
            else:
                CCG.add_node(node_id, nodeType=root_node.type,
                             startRow=start_row, endRow=end_row,
                             sourceLines=src_lines[start_row:end_row + 1],
                             defSet=set(),
                             useSet=set())
                CCG.add_edge(parent, node_id, 'CDG')
                # parent = node_id
    elif root_node.type in ['class_declaration', 'method_declaration', 'field_declaration','enum_declaration', 'interface_declaration',"constructor_declaration"]:
        if root_node.type in['method_declaration',"constructor_declaration"]:
            start_row = root_node.start_point[0]
            end_row = root_node.child_by_field_name('parameters').end_point[0]
        elif root_node.type in ['enum_declaration']:
            start_row = root_node.start_point[0]
            end_row = root_node.child_by_field_name('name').end_point[0]
        elif root_node.type in ['interface_declaration','class_declaration']:
            start_row = root_node.start_point[0]
            end_row = root_node.child_by_field_name('body').start_point[0]
            
        elif root_node.type == 'field_declaration':
            start_row = root_node.start_point[0]
            end_row = root_node.end_point[0]
        if parent is None:
            CCG.add_node(node_id, nodeType=root_node.type,
                         startRow=start_row, endRow=end_row,
                         sourceLines=src_lines[start_row:end_row + 1],
                         defSet=set(),
                         useSet=set())
            parent = node_id
        else:
            if CCG.nodes[parent]['startRow'] <= start_row and CCG.nodes[parent]['endRow'] >= end_row:
                pass
            else:
                CCG.add_node(node_id, nodeType=root_node.type,
                             startRow=start_row, endRow=end_row,
                             sourceLines=src_lines[start_row:end_row + 1],
                             defSet=set(),
                             useSet=set())
                CCG.add_edge(parent, node_id, 'CDG')
                parent = node_id
    elif root_node.type in ['while_statement', 'for_statement','enhanced_for_statement','do_statement']:
        if root_node.type == 'for_statement':
            start_row = root_node.start_point[0]
            right_node = root_node.child_by_field_name('update')
            if right_node is not None:
                end_row = right_node.end_point[0]
            else:
                right_node = root_node.child_by_field_name('condition')
                if right_node is not None:
                    end_row=right_node.end_point[0]
                else:
                    end_row=start_row
        elif root_node.type == 'while_statement':
            start_row = root_node.start_point[0]
            end_row = root_node.child_by_field_name('condition').end_point[0]
        elif root_node.type == 'do_statement':
            start_row = root_node.start_point[0]
            end_row = start_row
        elif root_node.type == 'enhanced_for_statement':
            start_row = root_node.start_point[0]
            end_row = root_node.child_by_field_name('value').end_point[0]
        if parent is None:
            CCG.add_node(node_id, nodeType=root_node.type,
                         startRow=start_row, endRow=end_row,
                         sourceLines=src_lines[start_row:end_row + 1],
                         defSet=set(),
                         useSet=set())
            parent = node_id
        else:
            if CCG.nodes[parent]['startRow'] <= start_row and CCG.nodes[parent]['endRow'] >= end_row:
                pass
            else:
                CCG.add_node(node_id, nodeType=root_node.type,
                             startRow=start_row, endRow=end_row,
                             sourceLines=src_lines[start_row:end_row + 1],
                             defSet=set(),
                             useSet=set())
                CCG.add_edge(parent, node_id, 'CDG')
                parent = node_id
    elif root_node.type == 'if_statement':
        start_row = root_node.start_point[0]
        end_row = root_node.child_by_field_name('condition').end_point[0]
        if parent is None:
            CCG.add_node(node_id, nodeType=root_node.type,
                         startRow=start_row, endRow=end_row,
                         sourceLines=src_lines[start_row:end_row + 1],
                         defSet=set(),
                         useSet=set())
            parent = node_id
        else:
            if CCG.nodes[parent]['startRow'] <= start_row and CCG.nodes[parent]['endRow'] >= end_row:
                pass
            else:
                CCG.add_node(node_id, nodeType=root_node.type,
                             startRow=start_row, endRow=end_row,
                             sourceLines=src_lines[start_row:end_row + 1],
                             defSet=set(),
                             useSet=set())
                CCG.add_edge(parent, node_id, 'CDG')
                parent = node_id
    elif root_node.type in ['else', 'except_clause', 'catch_clause', 'finally_clause']:
        start_row = root_node.start_point[0]
        end_row = root_node.start_point[0]
        if root_node.type=='else' and root_node.parent.child_by_field_name('alternative')!=None and root_node.parent.child_by_field_name('alternative').type=='if_statement':
            return
        if parent is None:
            CCG.add_node(node_id, nodeType=root_node.type,
                         startRow=start_row, endRow=end_row,
                         sourceLines=src_lines[start_row:end_row + 1],
                         defSet=set(),
                         useSet=set())
            parent = node_id
        else:
            if CCG.nodes[parent]['startRow'] <= start_row and CCG.nodes[parent]['endRow'] >= end_row:
                pass
            else:
                CCG.add_node(node_id, nodeType=root_node.type,
                             startRow=start_row, endRow=end_row,
                             sourceLines=src_lines[start_row:end_row + 1],
                             defSet=set(),
                             useSet=set())
                CCG.add_edge(parent, node_id, 'CDG')
                parent = node_id
    elif root_node.type=='try_with_resources_statement':
        start_row = root_node.start_point[0]
        end_row = root_node.child_by_field_name('resources').end_point[0]
        if parent is None:
            CCG.add_node(node_id, nodeType=root_node.type,
                         startRow=start_row, endRow=end_row,
                         sourceLines=src_lines[start_row:end_row + 1],
                         defSet=set(),
                         useSet=set())
            parent = node_id
        else:
            if CCG.nodes[parent]['startRow'] <= start_row and CCG.nodes[parent]['endRow'] >= end_row:
                pass
            else:
                CCG.add_node(node_id, nodeType=root_node.type,
                             startRow=start_row, endRow=end_row,
                             sourceLines=src_lines[start_row:end_row + 1],
                             defSet=set(),
                             useSet=set())
                CCG.add_edge(parent, node_id, 'CDG')
                parent = node_id
    elif root_node.type == 'switch_block_statement_group':
        start_row = root_node.start_point[0]
        end_row = root_node.start_point[0]
        if parent is None:
            CCG.add_node(node_id, nodeType=root_node.type,
                         startRow=start_row, endRow=end_row,
                         sourceLines=src_lines[start_row:end_row + 1],
                         defSet=set(),
                         useSet=set())
            parent = node_id
        else:
            if CCG.nodes[parent]['startRow'] <= start_row and CCG.nodes[parent]['endRow'] >= end_row:
                pass
            else:
                CCG.add_node(node_id, nodeType=root_node.type,
                             startRow=start_row, endRow=end_row,
                             sourceLines=src_lines[start_row:end_row + 1],
                             defSet=set(),
                             useSet=set())
                CCG.add_edge(parent, node_id, 'CDG')
                parent = node_id
    elif 'statement' in root_node.type or 'ERROR' in root_node.type or "explicit_constructor_invocation" in root_node.type:    
        start_row = root_node.start_point[0]
        end_row = root_node.end_point[0]
        if parent is None:
            CCG.add_node(node_id, nodeType=root_node.type,
                         startRow=start_row, endRow=end_row,
                         sourceLines=src_lines[start_row:end_row + 1],
                         defSet=set(),
                         useSet=set())
            parent = node_id
        else:
            if CCG.nodes[parent]['startRow'] <= start_row and CCG.nodes[parent]['endRow'] >= end_row:
                pass
            else:
                CCG.add_node(node_id, nodeType=root_node.type,
                             startRow=start_row, endRow=end_row,
                             sourceLines=src_lines[start_row:end_row + 1],
                             defSet=set(),
                             useSet=set())
                CCG.add_edge(parent, node_id, 'CDG')
                parent = node_id
    elif root_node.type in["throws","enum_constant"]:
        start_row = root_node.start_point[0]
        end_row = root_node.end_point[0]
        if parent is None:
            CCG.add_node(node_id, nodeType=root_node.type,
                         startRow=start_row, endRow=end_row,
                         sourceLines=src_lines[start_row:end_row + 1],
                         defSet=set(),
                         useSet=set())
            parent = node_id
        else:
            if CCG.nodes[parent]['startRow'] <= start_row and CCG.nodes[parent]['endRow'] >= end_row:
                pass
            else:
                CCG.add_node(node_id, nodeType=root_node.type,
                             startRow=start_row, endRow=end_row,
                             sourceLines=src_lines[start_row:end_row + 1],
                             defSet=set(),
                             useSet=set())
                CCG.add_edge(parent, node_id, 'CDG')
                parent = node_id
    elif root_node.type in ["assignment_expression", "local_variable_declaration",'parenthesized_expression']:
        start_row = root_node.start_point[0]
        end_row = root_node.end_point[0]
        if parent is None:
            CCG.add_node(node_id, nodeType=root_node.type,
                         startRow=start_row, endRow=end_row,
                         sourceLines=src_lines[start_row:end_row + 1],
                         defSet=set(),
                         useSet=set())
            parent = node_id
        else:
            if CCG.nodes[parent]['startRow'] <= start_row and CCG.nodes[parent]['endRow'] >= end_row:
                pass
            else:
                CCG.add_node(node_id, nodeType=root_node.type,
                             startRow=start_row, endRow=end_row,
                             sourceLines=src_lines[start_row:end_row + 1],
                             defSet=set(),
                             useSet=set())
                CCG.add_edge(parent, node_id, 'CDG')
                parent = node_id
    elif root_node.type =='switch_expression':
        start_row = root_node.start_point[0]
        end_row = root_node.child_by_field_name('condition').end_point[0]
        if parent is None:
            CCG.add_node(node_id, nodeType=root_node.type,
                         startRow=start_row, endRow=end_row,
                         sourceLines=src_lines[start_row:end_row + 1],
                         defSet=set(),
                         useSet=set())
            parent = node_id
        else:
            if CCG.nodes[parent]['startRow'] <= start_row and CCG.nodes[parent]['endRow'] >= end_row:
                pass
            else:
                CCG.add_node(node_id, nodeType=root_node.type,
                             startRow=start_row, endRow=end_row,
                             sourceLines=src_lines[start_row:end_row + 1],
                             defSet=set(),
                             useSet=set())
                CCG.add_edge(parent, node_id, 'CDG')
                parent = node_id

    for child in root_node.children:
        if child.type == 'identifier':
            row = child.start_point[0]
            col_start = child.start_point[1]
            col_end = child.end_point[1]
            identifier_name = src_lines[row][col_start:col_end].strip()
            if parent is None:
                continue
            if CCG.nodes[parent]['nodeType'] in ['class_declaration', 'method_declaration','enum_declaration', 'interface_declaration', 'constructor_declaration','yield_statement','catch_clause'] :
                CCG.nodes[parent]['defSet'].add(identifier_name)
            elif CCG.nodes[parent]['nodeType'] in ['enhanced_for_statement', 'for_statement']:
                p = child
                while 'for_statement' not in p.parent.type:
                    p = p.parent

                if 'for_statement' in p.parent.type and (p.prev_sibling.type == 'for' or 'type' in p.prev_sibling.type):
                    CCG.nodes[parent]['defSet'].add(identifier_name)
                else:
                    CCG.nodes[parent]['useSet'].add(identifier_name)
            elif CCG.nodes[parent]['nodeType'] in ['assignment_expression', 'local_variable_declaration','field_declaration','expression_statement','try_with_resources_statement','enum_constant']:
                if (child.next_sibling is not None and child.next_sibling.type=='=') or root_node.child_count==1:
                    CCG.nodes[parent]['defSet'].add(identifier_name)
                else:
                    CCG.nodes[parent]['useSet'].add(identifier_name)
            elif 'import' in CCG.nodes[parent]['nodeType']:
                CCG.nodes[parent]['defSet'].add(identifier_name)
            else:
                CCG.nodes[parent]['useSet'].add(identifier_name)
        elif child.type =='type_identifier' and root_node.type in ["superclass","generic_type","object_creation_expression","type_arguments"]and parent is not None:
            row = child.start_point[0]
            col_start = child.start_point[1]
            col_end = child.end_point[1]
            identifier_name = src_lines[row][col_start:col_end].strip()
            CCG.nodes[parent]['useSet'].add(identifier_name)
                            
        elif child.type in ['type_list']:
            for ch in child.children:
                row = ch.start_point[0]
                col_start = ch.start_point[1]
                col_end = ch.end_point[1]
                identifier_name = src_lines[row][col_start:col_end].strip()
                CCG.nodes[parent]['useSet'].add(identifier_name)
        
        java_control_dependence_graph(child, CCG, src_lines, parent)
    return

def add_else_cdg(CCG,src_lines):
    for v in CCG.nodes:
        if_parent=set()
        if CCG.nodes[v]['nodeType']=='if_statement'and not 'else' in src_lines[CCG.nodes[v]['startRow']]:
            if_parent.add(v)
            deep_search_else_if(CCG,src_lines,if_parent,v)

def deep_search_else_if(CCG,src_lines,if_parent,v):
            chs=list(CCG.neighbors(v))
            if len(chs)!=0:
                chs.sort()
                next_child=False
                for ch in chs:
                    if CCG.nodes[ch]['nodeType']=='if_statement'and 'else' in src_lines[CCG.nodes[ch]['startRow']]:
                        for p in if_parent:
                            if p!=ch:
                                CCG.add_edge(p,ch,'CDG')        
                                # print("add edge ",p,'to',ch)
                        if_parent.add(ch)
                        deep_search_else_if(CCG,src_lines,if_parent,ch)
                    elif CCG.nodes[ch]['nodeType']=='else':
                        if_parent.add(ch)
                        next_child=True
                    if next_child :
                        for p in if_parent:
                            if p!=ch:
                                CCG.add_edge(p,ch,'CDG')        
                                # print("add edge ",p,'to',ch)

def java_control_flow_graph(CCG):
    CFG = nx.MultiDiGraph()

    next_sibling = dict()
    first_children = dict()

    start_nodes = []
    #无依赖的节点就是开始分析的节点
    for v in CCG.nodes:
        if len(list(CCG.predecessors(v))) == 0:
            start_nodes.append(v)
    #将所有开始节点连接起来
    start_nodes.sort()
    for i in range(0, len(start_nodes) - 1):
        v = start_nodes[i]
        u = start_nodes[i + 1]
        next_sibling[v] = u    
    if start_nodes:
        next_sibling[start_nodes[-1]] = None

    for v in CCG.nodes:
        #v->u,w
        children = list(CCG.neighbors(v))
        if len(children) != 0:
            children.sort()
            for i in range(0, len(children) - 1):
                u = children[i]
                w = children[i + 1]
                if CCG.nodes[v]['nodeType'] in[ 'if_statement','try_statement'] and (CCG.nodes[w]['nodeType']=='else' or 'clause' in CCG.nodes[w]['nodeType']):
                    next_sibling[u] = None
                else:
                    next_sibling[u] = w
            next_sibling[children[-1]] = None

            first_children[v] = children[0]
        else:
            first_children[v] = None

    edge_list = []

    for v in CCG.nodes:
        # block start control flow
        # v->u
        if v in first_children.keys():
            u = first_children[v]
            if u is not None:
                edge_list.append((v, u, 'CFG'))
        # block end control flow
        if CCG.nodes[v]['nodeType'] == 'return_statement':
            pass
        elif CCG.nodes[v]['nodeType'] in ['break_statement', 'continue_statement']:
            u = None
            
            p = list(CCG.predecessors(v))[0]
            
            while CCG.nodes[p]['nodeType'] not in ['for_statement', 'while_statement','enhanced_for_statement','do_statement','switch_expression']:
                p = list(CCG.predecessors(p))[0]

            if CCG.nodes[v]['nodeType'] == 'break_statement':
                u = next_sibling[p]
            if CCG.nodes[v]['nodeType'] == 'continue_statement':
                u = p
            if u is not None:
                edge_list.append((v, u, 'CFG'))
        elif CCG.nodes[v]['nodeType'] in  ['for_statement','enhanced_for_statement']:
            u = first_children[v]
            if u is not None:
                edge_list.append((v, u, 'CFG'))
        elif CCG.nodes[v]['nodeType'] in ['while_statement','do_statement']:
            u = first_children[v]
            if u is not None:
                edge_list.append((v, u, 'CFG'))
            u = next_sibling[v]
            if u is not None:
                edge_list.append((v, u, 'CFG'))
        elif CCG.nodes[v]['nodeType'] in ['if_statement' , 'try_statement','try_with_resources_statement']:
            u = first_children[v]
            if u is not None:
                edge_list.append((v, u, 'CFG'))
            for u in CCG.neighbors(v):
                if 'clause' in CCG.nodes[u]['nodeType'] :
                    edge_list.append((v, u, 'CFG'))
                elif CCG.nodes[u]['nodeType']=='else':
                    edge_list.append((v, u, 'CFG'))
        elif CCG.nodes[v]['nodeType'] == 'switch_expression':
            u = first_children[v]
            if u is not None:
                edge_list.append((v, u, 'CFG'))
            while next_sibling[u] is not None:
                u = next_sibling[u]
                edge_list.append((v, u, 'CFG'))
        elif 'clause' in CCG.nodes[v]['nodeType']:
            u = first_children[v]
            if u is not None:
                edge_list.append((v, u, 'CFG'))
            

        u = next_sibling[v]
        if u is None:
            p = v
            while len(list(CCG.predecessors(p))) != 0:
                p = list(CCG.predecessors(p))[0]
                if CCG.nodes[p]['nodeType'] in ['for_statement','while_statement','enhanced_for_statement','do_statement']:
                        edge_list.append((v, p, 'CFG'))

                if CCG.nodes[p]['nodeType'] in ['try_statement', 'if_statement','catch_clause','finally_clause','try_with_resources_statement']:
                    if next_sibling[p] is not None:
                        edge_list.append((v, next_sibling[p], 'CFG'))
                        break
        if u is not None:
            edge_list.append((v, u, 'CFG'))


    CFG.add_edges_from(edge_list)
    for v in CCG.nodes:
        if v not in CFG.nodes:
            CFG.add_node(v)
    return CFG, edge_list


# def java_data_dependence_graph(CFG, CCG):
#     for v in CCG.nodes:
#         for u in CCG.nodes:
#             if v == u or 'import' in CCG.nodes[u]['nodeType']:
#                 continue
#             if len(CCG.nodes[v]['defSet'] & CCG.nodes[u]['useSet']) != 0 and nx.has_path(CFG, v, u):
#                 has_path = False
#                 paths = list(nx.all_shortest_paths(CFG, source=v, target=u))
#                 variables = CCG.nodes[v]['defSet'] & CCG.nodes[u]['useSet']
#                 for var in variables:
#                     has_def = False
#                     for path in paths:
#                         for p in path[1:-1]:
#                             if var in CCG.nodes[p]['defSet']:
#                                 has_def = True
#                                 break
#                         if not has_def:
#                             has_path = True
#                             break
#                     if has_path:
#                         break
#                 if has_path:
#                     CCG.add_edge(v, u, 'DDG')
#     return
def java_data_dependence_graph(CFG, CCG):
    def_nodes=defaultdict(set)
    use_nodes=defaultdict(set)

    for node in CCG.nodes:
        for v in CCG.nodes[node]['defSet']:
            def_nodes[v].add(node)
        for v in CCG.nodes[node]['useSet']:
            use_nodes[v].add(node)

    for var,uses in use_nodes.items():
        defs=def_nodes.get(var,set())
        if len(defs)==0:
            continue

        for u in uses:
            seen={u}
            q=deque([u])
            while q:
                curr=q.popleft()
                for pred in CFG.predecessors(curr):
                    if pred in seen:
                        continue
                    seen.add(pred)

                    if var in CCG.nodes[pred]['defSet']:
                        CCG.add_edge(pred,u,'DDG')
                        continue
                    else:
                        q.append(pred)
    
def is_inside_string(line, pos):
    """判断 line[pos] 是否位于单引号或双引号字符串内部（不考虑转义）"""
    in_single = False
    in_double = False
    if pos<0 or pos>=len(line):
        return False
    for i in range(pos):
        c = line[i]
        if c == "'" and not in_double:
            in_single = not in_single
        elif c == '"' and not in_single:
            in_double = not in_double
    return in_single or in_double

def create_graph(code_lines):

    src_lines = "".join(code_lines).encode('ascii', errors='ignore').decode('ascii')

    src_lines = src_lines.splitlines(keepends=True)
    if len(src_lines) != 0:
        src_lines[-1] = src_lines[-1].rstrip().strip('(').strip('[').strip(',')
    # Define tree-sitter parser
    Language.build_library('./my-languages.so', ['./tree-sitter-java'])
    language = Language('./my-languages.so', "java")
    parser = Parser()
    parser.set_language(language)

    if len(src_lines) == 0:
        return None

    #remove comment
    comment_lines = []
    i=0
    in_block_comment=False
    while i < len(src_lines):
        line = src_lines[i]
        stripped = line.lstrip()

        if in_block_comment :
            comment_lines.append(i)
            src_lines[i] = '\n'
            if '*/' in stripped:
                in_block_comment = False
        elif stripped.startswith('//'):
            comment_lines.append(i)
            src_lines[i] = '\n'
        elif stripped.startswith('/*'):
            comment_lines.append(i)
            src_lines[i] = '\n'
            if '*/' not in stripped:
                in_block_comment = True
        else:
            line = src_lines[i]
            if '/*' not in line:
                i+=1
                continue
            semicolon_pos = line.find(';')
            if semicolon_pos != -1:
                # 在分号之后查找 /*
                comment_start = line.find('/*', semicolon_pos)
                if comment_start != -1:
                    if is_inside_string(line,comment_start):
                        i+=1
                        continue
                    if not line.rstrip().endswith("*/"):
                        in_block_comment=True
                    src_lines[i] = line[:comment_start] + '\n' if line.endswith('\n') else line[:comment_start]
        i += 1

    # Parser file to get a tree
    def read_callable(byte_offset, point):
        row, column = point
        if row >= len(src_lines) or column >= len(src_lines[row]):
            return None
        return src_lines[row][column:].encode('utf8', errors='ignore')
    tree = parser.parse(read_callable)

    all_comment = True
    for child in tree.root_node.children:
        if child.type != "comment":
            all_comment = False
    if all_comment:
        return None

    # Initialize program dependence graph
    ccg = nx.MultiDiGraph()

    # Construct control dependence edge
    for child in tree.root_node.children:
        java_control_dependence_graph(child, ccg, code_lines, None)
    add_else_cdg(ccg,code_lines)
    # Construct control flow graph
    cfg, cfg_edge_list = java_control_flow_graph(ccg)

    # Construct data dependence graph
    java_data_dependence_graph(cfg, ccg)

    ccg.add_edges_from(cfg_edge_list)

    return ccg
