
import json
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from util.CCG_build import create_graph
from networkx.readwrite import json_graph
from util.util import set_default, make_needed_dir, dump_jsonl, graph_to_json,CONSTANTS
from util.make_slicing import CCGBuilder
import re
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, softmax
from torch_geometric.loader import DataLoader
from transformers import AutoTokenizer, AutoModel

def generate_vector(code):
    tokenizer = AutoTokenizer.from_pretrained("huggingface/CodeBERTa-small-v1")
    model = AutoModel.from_pretrained("huggingface/CodeBERTa-small-v1")

    inputs = tokenizer(code, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        code_embedding = outputs.last_hidden_state.mean(dim=1)
        return code_embedding[0]


def read_graph():
    dataset=[]
    output_path=Path(CONSTANTS.repository_dir+'/train_graph_dataset.jsonl')
    with open(output_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            G = json_graph.node_link_graph(data["relate_graph"])

            diff_nodes=set(data["diff_node"])
            label = torch.tensor([data["label"]], dtype=torch.long)
            nodes_list=list(G.nodes())
            node_embeding=[]

            diff_node_idx=[nodes_list.index(node) for node in diff_nodes]

                
            edge_types=[]
            edge_index=[[],[]]
            for u, v in G.edges():
                types_list=[*(G.get_edge_data(u,v))]
                if 'CFG' in types_list:
                    edge_types.append(0)
                    edge_index[0].append(nodes_list.index(u))
                    edge_index[1].append(nodes_list.index(v))
                
                if 'CDG' in types_list:
                    edge_types.append(1)
                    edge_index[0].append(nodes_list.index(u))
                    edge_index[1].append(nodes_list.index(v))
                
                if 'DDG' in types_list:
                    edge_types.append(2)
                    edge_index[0].append(nodes_list.index(u))
                    edge_index[1].append(nodes_list.index(v))

            for value in nodes_list:
                mix=G.nodes[value]["nodeType"]+":"+"".join(G.nodes[value]["sourceLines"])
                mix = re.sub(r'\s+', ' ', mix)
                node_embeding.append(generate_vector(mix))
            
            data_item=Data(
                x=torch.stack(node_embeding),
                edge_index=torch.tensor(edge_index, dtype=torch.long),
                edge_type=torch.tensor(edge_types, dtype=torch.long),
                diff_idx=torch.tensor(diff_node_idx, dtype=torch.long),
                y=label
            )
            dataset.append(data_item)

    return dataset
    


# -------------------------
# 自定义带权GGNN
# -------------------------
class WeightedGGNN(MessagePassing):
    def __init__(self, in_channels, out_channels, hidden_dim,num_edge_types):
        super(WeightedGGNN, self).__init__(aggr='mean')  # mean 聚合
        self.lin = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_channels)
        )
        self.gru = nn.GRUCell(out_channels, out_channels)
        # 为每种边类型学习一个标量权重（也可以用 embedding -> 向量）
        # 我们保存未约束的原始参数，然后在 message 中用 softplus/sigmoid 映射为正值（可选）
        self.edge_type_weight = nn.Parameter(torch.randn(num_edge_types))

    def forward(self, x, edge_index, edge_type):
        # x: [num_nodes, in_channels]
        # edge_index: [2, num_edges]
        # edge_type: [num_edges]  每条边的类型 id (0..num_edge_types-1)
        x = self.lin(x)
        for _ in range(3):
            # 把 edge_type 传给 propagate，message 中会接收它
            m = self.propagate(edge_index, x=x, edge_type=edge_type)
            x = self.gru(m, x)
        return x

    def message(self, x_j, edge_type):
        # x_j: [num_edges, out_channels]（按 edge_index 列顺序）
        # edge_type: [num_edges]  整数索引
        # 将原始权重映射为正值（softplus）或限制在 (0,1)（sigmoid），可以避免符号/爆炸问题
        w = F.softplus(self.edge_type_weight[edge_type]).view(-1, 1)  # [num_edges, 1]
        return w * x_j

class GGNNNet(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim,num_edge_types):
        super().__init__()
        self.ggnn = WeightedGGNN(in_channels, out_channels, hidden_dim,num_edge_types)
        self.fc = nn.Sequential(
            nn.Linear(out_channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
            )
        self.w_imp = nn.Parameter(torch.tensor(1.0))


    def forward(self, data):
        x = self.ggnn(data.x, data.edge_index, data.edge_type)
        alpha = torch.ones(x.size(0), 1, device=x.device)
         # 关键行：通过 softplus 保证 > 0
        w_pos = F.softplus(self.w_imp)
        if data.diff_idx is not None:
            alpha[data.diff_idx] = 1+w_pos
        x = (alpha * x).sum(dim=0, keepdim=True) / alpha.sum()
        out = self.fc(x)
        return F.log_softmax(out, dim=1)
    
def ggnn_train(dataset, val_dataset=None):
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True)  
    val_loader = DataLoader(val_dataset, batch_size=8) if val_dataset is not None else None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = GGNNNet(768, 128, 256, num_edge_types=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(20):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss=round(total_loss / len(train_loader), 4)
        print(f"Epoch {epoch}, Avg Loss: {avg_train_loss}")

        # ---------- Validate ----------
        if val_loader is not None:
            model.eval()
            val_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    out = model(batch)
                    loss = criterion(out, batch.y)
                    val_loss += loss.item()
                    preds = out.argmax(dim=1)
                    correct += (preds == batch.y).sum().item()
                    total += batch.y.size(0)
            avg_val_loss = val_loss / len(val_loader)
            val_acc = correct / total
            print(f"Epoch {epoch:02d}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.4f}")
        else:
            print(f"Epoch {epoch:02d}: Train Loss={avg_train_loss:.4f}")

    # ---------- After Training ----------


    with torch.no_grad():
        mapped = F.softplus(model.ggnn.edge_type_weight)
        print("Trained edge-type weights (softplus):", mapped)
        print(f"Trained node importance weight w_imp (softplus): {model.w_imp.item():.4f}")

    torch.save(model.state_dict(), 'ggnn_model.pt')










