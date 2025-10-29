
import json
import os
import sys
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
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
from torch_geometric.nn import global_mean_pool
from torch_scatter import scatter_add
from torch_geometric.data import Batch
from sklearn.metrics import f1_score, confusion_matrix
from torch.utils.data import WeightedRandomSampler

tokenizer = AutoTokenizer.from_pretrained("huggingface/CodeBERTa-small-v1")
model = AutoModel.from_pretrained("huggingface/CodeBERTa-small-v1")
model.eval()  # 不训练，只做推理

executor = ThreadPoolExecutor(max_workers=4)  # 根据CPU/GPU核数调整

def _generate_vector(code):
    inputs = tokenizer(code,
                       return_tensors="pt",
                       max_length=512,
                       truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        code_embedding = outputs.last_hidden_state.mean(dim=1)
        return code_embedding[0]

async def generate_vector_async(code):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, _generate_vector, code)

async def process_node(value, G):
    mix = G.nodes[value]["nodeType"] + ":" + "".join(G.nodes[value]["sourceLines"])
    mix = re.sub(r'\s+', ' ', mix)
    vec = await generate_vector_async(mix)
    return vec

async def process_graph_line(line_idx, line):
    try:
        print(f"processing line {line_idx} ")
        data = json.loads(line.strip())
        G = json_graph.node_link_graph(data["relate_graph"])

        diff_nodes = set(data["diff_node"])
        label = torch.tensor([data["label"]], dtype=torch.long)
        nodes_list = list(G.nodes())
        node_embeding = []

        diff_node_idx = [nodes_list.index(node) for node in diff_nodes]
        buggy_node_idx = [nodes_list.index(node) for node in data["buggy_nodes"]]

        edge_types = []
        edge_index = [[], []]
        for u, v in G.edges():
            types_list = [*(G.get_edge_data(u, v))]
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

        # 异步生成节点向量
        tasks = [process_node(node, G) for node in nodes_list]
        node_embeding = await asyncio.gather(*tasks)

        data_item = Data(
            x=torch.stack(node_embeding),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            edge_type=torch.tensor(edge_types, dtype=torch.long),
            diff_idx=torch.tensor(diff_node_idx, dtype=torch.long),
            buggy_idx=torch.tensor(buggy_node_idx, dtype=torch.long),
            y=label
        )
        return data_item
    except Exception as e:
        print(f"error line {line_idx}: {e}")
        return None

async def read_graph_async():
    dataset = []
    name = 'train'
    data_path = Path(CONSTANTS.repository_dir + '/' + name + '_graph_dataset.jsonl')

    with open(data_path, 'r', encoding='utf-8') as f:
        tasks = [process_graph_line(idx, line) for idx, line in enumerate(f)]
        results = await asyncio.gather(*tasks)

    dataset = [item for item in results if item is not None]
    torch.save(dataset, name + '_graph_dataset.pt')


# # 运行异步函数
# asyncio.run(read_graph_async())


# -------------------------
# 自定义带权GGNN
# -------------------------
class WeightedGGNN(MessagePassing):
    def __init__(self, in_channels, out_channels, hidden_dim,num_edge_types):
        super(WeightedGGNN, self).__init__(aggr='mean')  # mean 聚合
        self.lin = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.4),
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
            x = F.layer_norm(x, x.size()[1:])
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
            nn.Linear(out_channels, out_channels//2),
            nn.ReLU(),
            #nn.Dropout(p=0.3),
            nn.Linear(out_channels//2, out_channels//4),
            nn.ReLU(),
            #nn.Dropout(p=0.3),
            nn.Linear(out_channels//4, 2)
            )
        self.w_imp = nn.Parameter(torch.tensor(1.0))


    def forward(self, data):
        x = self.ggnn(data.x, data.edge_index, data.edge_type)

        alpha = torch.ones(x.size(0), 1, device=x.device)
         # 通过 softplus 保证 > 0
        w_pos = F.softplus(self.w_imp)
        if data.diff_idx is not None:
            alpha[data.diff_idx] = 1+w_pos
        # 计算每张图 alpha * x 的和
        x_weighted_sum = scatter_add(alpha * x, data.batch, dim=0)
        #计算每张图 alpha 的和
        alpha_sum = scatter_add(alpha, data.batch, dim=0)  # [num_graphs, 1]
        # 每个图的加权平均
        x_graph = x_weighted_sum / alpha_sum
        out = self.fc(x_graph)
        return out
    

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.95, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # 正样本权重
        self.gamma = gamma  # 聚焦参数
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return loss.mean()
        return loss.sum()

def collate_with_diff(data_list):
    # 先用 PyG 默认方式合并图（自动处理 x, edge_index, y, batch 等）
    batch = Batch.from_data_list(data_list)
    
    # 现在动态计算 diff_idx 的全局偏移
    diff_idx_list = []
    node_ptr = batch.ptr  # ptr[i] 是第 i 个图的起始节点索引，长度为 len(data_list)+1
    
    for i, data in enumerate(data_list):
        if hasattr(data, 'diff_idx') and data.diff_idx is not None:
            # 将局部索引转为全局索引
            global_diff_idx = data.diff_idx + node_ptr[i]
            diff_idx_list.append(global_diff_idx)
    
    if diff_idx_list:
        batch.diff_idx = torch.cat(diff_idx_list, dim=0)

    return batch

def ggnn_train(tran_dataset, val_dataset=None):
    bs=8
    lr = 1e-4
    sample_weights = [1.0 if data.y.item() == 0 else 38 for data in tran_dataset]  # 正样本权重大
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(tran_dataset, batch_size=bs, sampler=sampler,collate_fn=collate_with_diff)  
    val_loader = DataLoader(val_dataset, batch_size=bs,collate_fn=collate_with_diff) if val_dataset is not None else None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = GGNNNet(768, 256, 512, num_edge_types=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=1e-4)
    criterion = FocalLoss(alpha=0.9, gamma=1.5)
    best_f1 = 0.0

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
            #print(f"Batch Loss: {loss.item():.4f}")
        avg_train_loss=round(total_loss / len(train_loader), 4)
        print(f"Epoch {epoch}, Avg Loss: {avg_train_loss}")

        # ---------- Validate ----------

        best_model_path = "best_ggnn_model.pt"
        if val_loader is not None:
            model.eval()
            val_loss = 0
            correct = 0
            total = 0
            all_preds = []
            all_labels = []

            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    out = model(batch)
                    loss = criterion(out, batch.y)
                    val_loss += loss.item()

                    preds = out.argmax(dim=1)
                    correct += (preds == batch.y).sum().item()
                    total += batch.y.size(0)

                    # 保存预测与真实标签，用于计算 F1
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(batch.y.cpu().numpy())

            avg_val_loss = val_loss / len(val_loader)
            val_acc = correct / total

            val_f1 = f1_score(all_labels, all_preds, average='binary')

            tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()

            print(f"Epoch {epoch:02d}: "
                f"Val Loss={avg_val_loss:.4f}, "
                f"Val Acc={val_acc:.4f}, "
                f"Val F1={val_f1:.4f}, "
                f"TP={tp}, FN={fn}, FP={fp}, TN={tn}")
            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save(model.state_dict(), best_model_path)
                print(f"✨ New best model saved at epoch {epoch} (F1={val_f1:.4f})")
        else:
            print(f"Epoch {epoch:02d}: Train Loss={avg_train_loss:.4f}")

    # ---------- After Training ----------


    with torch.no_grad():
        mapped = F.softplus(model.ggnn.edge_type_weight)
        print("Trained edge-type weights (softplus):", mapped)
        print(f"Trained node importance weight w_imp (softplus): {model.w_imp.item():.4f}")





def tran_start():
    tran_dataset = torch.load('train_graph_dataset.pt', weights_only=False)
    valid_dataset=torch.load('valid_graph_dataset.pt', weights_only=False)
    ggnn_train(tran_dataset, valid_dataset)


tran_start()