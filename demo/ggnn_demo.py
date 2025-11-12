
import json
import os
import sys
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from tqdm import tqdm
sys.path.append(str(Path(__file__).resolve().parent.parent))
from util.CCG_build import create_graph
from networkx.readwrite import json_graph
from util.util import set_default, make_needed_dir, dump_jsonl, graph_to_json,CONSTANTS,preprocess_code_line
from util.make_slicing import CCGBuilder
import re
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing,TopKPooling
from torch_geometric.utils import add_self_loops, softmax
from torch_geometric.loader import DataLoader
from transformers import AutoTokenizer, AutoModel
from torch_geometric.nn import global_mean_pool,global_add_pool
from torch_scatter import scatter_add,scatter_mean
from torch_geometric.data import Batch
from sklearn.metrics import f1_score, confusion_matrix
from torch.utils.data import WeightedRandomSampler
import random
import csv
import pandas as pd
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModel.from_pretrained("microsoft/codebert-base").to(device)
model.eval()  # 不训练，只做推理

def _generate_vector(code):
    inputs = tokenizer(code,
                       return_tensors="pt", # 返回 PyTorch 张量
                       max_length=512,      # 设置最大长度
                       padding='max_length', # 建议填充，确保张量形状一致
                       truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        code_embeddings = outputs.last_hidden_state[:, 0, :]
    return [embedding.cpu() for embedding in code_embeddings]

async def process_graph_line(line_idx, line,name,features):
    # try:
        data = json.loads(line.strip())
        G = json_graph.node_link_graph(data["relate_graph"])

        diff_nodes = set(data["diff_node"])
        label = torch.tensor([data["label"]], dtype=torch.long)
        nodes_list = sorted(G.nodes())


        diff_node_idx = [nodes_list.index(node) for node in diff_nodes]
        buggy_node_idx = [nodes_list.index(node) for node in data["buggy_nodes"]]
        codes_to_process = []
        for value in nodes_list:
            mix = G.nodes[value]["nodeType"] + ":" + "".join(G.nodes[value]["sourceLines"])
            codes_to_process.append(preprocess_code_line(mix))
        node_embedding = _generate_vector(codes_to_process)

        edge_types = []
        edge_index = [[], []]
        cfg_num=0
        cdg_num=0
        ddg_num=0
        for u, v in G.edges():
            types_list = [*(G.get_edge_data(u, v))]
            if 'CFG' in types_list:
                edge_types.append(0)
                edge_index[0].append(nodes_list.index(u))
                edge_index[1].append(nodes_list.index(v))
                cfg_num+=1

            if 'CDG' in types_list:
                edge_types.append(1)
                edge_index[0].append(nodes_list.index(u))
                edge_index[1].append(nodes_list.index(v))
                cdg_num+=1

            if 'DDG' in types_list:
                edge_types.append(2)
                edge_index[0].append(nodes_list.index(u))
                edge_index[1].append(nodes_list.index(v))
                ddg_num+=1


        expert_features=[]
        msg=""
        expert_features.extend([len(data['add_codes']),len(data['remove_codes']),len(diff_nodes),len(nodes_list),cfg_num,cdg_num,ddg_num,nodes_list[-1]-nodes_list[0],max(diff_nodes)-min(diff_nodes)])

        hash_feature=features[data["commit_hash"]]
        msg=hash_feature['commit_message']

        diff_embeding=joint_embedding(msg,data)
        data_item = Data(
            commit_hash=data["commit_hash"],
            expert_features=torch.tensor(expert_features, dtype=torch.float),
            x=torch.stack(node_embedding),
            diff_embeding=diff_embeding,
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            edge_type=torch.tensor(edge_types, dtype=torch.long),
            diff_idx=torch.tensor(diff_node_idx, dtype=torch.long),
            buggy_idx=torch.tensor(buggy_node_idx, dtype=torch.long),
            y=label
        )
        print(f"processed line {line_idx} ")
        return data_item
    # except Exception as e:
    #     print(f"error line {line_idx}: {e}")
    #     return None

async def read_graph_async(name):
    dataset = []
    data_path = Path(CONSTANTS.repository_dir + '/' + name + '_graph_dataset.jsonl')
    with open(CONSTANTS.repository_dir + '/'+'features_' + name + '.json', 'r', encoding='utf-8') as f:
        features = json.load(f)
    with open(data_path, 'r', encoding='utf-8') as f:
        tasks = [process_graph_line(idx, line,name,features) for idx, line in enumerate(f)]
        results = await asyncio.gather(*tasks)

    dataset = [item for item in results if item is not None]
    torch.save(dataset, name + '_graph_dataset.pt')


def joint_embedding(msg,data):
    msg_token=tokenizer.tokenize(preprocess_code_line(msg))
    add_line_token=tokenizer.tokenize('[ADD]'.join([preprocess_code_line(d) for d in data['add_codes']]))
    del_line_token=tokenizer.tokenize('[DEL]'.join([preprocess_code_line(d) for d in data['remove_codes']]))
    joint_token=msg_token+['[ADD]']+add_line_token+['[DEL]']+del_line_token
    input_token=[tokenizer.cls_token]+joint_token[:512-2]+[tokenizer.sep_token]
    input_ids = tokenizer.convert_tokens_to_ids(input_token)
    attention_mask_list = [1] * len(input_ids)
    max_length = 512
    padding_length = max_length - len(input_ids)

    # 填充 input_ids 和 attention_mask
    input_ids += [tokenizer.pad_token_id] * padding_length
    attention_mask_list += [0] * padding_length

    # 3. 转换为 PyTorch 张量，并增加 Batch 维度
    # model 需要 (Batch Size, Sequence Length) 格式的输入
    inputs = {
        'input_ids': torch.tensor([input_ids], dtype=torch.long),
        'attention_mask': torch.tensor([attention_mask_list], dtype=torch.long)
    }

    # (可选但推荐) 如果模型在 GPU 上，将输入张量移动到 GPU
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # 4. 模型推理
    with torch.no_grad():
        # 使用 **inputs 解包字典，将 input_ids 和 attention_mask 作为关键字参数传入
        outputs = model(**inputs) 
        code_embedding = outputs.last_hidden_state[:, 0, :]
        return code_embedding[0]


# # 运行异步函数
asyncio.run(read_graph_async("train"))


# -------------------------
# 自定义带权GGNN
# -------------------------
def graph_mixup(data1, data2):
    """合并两个 Data 图，保留原图结构，更新 edge_index 和 diff_idx"""
    num_nodes1 = data1.x.size(0)
    
    # === 1. 合并节点特征 ===
    x = torch.cat([data1.x, data2.x], dim=0)  # [N1 + N2, F]
    
    # === 2. 更新边索引 ===
    edge_index2 = data2.edge_index + num_nodes1  # 把第二个图的索引整体偏移
    edge_index = torch.cat([data1.edge_index, edge_index2], dim=1)
    
    # === 3. 合并 edge_type ===
    edge_type = torch.cat([data1.edge_type, data2.edge_type], dim=0)
    
    # === 4. 合并 diff_idx（需要偏移） ===
    diff_idx2 = data2.diff_idx + num_nodes1
    diff_idx1 = data1.diff_idx
    diff_idx = torch.cat([diff_idx1, diff_idx2], dim=0)

       # === 5. 添加图间 diff 边（每个 diff_idx1 都至少连一条） ===
    if len(diff_idx1) > 0 and len(diff_idx2) > 0:
        src_list = []
        dst_list = []
        for node1 in diff_idx1:
            node2 = torch.randint(num_nodes1, num_nodes1+data2.x.size(0), (1,)).item()
            src_list.append(node1)
            dst_list.append(node2)

        # 转成 tensor
        src = torch.tensor(src_list, dtype=torch.long)
        dst = torch.tensor(dst_list, dtype=torch.long)

        cross_edges = torch.stack([src, dst], dim=0)

        # 拼接进原图边集合
        edge_index = torch.cat([edge_index, cross_edges], dim=1)

        # 为跨图边设置类型
        num_edge_types = 3  # 总共有 3 种边类型
        cross_types = torch.randint(0, num_edge_types, (cross_edges.size(1),), dtype=torch.long)
        edge_type = torch.cat([edge_type, cross_types], dim=0)

    y = torch.tensor([1]) 
    
    return Data(x=x, edge_index=edge_index, edge_type=edge_type, diff_idx=diff_idx, buggy_idx=data1.buggy_idx,y=y)



def stom_oversample(dataset, minority_class=1, target_ratio=0.5):
    """对图分类数据集进行过采样"""
    # 分开正负样本
    minority = [d for d in dataset if d.y.item() == minority_class]
    majority = [d for d in dataset if d.y.item() != minority_class]

    new_data = []
    while (1-target_ratio)*(len(minority) + len(new_data)) < target_ratio*len(majority) :
        # 从负样本集中随机取一个
        
        d1 = random.choice(minority)
        d2 = random.choice(majority)
        d3=random.choice(majority)
        new_data.append(graph_mixup(d1, d2))
        new_data.append(graph_mixup(d3, d1))
        for _ in range(3):
            #丢弃负样本
            d1_idx = random.randrange(len(majority))
            del majority[d1_idx]

    new_dataset = majority + minority + new_data
    print("positive",len(new_data)+len(minority),"negative",len(majority))
    random.shuffle(new_dataset)

    return new_dataset

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.95, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # 正样本权重
        self.gamma = gamma  # 聚焦参数
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return loss.mean()
        return loss.sum()
class WeightedGGNN(MessagePassing):
    def __init__(self, in_channels, out_channels, hidden_dim,num_edge_types):
        super(WeightedGGNN, self).__init__(aggr='sum')  # sum 聚合
        self.lin = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            # nn.Linear(hidden_dim, out_channels)
        )
        self.gru = nn.GRUCell(out_channels, out_channels)
        # 为每种边类型学习一个标量权重（也可以用 embedding -> 向量）
        # 我们保存未约束的原始参数，然后在 message 中用 softplus/sigmoid 映射为正值（可选）
        self.edge_type_weight = nn.Parameter(torch.randn(num_edge_types))
        self.gammar=nn.Parameter(torch.tensor(1.5))
        
    def forward(self, x, edge_index, edge_type,diff_idx=None):
        # x: [num_nodes, in_channels]
        # edge_index: [2, num_edges]
        # edge_type: [num_edges]  每条边的类型 id (0..num_edge_types-1)
        x = self.lin(x)
        if diff_idx is not None:
            x[diff_idx] = x[diff_idx] * self.gammar
        for _ in range(3):
            # 把 edge_type 传给 propagate，message 中会接收它
            m = self.propagate(edge_index, x=x, edge_type=edge_type)
            x = self.gru(m, x)
        #x = F.layer_norm(x, x.size()[1:])
        return x

    def message(self, x_j, edge_type):
        # x_j: [num_edges, out_channels]（按 edge_index 列顺序）
        # edge_type: [num_edges]  整数索引
        # 将原始权重映射为正值（softplus）或限制在 (0,1)（sigmoid），可以避免符号/爆炸问题
        w = F.softplus(self.edge_type_weight[edge_type]).view(-1, 1)  # [num_edges, 1]
        msg=w * x_j
        return msg 

class GGNNNet(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim,num_edge_types,batch_size=8):
        super().__init__()
        self.ggnn = WeightedGGNN(in_channels, out_channels, hidden_dim,num_edge_types)
        self.fc = nn.Sequential(
            nn.Linear(out_channels*2, 768),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 2),
            )
        
        self.align=nn.Sequential(
            nn.Linear(9, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_channels)
        )
        
        self.w_imp = nn.Parameter(torch.tensor(1.0))
        self.batch_size=batch_size
        self.index_conv = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)


    def forward(self, data):
        x = self.ggnn(data.x, data.edge_index, data.edge_type,data.diff_idx)
        # alpha = torch.ones(x.size(0), 1, device=x.device)
        #  # 通过 softplus 保证 > 0
        # w_pos = F.softplus(self.w_imp)
        # if data.diff_idx is not None:
        #     alpha[data.diff_idx] = 1+w_pos
        # alpha_sum = scatter_add(alpha, data.batch, dim=0)
        # x = (alpha * x) / alpha_sum[data.batch] # [num_graphs, 1]

        # for i in range(self.batch_size):
        #     node_idx = (data.batch == i).nonzero(as_tuple=True)[0].to(x.device)
        #     if node_idx.numel() < 3:  # 如果图太小，跳过卷积步骤
        #         continue
        #     x_i = x[node_idx]  # [num_nodes_i, C]
        #     x_i = x_i.unsqueeze(0).transpose(1,2)  # [1, C, N_i]
        #     x_i = self.index_conv(x_i)
        #     x_i = F.relu(x_i)
        #     x_i = x_i.transpose(1,2).squeeze(0)
        #     x[node_idx] = x_i
        

        # x_graph = scatter_add(x, data.batch, dim=0) 
        x_graph=global_add_pool(x,data.batch)
        x_graph = F.normalize(x_graph, p=2, dim=1)
        diff_feature=data.diff_embeding.reshape(-1, 768)
        diff_feature=F.normalize(diff_feature, p=2, dim=1)
        # expert_feature=self.align(data.expert_features.float().reshape(-1, 9))
        # expert_feature=F.normalize(expert_feature, p=2, dim=1)
        x_graph=torch.cat([x_graph,diff_feature],dim=1)
        return x_graph
    



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
def compute_contrastive_loss(embeddings, labels, temperature=0.5):
    """
    embeddings: [B, D] 图嵌入向量
    labels: [B] 每个图的标签
    """
    sim_matrix = torch.matmul(embeddings, embeddings.T) / temperature  # [B, B]

    # 构造正样本 mask
    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(embeddings.device)#[B,B]
    # 防止梯度爆炸
    logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
    sim_matrix = sim_matrix - logits_max.detach()
    exp_sim = torch.exp(sim_matrix) * (1 - torch.eye(labels.size(0), device=embeddings.device))

    log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-9)

    mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-9)

    # InfoNCE 损失
    loss = -mean_log_prob_pos.mean()
    return loss



def ggnn_train(train_dataset, val_dataset=None,contras_learn=True):
    bs=8
    lr = 1e-4
    sample_weights = [1.0 if data.y.item() == 0 else 38 for data in train_dataset]  # 正样本权重大
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=bs, sampler=sampler,collate_fn=collate_with_diff)  

    # train_dataset = stom_oversample(train_dataset, minority_class=1, target_ratio=0.5)
    # sample_weights = [1.0 if data.y.item() == 0 else 1 for data in train_dataset]  # 正样本权重大
    # sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    # train_loader = DataLoader(train_dataset, batch_size=bs, sampler=sampler,collate_fn=collate_with_diff)

    val_loader = DataLoader(val_dataset, batch_size=bs,collate_fn=collate_with_diff) if val_dataset is not None else None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = GGNNNet(768, 768, 768, num_edge_types=3,batch_size=bs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=1e-4)
    criterion = FocalLoss(alpha=0.5, gamma=1)
    best_f1 = 0.34

    for epoch in range(100):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            x_graph = model(batch)
            out = model.fc(x_graph)
            cls_loss = criterion(out, batch.y)
            if contras_learn:
                contrastive_loss = compute_contrastive_loss(x_graph, batch.y)
                loss = cls_loss + contrastive_loss
            else:
                loss = cls_loss

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
            all_hashes = []
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    x_graph = model(batch)
                    out = model.fc(x_graph)
                    loss = criterion(out, batch.y)
                    val_loss += loss.item()

                    preds = out.argmax(dim=1)
                    correct += (preds == batch.y).sum().item()
                    total += batch.y.size(0)

                    # 保存预测与真实标签，用于计算 F1
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(batch.y.cpu().numpy())
                    all_hashes.extend(batch.commit_hash)
            
            avg_val_loss = val_loss / len(val_loader)

            val_f1 = process_results(all_hashes, all_preds,what="valid",max_f1=best_f1)

            print(f"Epoch {epoch:02d}: "
                f"Val Loss={avg_val_loss:.4f}, ")
            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save(model.state_dict(), best_model_path)
                print(f"✨ New best model saved at epoch {epoch} (F1={val_f1:.4f})")
                print("=====================================Test on best model=============================================")
                test_start(model)
        else:
            print(f"Epoch {epoch:02d}: Train Loss={avg_train_loss:.4f}")
        #forget_some_params(model, forget_ratio=0.1)

    # ---------- After Training ----------


        # with torch.no_grad():
        #     mapped = F.softplus(model.ggnn.edge_type_weight)
        #     print("Trained edge-type weights (softplus):", mapped)
        #     print(f"Trained node importance weight w_imp (softplus): {model.w_imp.item():.4f}")



def forget_some_params(model, forget_ratio=0.05):
    """随机遗忘模型中一部分参数"""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.requires_grad:
                # 按元素随机遗忘
                mask = torch.rand_like(param) < forget_ratio
                param[mask] = torch.randn_like(param[mask]) * 0.02  # 重新初始化为小随机数

def tran_start():
    tran_dataset = torch.load('train_graph_dataset.pt', weights_only=False)
    valid_dataset=torch.load('valid_graph_dataset.pt', weights_only=False)
    ggnn_train(tran_dataset, valid_dataset)



def test_start(model=None):
    bs=8
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_dataset = torch.load('test_graph_dataset.pt', weights_only=False)
    test_loader = DataLoader(test_dataset, batch_size=bs,collate_fn=collate_with_diff)
    if model is None:
        model=GGNNNet(768, 768, 768, num_edge_types=3,batch_size=bs).to(device)
        model.load_state_dict(torch.load('best_ggnn_model.pt'))


    print(f"Using device: {device}")
    criterion = FocalLoss(alpha=0.5, gamma=2)
    model.eval()
    val_loss = 0
    all_preds = []
    all_labels = []
    all_hashes = []
    with torch.no_grad():
            for batch in test_loader:
                    batch = batch.to(device)
                    x_graph = model(batch)
                    out = model.fc(x_graph)
                    loss = criterion(out, batch.y)
                    val_loss += loss.item()

                    preds = out.argmax(dim=1)

                    # 保存预测与真实标签，用于计算 F1
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(batch.y.cpu().numpy())
                    all_hashes.extend(batch.commit_hash)
            avg_val_loss = val_loss / len(test_loader)
            print(f"Test Loss={avg_val_loss:.4f}")
            process_results(all_hashes, all_preds)

def process_results(all_hashes, all_preds,what="test",max_f1=0):
    n=len(all_hashes)
    hashes=[]
    real_label=[]
    pred_label=[]
    with open(Path(CONSTANTS.repository_dir)/f"features_{what}.json", 'r',encoding='utf-8') as f:
        data = json.load(f)
        pre_hash=None
        for i in range(n):
            hash=all_hashes[i]
            if hash==pre_hash and all_preds[i]==1:
                pred_label[-1]=1
            elif hash!=pre_hash:
                hashes.append(hash)
                pred_label.append(all_preds[i])
                real_label.append(int(float(data[hash]["is_buggy_commit"])))
                pre_hash=hash
                
    output_path = f"./{what}_prediction_results.csv"

    real_label = np.array(real_label).astype(int)
    pred_label = np.array(pred_label).astype(int)
    f1 = f1_score( real_label, pred_label, average='binary')
    tn, fp, fn, tp = confusion_matrix(real_label, pred_label).ravel()
    print(f" F1={f1:.4f}, "
                f"TP={tp}, FN={fn}, FP={fp}, TN={tn}")
    if f1>max_f1:
        max_f1=f1
        df = pd.DataFrame({
        "commit_hash": hashes,
        "real_label": real_label,
        "pred_label": pred_label
        })
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"✅ 预测结果已保存到: {output_path}")
        return f1
    return max_f1
# tran_start()
# test_start()

