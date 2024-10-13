import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import BertModel
from models.gat import GAT
from models.aggregator import MeanAggregator, MaxAggregator, AttentionAggregator, MultimodalGraphReadout
import pennylane as qml

def get_padding_mask(lengths, max_len=None):
    bsz = len(lengths)
    if not max_len:
        max_len = lengths.max()
    mask = torch.zeros((bsz, max_len))
    for i in range(bsz):
        index = torch.arange(int(lengths[i].item()), max_len)
        mask[i] = mask[i].index_fill_(0, index, -1e9)

    return mask



qbits=6
def state_prepare(input, qbits):
    # 输入12个维度的向量，需要拆分成三部分，每部分4个维度
    text_data = input[:, 0:4]  # 前四个维度编码到1,2量子比特（文本模态）
    middle_data = input[:, 4:8]  # 中间四个维度编码到3,4量子比特
    end_data = input[:, 8:12]  # 最后四个维度编码到5,6量子比特

    # 手动编码：为每个量子比特分配两个特征维度，并应用不同的旋转门

    # 对量子比特0和1编码前4个维度（文本模态）
    qml.RY(text_data[:, 0], wires=0)
    qml.RZ(text_data[:, 1], wires=0)
    qml.RY(text_data[:, 2], wires=1)
    qml.RZ(text_data[:, 3], wires=1)

    # 对量子比特2和3编码中间4个维度
    qml.RY(middle_data[:, 0], wires=2)
    qml.RZ(middle_data[:, 1], wires=2)
    qml.RY(middle_data[:, 2], wires=3)
    qml.RZ(middle_data[:, 3], wires=3)

    # 对量子比特4和5编码最后4个维度
    qml.RY(end_data[:, 0], wires=4)
    qml.RZ(end_data[:, 1], wires=4)
    qml.RY(end_data[:, 2], wires=5)
    qml.RZ(end_data[:, 3], wires=5)

def conv(b1, b2, params):
    qml.RZ(-torch.pi / 2, wires=b2)
    qml.RZ(params[0], wires=b1)
    qml.CNOT([b2, b1])
    qml.RY(params[1], wires=b2)
    qml.RY(params[2], wires=b2)
    qml.RZ(torch.pi / 2, wires=b1)


# 定义一个具有6个量子比特的量子设备
dev6 = qml.device('default.qubit', wires=6)


def conv(b1, b2, params):
    qml.RZ(-torch.pi / 2, wires=b2)
    qml.RZ(params[0], wires=b1)
    qml.CNOT([b2, b1])
    qml.RY(params[1], wires=b2)
    qml.RY(params[2], wires=b2)
    qml.RZ(torch.pi / 2, wires=b1)

@qml.qnode(dev6, interface='torch', diff_method='backprop')
def qfnn6(weights, inputs):
    # 准备6个量子比特的量子态
    state_prepare(inputs, 6)

    # 应用卷积和池化操作
    conv(0, 1, weights[0:3])

    conv(2, 3, weights[3:6])

    conv(4, 5, weights[6:9])

    # 返回每个量子比特的 Pauli-Z 算符的期望值
    return [qml.expval(qml.PauliZ(i)) for i in range(6)]



class QFNN(nn.Module):
    def __init__(self, qbits):
        super().__init__()
        self.qbits = qbits
        self.qcnn_layer = qml.qnn.TorchLayer(qfnn6, weight_shapes={'weights': 9})

    def forward(self, x):
        # 前向传播，通过量子层处理输入
        return self.qcnn_layer(x)









def qmfnn_state_prepare(input, qbits):
    # 对输入进行归一化，用于振幅嵌入
    input = input / input.sum(dim=-1, keepdim=True)
    # 使用振幅嵌入（AmplitudeEmbedding）来准备量子态
    qml.AmplitudeEmbedding(input, wires=range(qbits), normalize=True)

dev_qmfnn= qml.device('default.qubit', wires=6)
# 定义量子线路
@qml.qnode(dev_qmfnn, interface='torch', diff_method='backprop')
def qmfnn6(weights, inputs):
    # 准备6个量子比特的量子态
    qmfnn_state_prepare(inputs, 6)

    # 第一步：对每个量子比特应用RZ门
    for i in range(6):
        qml.RZ(weights[i], wires=i)  # 对1-6号量子比特应用RZ门

    # 第二步：应用CNOT门（2控制1，3控制2，... 1控制6）
    qml.CNOT(wires=[1, 0])  # 2控制1
    qml.CNOT(wires=[2, 1])  # 3控制2
    qml.CNOT(wires=[3, 2])  # 4控制3
    qml.CNOT(wires=[4, 3])  # 5控制4
    qml.CNOT(wires=[5, 4])  # 6控制5
    qml.CNOT(wires=[0, 5])  # 1控制6

    # 第三步：对每个量子比特应用RY门
    for i in range(6):
        qml.RY(weights[6 + i], wires=i)  # 对1-6号量子比特应用RY门

    # 返回每个量子比特的 Pauli-Z 算符的期望值
    return [qml.expval(qml.PauliZ(i)) for i in range(6)]


# 修改QFNN类，支持6个量子比特
class QMFNN(nn.Module):
    def __init__(self, qbits):
        super().__init__()
        self.qbits = qbits

        self.qcnn_layer = qml.qnn.TorchLayer(qmfnn6, weight_shapes={'weights': 12})

    def forward(self, x):
        # 前向传播，通过量子层处理输入
        return self.qcnn_layer(x)














class MultimodalGraphFusionNetwork(nn.Module):
    def __init__(self, config):
        super(MultimodalGraphFusionNetwork, self).__init__()
        dt, da, dv = config["t_size"], config["a_size"], config["v_size"]
        h = config["hidden_size"]
        m_dim = 3*h

        self.config = config
        self.h = h
        self.encoder_t = self._get_encoder(modality='t')
        self.encoder_v = self._get_encoder(modality='v')
        self.encoder_a = self._get_encoder(modality='a')

        self.gat_t = GAT(input_dim=h,
                        gnn_dim=h // config["num_gnn_heads"],
                        num_layers=config["num_gnn_layers"],
                        num_heads=config["num_gnn_heads"],
                        dropout=config["dropout_gnn"],
                        leaky_alpha=0.2)
        self.gat_v = GAT(input_dim=h,
                         gnn_dim=h // config["num_gnn_heads"],
                         num_layers=config["num_gnn_layers"],
                         num_heads=config["num_gnn_heads"],
                         dropout=config["dropout_gnn"],
                        leaky_alpha=0.2)
        self.gat_a = GAT(input_dim=h,
                         gnn_dim=h // config["num_gnn_heads"],
                         num_layers=config["num_gnn_layers"],
                         num_heads=config["num_gnn_heads"],
                         dropout=config["dropout_gnn"],
                        leaky_alpha=0.2)
        self.gat_m = GAT(input_dim=h,
                       gnn_dim=h // config["num_gnn_heads"],
                       num_layers=config["num_gnn_layers"],
                       num_heads=config["num_gnn_heads"],
                       dropout=config["dropout_gnn"],
                        leaky_alpha=0.2)

        self.project_t = nn.Linear(dt, h)
        self.project_v = nn.Linear(dv*2, h)
        self.project_a = nn.Linear(da*2, h)

        self.readout_t = AttentionAggregator(h)
        self.readout_v = AttentionAggregator(h)
        self.readout_a = AttentionAggregator(h)
        self.readout_m = MultimodalGraphReadout(m_dim, self.readout_t, self.readout_v, self.readout_a)

        self.fc_out = nn.Linear(m_dim, 1)
        self.ret = nn.Linear(128, 4)
        self.rev = nn.Linear(128, 4)
        self.rea = nn.Linear(128, 4)
        self.rem = nn.Linear(384, 64)

        self.ret2 = nn.Linear(130, 128)
        self.rev2 = nn.Linear(130, 128)
        self.rea2 = nn.Linear(130, 128)
        self.rem2 = nn.Linear(390, 384)





        self.dropout_m = nn.Dropout(config["dropout"])
        self.dropout_t = nn.Dropout(config["dropout_t"])
        self.dropout_v = nn.Dropout(config["dropout_v"])
        self.dropout_a = nn.Dropout(config["dropout_a"])

        qbits = 6
        self.qfnn = QFNN(qbits)
        self.qmfnn = QMFNN(qbits)




    def _get_encoder(self, modality='t', *args):
        if modality == 't':
            return BertModel.from_pretrained(self.config["bert_path"])
        elif modality == 'v':
            return nn.LSTM(self.config["v_size"], self.config["v_size"], batch_first=True, bidirectional=True)
        elif modality == 'a':
            return nn.LSTM(self.config["a_size"], self.config["a_size"], batch_first=True, bidirectional=True)
        else:
            raise ValueError('modality should be t or v or a!')

    def _lstm_encode(self, inputs, lengths, lstm, h_size):
        batch_size, t, _ = inputs.size()
        h0 = torch.zeros(2, batch_size, h_size).to(inputs.device)
        c0 = torch.zeros(2, batch_size, h_size).to(inputs.device)

        pack = pack_padded_sequence(inputs, lengths.cpu(), batch_first=True)
        lstm.flatten_parameters()
        out, _ = lstm(pack, (h0, c0))
        out, lens = pad_packed_sequence(out, batch_first=True)

        memory = out.contiguous().view(batch_size * t, -1)
        index = lens - 1 + torch.arange(batch_size) * t
        if torch.cuda.is_available():
            index = index.cuda()

        last_h = torch.index_select(memory, 0, index)

        return out, last_h

    def _delete_edge(self, adj, ratio=0.1):
        bsz = adj.size(0)
        adj_del = adj.clone().contiguous().view(bsz, -1)
        for i in range(bsz):
            edges = torch.where(adj_del[i] == 1)[0]
            del_num = math.ceil(len(edges) * ratio)
            del_edges = random.sample(edges.cpu().numpy().tolist(), del_num)
            adj_del[i, del_edges] = 0
        adj_del = adj_del.reshape_as(adj)
        return adj_del

    def _add_edge(self, adj, ratio=0.1):
        bsz = adj.size(0)
        adj_add = adj.clone().contiguous().view(bsz, -1)
        for i in range(bsz):
            non_edges = torch.where(adj_add[i] == 1)[0]
            add_num = math.ceil(len(non_edges) * ratio)
            add_edges = random.sample(non_edges.cpu().numpy().tolist(), add_num)
            adj_add[i, add_edges] = 0
        adj_add = adj_add.reshape_as(adj)
        return adj_add

    def _delete_node(self, reps, adj, ratio=0.1):
        bsz, n, _ = adj.size()
        reps_del = reps.clone()
        adj_del = adj.clone()
        for i in range(bsz):
            del_num = math.ceil(n * ratio)
            del_nodes = random.sample(list(range(n)), del_num)
            adj_del[i, del_nodes, :] = 0
            adj_del[i, :, del_nodes] = 0
            reps_del[i, del_nodes] = 0
        return reps_del, adj_del

    def forward(self, text_tensor=None, video_tensor=None, audio_tensor=None, lengths=None,
                bert_sent_type=None, bert_sent_mask=None, adj_matrix=None):
        bsz, max_len, _ = video_tensor.size()

        # get padding mask
        mask = torch.zeros((bsz, max_len))
        for i in range(bsz):
            index = torch.arange(int(lengths[i].item()), max_len)
            mask[i] = mask[i].index_fill_(0, index, -1e9)
        mask = mask.to(text_tensor.device)

        # get unimodal adj
        adj_matrix_t = adj_matrix[:, :max_len, :max_len]
        adj_matrix_v = adj_matrix[:, max_len:2*max_len, max_len:2*max_len]
        adj_matrix_a = adj_matrix[:, 2*max_len:3*max_len, 2*max_len:3*max_len]

        # encode
        bert_output = self.encoder_t(input_ids=text_tensor,
                                     # token_type_ids=bert_sent_type,
                                     attention_mask=bert_sent_mask)
        hs_t = bert_output[0][:, 1:-1]
        hs_t = F.relu(self.project_t(hs_t))

        hs_v, last_v = self._lstm_encode(video_tensor, lengths, self.encoder_v, video_tensor.size(-1))
        hs_v = F.relu(self.project_v(hs_v))

        hs_a, last_a = self._lstm_encode(audio_tensor, lengths, self.encoder_a, audio_tensor.size(-1))
        hs_a = F.relu(self.project_a(hs_a))

        # multimodal graph
        hs = torch.cat([hs_t, hs_v, hs_a], dim=1)
        hs_gnn, attn = self.gat_m(hs, adj_matrix)
        hs_gnn = F.relu(hs_gnn + hs)
        reps_m = self.readout_m(hs_gnn, mask)

        # unimodal graphs
        hs_t_gnn, _ = self.gat_t(hs_t, adj_matrix_t)
        hs_v_gnn, _ = self.gat_v(hs_v, adj_matrix_v)
        hs_a_gnn, _ = self.gat_a(hs_a, adj_matrix_a)



        hs_t_gnn = F.relu(hs_t_gnn + hs_t)
        hs_v_gnn = F.relu(hs_v_gnn + hs_v)
        hs_a_gnn = F.relu(hs_a_gnn + hs_a)
        reps_t, _ = self.readout_t(hs_t_gnn, mask)
        reps_v, _ = self.readout_v(hs_v_gnn, mask)
        reps_a, _ = self.readout_a(hs_a_gnn, mask)

        qt_input = self.ret(reps_t)
        qv_input = self.rev(reps_v)
        qa_input = self.rea(reps_a)
        qinput = torch.cat([qt_input, qv_input, qa_input], dim=1)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        qfnn = self.qfnn
        qfnn.to(device)
        qfoutputs = qfnn(qinput)
        #print(outputs.shape)
        tqoutputs = qfoutputs[:, 0:2]
        vqoutputs = qfoutputs[:, 0:2]
        aqoutputs = qfoutputs[:, 0:2]



        reps_t = torch.cat([reps_t, tqoutputs], dim=1)
        reps_v = torch.cat([reps_v, vqoutputs], dim=1)
        reps_a = torch.cat([reps_a, aqoutputs], dim=1)

        reps_t = self.ret2(reps_t)
        reps_v = self.rev2(reps_v)
        reps_a = self.rea2(reps_a)

        repm = self.dropout_m(reps_m)
        qminput = self.rem(repm)
        qmfnn = self.qmfnn
        qmfnn.to(device)
        qmfoutputs = qmfnn(qminput)
        repm = torch.cat([repm, qmfoutputs], dim=1)

        repm = self.rem2(repm)

        output = self.fc_out(repm)

        # augmentation
        adj_matrix_aug1 = self._delete_edge(adj_matrix, self.config["aug_ratio"])
        adj_matrix_aug1 = self._add_edge(adj_matrix_aug1, self.config["aug_ratio"])
        hs_gnn_aug1, _ = self.gat_m(hs, adj_matrix_aug1)
        hs_gnn_aug1 = F.relu(hs_gnn_aug1 + hs)
        reps_m_aug1 = self.readout_m(hs_gnn_aug1, mask)

        adj_matrix_t_aug1 = self._delete_edge(adj_matrix_t, self.config["aug_ratio"])
        adj_matrix_v_aug1 = self._delete_edge(adj_matrix_v, self.config["aug_ratio"])
        adj_matrix_a_aug1 = self._delete_edge(adj_matrix_a, self.config["aug_ratio"])
        adj_matrix_t_aug1 = self._add_edge(adj_matrix_t_aug1, self.config["aug_ratio"])
        adj_matrix_v_aug1 = self._add_edge(adj_matrix_v_aug1, self.config["aug_ratio"])
        adj_matrix_a_aug1 = self._add_edge(adj_matrix_a_aug1, self.config["aug_ratio"])
        hs_t_gnn_aug1, _ = self.gat_t(hs_t, adj_matrix_t_aug1)
        hs_v_gnn_aug1, _ = self.gat_v(hs_v, adj_matrix_v_aug1)
        hs_a_gnn_aug1, _ = self.gat_a(hs_a, adj_matrix_a_aug1)
        hs_t_gnn_aug1 = F.relu(hs_t_gnn_aug1 + hs_t)
        hs_v_gnn_aug1 = F.relu(hs_v_gnn_aug1 + hs_v)
        hs_a_gnn_aug1 = F.relu(hs_a_gnn_aug1 + hs_a)
        reps_t_aug1, _ = self.readout_t(hs_t_gnn_aug1, mask)
        reps_v_aug1, _ = self.readout_v(hs_v_gnn_aug1, mask)
        reps_a_aug1, _ = self.readout_a(hs_a_gnn_aug1, mask)

        reps_t_aug = torch.stack([reps_t_aug1, reps_t], dim=1)
        reps_v_aug = torch.stack([reps_v_aug1, reps_v], dim=1)
        reps_a_aug = torch.stack([reps_a_aug1, reps_a], dim=1)
        reps_m_aug = torch.stack([reps_m_aug1, reps_m], dim=1)

        return output.view(-1), F.normalize(reps_m.unsqueeze(1), dim=-1), F.normalize(reps_t.unsqueeze(1), dim=-1), \
               F.normalize(reps_v.unsqueeze(1), dim=-1), F.normalize(reps_a.unsqueeze(1), dim=-1),\
               F.normalize(reps_m_aug, dim=-1), F.normalize(reps_t_aug, dim=-1), F.normalize(reps_v_aug, dim=-1),\
               F.normalize(reps_a_aug, dim=-1)