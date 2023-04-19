from tabnanny import verbose
import torch
from torch import nn
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
from torch.nn.init import xavier_normal_
import math
import torch.nn.functional as F
from enum import IntEnum
import numpy as np
import pickle
# 改动  第123行 加上self.reset()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device('cuda', 0)


def load_obj(name):
    with open('data/assist2009_pid/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2


class PearsonCorrelation(nn.Module):
    def forward(self, tensor_1, tensor_2):
        x = tensor_1
        y = tensor_2

        vx = x - torch.mean(x)
        vy = y - torch.mean(y)

        cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2))
                                     * torch.sqrt(torch.sum(vy ** 2)))
        return cost


pearson = PearsonCorrelation()


class AKT(nn.Module):
    def __init__(self, n_question, n_pid, d_model, n_blocks,
                 kq_same, dropout, model_type, final_fc_dim=512, n_heads=8, d_ff=2048,  l2=1e-5, separate_qa=False):
        super().__init__()
        """
        Input:
            d_model: dimension of attention block
            final_fc_dim: dimension of final fully connected net before prediction
            n_heads: number of heads in multi-headed attention
            d_ff : dimension for fully conntected net inside the basic block
        """
        self.n_question = n_question
        self.dropout = dropout
        self.kq_same = kq_same  # 知识点与问题是否相等
        # 主要区别在于数据集 assist 2015的数据集中不包含问题信息，而assist2009与2017的数据集中包含着问题信息 那么知识点与问题就不能化等号
        self.n_pid = n_pid
        self.l2 = l2
        self.temp = 0.05
        self.alpha = 0.1
        self.beta = 0.5
        self.cos = nn.CosineSimilarity(dim=1)
        self.model_type = model_type
        self.separate_qa = separate_qa
        embed_l = d_model
        self.q_embed_diff = nn.Embedding(self.n_question+1, embed_l)
        if self.n_pid > 0:
            self.difficult_param = nn.Embedding(self.n_pid+1, 1)
            # self.q_embed_diff = nn.Embedding(self.n_question+1, embed_l)
            self.qa_embed_diff = nn.Embedding(2 * self.n_question + 1, embed_l)
        # n_question+1 ,d_model
        self.q_embed = nn.Embedding(self.n_question+1, embed_l)
        if self.separate_qa:
            self.qa_embed = nn.Embedding(2*self.n_question+1, embed_l)
        else:
            self.qa_embed = nn.Embedding(2, embed_l)
        # Architecture Object. It contains stack of attention block
        self.model = Architecture(n_question=n_question, n_blocks=n_blocks, n_heads=n_heads, dropout=dropout,
                                  d_model=d_model, d_feature=d_model / n_heads, d_ff=d_ff,  kq_same=self.kq_same, model_type=self.model_type)

        self.out = nn.Sequential(
            nn.Linear(d_model + embed_l,
                      final_fc_dim), nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim, 256), nn.ReLU(
            ), nn.Dropout(self.dropout),
            nn.Linear(256, 1)
        )
        self.out_constra = nn.Sequential(
            nn.Linear(d_model*3,
                      final_fc_dim), nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim, 256), nn.ReLU(
            ), nn.Dropout(self.dropout),
            nn.Linear(256, 1)
        )
        self.out_constra_bml = nn.Sequential(
            nn.Linear(d_model,
                      final_fc_dim), nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim, d_model), nn.ReLU(
            ), nn.Dropout(self.dropout),
        )
        self.out_IRT = nn.Sequential(
            nn.Linear(d_model*2,
                      final_fc_dim), nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim, 256), nn.ReLU(
            ), nn.Dropout(self.dropout),
            nn.Linear(256, 256)
        )

        self.reset()

    def reset(self):
        for p in self.parameters():
            if p.size(0) == self.n_pid+1 and self.n_pid > 0:
                # p = torch.abs(p)
                torch.nn.init.constant_(p, 0)

    def forward(self, q_data, qa_data, target, pid_data=None):
        # Batch First
        q_embed_data = self.q_embed(q_data)  # BS, seqlen,  d_model# c_ct
        q_embed_diff_data = self.q_embed_diff(q_data)  # d_ct

        if self.separate_qa:
            # BS, seqlen, d_model #f_(ct,rt)
            qa_embed_data = self.qa_embed(qa_data)
        else:
            qa_data_flag = (qa_data-q_data)//self.n_question  # rt
            # BS, seqlen, d_model # c_ct+ g_rt =e_(ct,rt)
            qa_embed_data = self.qa_embed(qa_data_flag)+q_embed_data

        if self.n_pid > 0:
            pid_embed_data = self.difficult_param(pid_data)  # uq
            pid_all = torch.arange(self.n_pid+1).to(device)
            pid_all_parm = self.difficult_param(pid_all)
            q_embed_data_diff = q_embed_data + pid_embed_data * \
                q_embed_diff_data  # uq *d_ct + c_ct

            qa_data_flag = (qa_data-q_data)//self.n_question  # rt
            qa_embed_data_diff = self.qa_embed(qa_data_flag) + pid_embed_data * \
                q_embed_diff_data  # uq* f_(ct,rt) + e_(ct,rt)

            # c_reg_loss = (pid_embed_data ** 2.).sum() * self.l2
            # c_reg_loss = pid_embed_data.sum()*self.l2
            c_reg_loss = ((pid_embed_data)
                          ** 2).sum() * self.l2
        else:
            q_embed_diff_data = self.q_embed_diff(q_data)
            q_embed_data_diff = q_embed_data + q_embed_diff_data  # d_ct + c_ct
            qa_data = (qa_data-q_data)//self.n_question  # rt
            # BS, seqlen, d_model # c_ct+ g_rt =e_(ct,rt)
            qa_embed_data_diff = self.qa_embed(qa_data)+q_embed_data
            c_reg_loss = 0.

        # BS,seqlen,d_model
        # Pass to the decoder
        # output shape BS,seqlen,d_model or d_model//2
        # [64,200,256]---->[batch_size,seqlen,d_model]

        if self.n_pid > 0:
            pid_embed_data = self.difficult_param(pid_data)  # uq
            pid_all = torch.arange(self.n_pid+1).to(device)
            pid_all_parm = self.difficult_param(pid_all)
            q_embed_data_diff = q_embed_data + pid_embed_data * \
                q_embed_diff_data  # uq *d_ct + c_ct

            qa_data_flag = (qa_data-q_data)//self.n_question  # rt
            qa_embed_data_diff = self.qa_embed(qa_data_flag) + pid_embed_data * \
                q_embed_diff_data  # uq* f_(ct,rt) + e_(ct,rt)

            c_reg_loss = ((pid_embed_data)
                          ** 2).sum() * self.l2

            x_1 = q_embed_data_diff
            x_2 = q_embed_diff_data

            y_1 = qa_embed_data
            y_2 = q_embed_data

            z_1 = pid_embed_data * q_embed_diff_data + \
                self.qa_embed(qa_data_flag)  # uq *d_ct + c_ct
            z_2 = pid_embed_data * q_embed_diff_data
            out_x, out_y = self.model(
                x_1, x_2, y_1, y_2, z_1, z_2, n_pid=True)

            concat_h = torch.cat([out_x, out_y], dim=-1)
            d_output = self.out_IRT(concat_h)
            concat_q = torch.cat([d_output, q_embed_data], dim=-1)
        else:
            x_1 = q_embed_data_diff
            x_2 = q_embed_diff_data

            y_1 = qa_embed_data
            y_2 = q_embed_data

            z_1 = x_1
            z_2 = x_2
            # out_x, _ = self.model(
            #     q_embed_data, qa_embed_data, q_embed_data_diff, qa_embed_data_diff, n_pid=False)
            d_output, _ = self.model(
                x_1, x_2, y_1, y_2, z_1, z_2, n_pid=True)

            concat_q = torch.cat([d_output, q_embed_data], dim=-1)

        # [64,200,512]---->[batch_size,seqlen,2*d_model]

        # concat_h = torch.cat([out_x, out_y], dim=-1)
        # d_output = self.out_IRT(concat_h)
        # concat_q = torch.cat([d_output, q_embed_data], dim=-1)

        # [64,200,1]---->[batch_size,seqlen,result]
        output = self.out(concat_q)

        # output_q = self.diffloss(self.q_embed_diff)
        # output_qa = self.diffloss(self.qa_embed_diff)
        # a = np.load("difficult")
        # a = a.tolist()
        # diff = torch.Tensor(a).cuda()
        # loss_q = (output_q-diff)**2*self.l2+(output_qa-diff)**2*self.l2

        labels = target.reshape(-1)  # [12800]---->batch_size*seqlen  64*200
        m = nn.Sigmoid()
        preds = (output.reshape(-1))  # logit
        mask = labels > -0.9
        masked_labels = labels[mask].float()
        masked_preds = preds[mask]
        loss = nn.BCEWithLogitsLoss(reduction='none')
        output = loss(masked_preds, masked_labels)
        # return output.sum()+c_reg_loss, m(preds), mask.sum()

        Q_data = torch.arange(self.n_question).to(device)
        out_Q = self.q_embed(Q_data)
        Q = self.out_constra_bml(out_Q)
        flag1 = torch.tensor(1).to(device)
        flag0 = torch.tensor(0).to(device)

        q_pos = Q+self.qa_embed(flag1)
        q_neg = Q+self.qa_embed(flag0)
        contrastive_loss = ContrastiveLossELI5(
            self.n_question, 0.05, verbose=False)(q_pos, q_neg)

        bml_loss = BMLLoss(-0.1, 0.5, verbose=False)(q_pos, q_neg, Q)
        # sum_out = output.mean()
        # sum_creg_loss = c_reg_loss
        # sum_bml = bml_loss

        # 修改1：增添对比损失函数 loss_contra
        seqlength = d_output.size(1)
        bs = d_output.size(0)
        d_model_size = d_output.size(2)
        pad_zero = torch.zeros(bs, 1, d_model_size).to(device)
        d_output1 = torch.cat([pad_zero, d_output[:, :seqlength-1, :]], dim=1)
        d_output2 = torch.abs(d_output-d_output1)
        concat_contra = torch.cat([d_output, d_output1, d_output2], dim=-1)
        output_contra = self.out_constra(concat_contra)
        preds_contra = (output_contra.reshape(-1))
        masked_preds_contra = preds_contra[mask]
        masked_labels_contra = torch.ones_like(masked_labels).float()
        output_contra = loss(masked_preds_contra, masked_labels_contra)

        output_mean = output.mean()
        output_contra_mean = output_contra.mean()
        # return output.mean()+output_contra.mean()+c_reg_loss, m(preds), mask.sum()
        d_output = d_output[:, :seqlength-1, :]
        d_output1 = d_output1[:, 1:seqlength, :]
        loss_pearson = pearson(d_output1, d_output)

        return output.mean()+contrastive_loss+output_contra.mean()+c_reg_loss-loss_pearson, m(preds), mask.sum()

        # return output.mean()+output_contra.mean()+c_reg_loss+contrastive_loss+bml_loss-loss_pearson, m(preds), mask.sum()
        # return output.sum()+output_contra.sum()+c_reg_loss+loss_q, m(preds), mask.sum()
        # return output.mean()+c_reg_loss+contrastive_loss+bml_loss, m(preds), mask.sum()


class ContrastiveLossELI5(nn.Module):
    def __init__(self, batch, temperature=0.5, verbose=True):
        super().__init__()
        self.batch_size = batch
        self.register_buffer("temperature", torch.tensor(temperature))
        self.verbose = verbose

    def forward(self, emb_i, emb_j):
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        z_i, z_j as per SimCLR paper
        """
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)

        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(
            representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        if self.verbose:
            print("Similarity matrix\n", similarity_matrix, "\n")

        def l_ij(i, j):
            z_i_, z_j_ = representations[i], representations[j]
            sim_i_j = similarity_matrix[i, j]
            if self.verbose:
                print(f"sim({i}, {j})={sim_i_j}")

            numerator = torch.exp(sim_i_j / self.temperature)
            one_for_not_i = torch.ones(
                (2 * self.batch_size, )).scatter_(0, torch.tensor([i]), 0.0).to(emb_i.device)
            if self.verbose:
                print(f"1{{k!={i}}}", one_for_not_i)

            denominator = torch.sum(
                one_for_not_i *
                torch.exp(similarity_matrix[i, :] / self.temperature)
            )
            if self.verbose:
                print("Denominator", denominator)

            loss_ij = -torch.log(numerator / denominator)
            if self.verbose:
                print(f"loss({i},{j})={loss_ij}\n")

            return loss_ij.squeeze(0)

        N = self.batch_size
        loss = 0.0
        for k in range(0, N):
            loss += l_ij(k, k + N) + l_ij(k + N, k)
        return 1.0 / (2*N) * loss


class BMLLoss(nn.Module):
    def __init__(self, alpha=0.1, beta=0.5, verbose=True):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.verbose = verbose

    def forward(self, emb_i, emb_j, emb_q):
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        z_i, z_j as per SimCLR paper

        emb_i: 正例   emb_j:负例   emb_q:原知识点
        """
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)
        z_q = F.normalize(emb_q, dim=1)
        similarity_pos = F.cosine_similarity(z_i, z_q)
        similarity_neg = F.cosine_similarity(z_j, z_q)
        diag_pos = torch.diag(similarity_pos)
        diag_neg = torch.diag(similarity_neg)
        temp = diag_neg-diag_pos
        loss = torch.relu(temp+self.alpha)+torch.relu(-temp-self.beta)
        loss_bml = torch.mean(loss)

        return loss_bml


class Architecture(nn.Module):
    def __init__(self, n_question,  n_blocks, d_model, d_feature,
                 d_ff, n_heads, dropout, kq_same, model_type):
        super().__init__()
        """
            n_block : number of stacked blocks in the attention
            d_model : dimension of attention input/output
            d_feature : dimension of input in each of the multi-head attention part.
            n_head : number of heads. n_heads*d_feature = d_model
        """
        self.d_model = d_model  # 注意力机制的维度
        self.model_type = model_type

        if model_type in {'akt'}:
            self.blocks_1 = nn.ModuleList([
                TransformerLayer(d_model=d_model, d_feature=d_model // n_heads,
                                 d_ff=d_ff, dropout=dropout, n_heads=n_heads, forget=True, kq_same=kq_same)
                for _ in range(n_blocks)
            ])  # 带有遗忘机制的注意力
            self.blocks_2 = nn.ModuleList([
                TransformerLayer(d_model=d_model, d_feature=d_model // n_heads,
                                 d_ff=d_ff, dropout=dropout, n_heads=n_heads, forget=True, kq_same=kq_same)
                for _ in range(n_blocks)
            ])  # 带遗忘机制的注意力
            self.blocks_3 = nn.ModuleList([
                TransformerLayer(d_model=d_model, d_feature=d_model // n_heads,
                                 d_ff=d_ff, dropout=dropout, n_heads=n_heads, forget=True, kq_same=kq_same)
                for _ in range(n_blocks)
            ])  # 带有遗忘机制的注意力
            self.blocks_4 = nn.ModuleList([
                TransformerLayer(d_model=d_model, d_feature=d_model // n_heads,
                                 d_ff=d_ff, dropout=dropout, n_heads=n_heads, forget=False, kq_same=kq_same)
                for _ in range(n_blocks)
            ])  # 不带遗忘机制的注意力

    def forward(self, x_1, x_2, y_1, y_2, z_1, z_2, n_pid):
        # target shape  bs, seqlen
        seqlen, batch_size = x_1.size(1), x_1.size(0)

        x_1 = x_1
        x_2 = x_2

        y_1 = y_1
        y_2 = y_2

        z_1 = z_1
        z_2 = z_2

        # z_1[:, 1:, :] = z_1[:, 0:199, :]
        # z_1[:, 0, :] = 0

        # y_1[:, 1:, :] = y_1[:, 0:199, :]
        # y_1[:, 0, :] = 0

        # encoder
        # for block in self.blocks_1:  # encode qas
        #     y = block(mask=0, query=y, key=y, values=y)
        # flag_first = True
        # for block in self.blocks_2:
        #     if flag_first:  # peek current question
        #         x = block(mask=1, query=x, key=x,
        #                   values=x, apply_pos=False)
        #         flag_first = False
        #     else:  # dont peek current response
        #         x = block(mask=0, query=x, key=x, values=y, apply_pos=True)
        #         flag_first = True
        # return x
        """
        x多送入一层
        """
        # for block in self.blocks_1:  # encode qas
        #     y = block(mask=0, query=y, key=y, values=y)
        #     x = block(mask=1, query=x, key=x, values=x)
        # flag_first = False
        # for block in self.blocks_2:
        #     if flag_first:  # peek current question
        #         x = block(mask=1, query=x, key=x, values=x, apply_pos=False)
        #         flag_first = False
        #     else:  # dont peek current response
        #         x = block(mask=0, query=x, key=x, values=y, apply_pos=True)
        #         flag_first = False
        # return x

        """
        两参数 IRT
        """
        if n_pid:
            for block in self.blocks_1:  # encode qas
                x = block(mask=0, query=x_1, key=x_1, values=x_2)
                # x = block(mask=1, query=x, key=x, values=x_diff)

            for block in self.blocks_2:  # encode qas
                y = block(mask=0, query=y_1, key=y_1,
                          values=y_2, apply_pos=False)

            for block in self.blocks_3:  # encode qas
                z = block(mask=0, query=z_1, key=z_1, values=z_2)

            flag_first = True
            for block in self.blocks_2:
                x = block(mask=0, query=x, key=x,
                          values=z, apply_pos=True)
                y = block(mask=0, query=y, key=y,
                          values=z, apply_pos=True)
                flag_first = False
            return x, y

        else:
            for block in self.blocks_1:  # encode qas
                x = block(mask=0, query=x_1, key=x_1, values=x_2)
                # x = block(mask=1, query=x, key=x, values=x_diff)

            for block in self.blocks_2:  # encode qas
                y = block(mask=0, query=y_1, key=y_1,
                          values=y_2, apply_pos=False)

            flag_first = False
            for block in self.blocks_4:
                if flag_first:  # peek current question
                    y = block(mask=1, query=y, key=y,
                              values=y, apply_pos=False)
                    flag_first = False
                else:  # dont peek current response
                    x = block(mask=1, query=x, key=x,
                              values=y, apply_pos=True)
                    flag_first = False
            return x, y


class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_feature,
                 d_ff, n_heads, dropout, forget, kq_same):
        super().__init__()
        """
            This is a Basic Block of Transformer paper. It containts one Multi-head attention object. Followed by layer norm and postion wise feedforward net and dropout layer.
        """
        kq_same = kq_same == 1
        # Multi-Head Attention Block
        self.masked_attn_head = MultiHeadAttention(
            d_model, d_feature, n_heads, dropout, forget,  kq_same=kq_same)

        # Two layer norm layer and two droput layer
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, mask, query, key, values, apply_pos=True):
        """
        Input:
            block : object of type BasicBlock(nn.Module). It contains masked_attn_head objects which is of type MultiHeadAttention(nn.Module).
            mask : 0 means, it can peek only past values. 1 means, block can peek only current and pas values
            query : Query. In transformer paper it is the input for both encoder and decoder
            key : Keys. In transformer paper it is the input for both encoder and decoder
            Values. In transformer paper it is the input for encoder and  encoded output for decoder (in masked attention part)

        Output:
            query: Input gets changed over the layer and returned.

        """

        seqlen, batch_size = query.size(1), query.size(0)
        nopeek_mask = np.triu(
            np.ones((1, 1, seqlen, seqlen)), k=mask).astype('uint8')  # 包含对角线元素的上三角矩阵
        src_mask = (torch.from_numpy(nopeek_mask) == 0).to(device)
        if mask == 0:  # If 0, zero-padding is needed.
            # Calls block.masked_attn_head.forward() method
            query2 = self.masked_attn_head(
                query, key, values, mask=src_mask, zero_pad=True)
        else:
            # Calls block.masked_attn_head.forward() method
            query2 = self.masked_attn_head(
                query, key, values, mask=src_mask, zero_pad=False)

        query = query + self.dropout1((query2))
        query = self.layer_norm1(query)
        if apply_pos:
            query2 = self.linear2(self.dropout(
                self.activation(self.linear1(query))))
            query = query + self.dropout2((query2))
            query = self.layer_norm2(query)
        return query


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_feature, n_heads, dropout, forget,  kq_same, bias=True):
        super().__init__()
        """
        It has projection layer for getting keys, queries and values. Followed by attention and a connected layer.
        """
        self.d_model = d_model
        self.d_k = d_feature
        self.h = n_heads
        self.kq_same = kq_same
        self.forget = forget

        self.v_linear = nn.Linear(d_model, d_model, bias=bias)
        self.k_linear = nn.Linear(d_model, d_model, bias=bias)
        if kq_same is False:
            self.q_linear = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.proj_bias = bias
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.gammas = nn.Parameter(torch.zeros(n_heads, 1, 1))
        # 遗忘机制内的参数
        torch.nn.init.xavier_uniform_(self.gammas)

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.k_linear.weight)
        xavier_uniform_(self.v_linear.weight)
        if self.kq_same is False:
            xavier_uniform_(self.q_linear.weight)

        if self.proj_bias:
            constant_(self.k_linear.bias, 0.)
            constant_(self.v_linear.bias, 0.)
            if self.kq_same is False:
                constant_(self.q_linear.bias, 0.)
            constant_(self.out_proj.bias, 0.)

    def forward(self, q, k, v, mask, zero_pad):

        bs = q.size(0)

        # perform linear operation and split into h heads

        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        if self.kq_same is False:
            q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        else:
            q = self.k_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model

        k = k.transpose(1, 2)
        # [64,8,200,32]  --->[batchsize,head,seqlen,d_model]
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        # calculate attention using function we will define next
        gammas = self.gammas
        if self.forget:
            scores = attention(q, k, v, self.d_k,
                               mask, self.dropout, zero_pad, gammas)  # 注意力之后维度为[64,8,200,32]---->[batch_size,head,seqlen,d_k]
        else:
            scores = attention_noforget(q, k, v, self.d_k,
                                        mask, self.dropout, zero_pad, gammas)  # 注意力之后维度为[64,8,200,32]---->[batch_size,head,seqlen,d_k]

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous()\
            .view(bs, -1, self.d_model)  # [64,200,256]--->[batch_size,seqlen,d_model]

        output = self.out_proj(concat)  # [64,200,64]

        return output


def attention(q, k, v, d_k, mask, dropout, zero_pad, gamma=None):
    """
    This is called by Multi-head atention object to find the values.
    """
    scores = torch.matmul(q, k.transpose(-2, -1)) / \
        math.sqrt(d_k)  # BS, 8, seqlen, seqlen
    bs, head, seqlen = scores.size(0), scores.size(1), scores.size(2)

    x1 = torch.arange(seqlen).expand(seqlen, -1).to(device)
    x2 = x1.transpose(0, 1).contiguous()
    # 多维数组的低维存储

    with torch.no_grad():
        scores_ = scores.masked_fill(mask == 0, -1e32)
        scores_ = F.softmax(scores_, dim=-1)  # BS,8,seqlen,seqlen
        scores_ = scores_ * mask.float().to(device)
        distcum_scores = torch.cumsum(scores_, dim=-1)  # bs, 8, sl, sl
        disttotal_scores = torch.sum(
            scores_, dim=-1, keepdim=True)  # bs, 8, sl, 1
        position_effect = torch.abs(
            x1-x2)[None, None, :, :].type(torch.FloatTensor).to(device)  # 1, 1, seqlen, seqlen
        # bs, 8, sl, sl positive distance
        dist_scores = torch.clamp(
            (disttotal_scores-distcum_scores)*position_effect, min=0.)
        dist_scores = dist_scores.sqrt().detach()
    m = nn.Softplus()
    gamma = -1. * m(gamma).unsqueeze(0)  # 1,8,1,1
    # Now after do exp(gamma*distance) and then clamp to 1e-5 to 1e5
    total_effect = torch.clamp(torch.clamp(
        (dist_scores*gamma).exp(), min=1e-5), max=1e5)
    scores = scores * total_effect

    scores.masked_fill_(mask == 0, -1e32)
    scores = F.softmax(scores, dim=-1)  # BS,8,seqlen,seqlen
    if zero_pad:
        pad_zero = torch.zeros(bs, head, 1, seqlen).to(device)
        scores = torch.cat([pad_zero, scores[:, :, 1:, :]], dim=2)
    scores = dropout(scores)
    output = torch.matmul(scores, v)
    return output


class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = 0.1 * torch.randn(max_len, d_model)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=True)

    def forward(self, x):
        return self.weight[:, :x.size(Dim.seq), :]  # ( 1,seq,  Feature)


class CosinePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = 0.1 * torch.randn(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=False)

    def forward(self, x):
        return self.weight[:, :x.size(Dim.seq), :]  # ( 1,seq,  Feature)


def attention_noforget(q, k, v, d_k, mask, dropout, zero_pad, gamma=None):
    """
    This is called by Multi-head atention object to find the values.
    """
    scores = torch.matmul(q, k.transpose(-2, -1)) / \
        math.sqrt(d_k)  # BS, 8, seqlen, seqlen
    bs, head, seqlen = scores.size(0), scores.size(1), scores.size(2)

    # 多维数组的低维存储

    scores.masked_fill_(mask == 0, -1e32)
    scores = F.softmax(scores, dim=-1)  # BS,8,seqlen,seqlen
    if zero_pad:
        pad_zero = torch.zeros(bs, head, 1, seqlen).to(device)
        scores = torch.cat([pad_zero, scores[:, :, 1:, :]], dim=2)
    scores = dropout(scores)
    output = torch.matmul(scores, v)
    return output
