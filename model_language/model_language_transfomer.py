import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Transformer(nn.Module):
    def __init__(self, vocab_size, em_dim, num_layers, is_training=True):
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, em_dim, padding_idx=0)
        self.position_em = PositionalEncoding(em_dim)
        self.num_layers = num_layers
        self.is_training = is_training
        self.mha = nn.ModuleList([MultiHeadAttention(em_dim, heads=8) for _ in range(num_layers)])
        self.dropout_em = nn.Dropout(p=0.1)
        self.dropout_attn = nn.ModuleList([nn.Dropout(p=0.1) for _ in range(num_layers)])
        self.dropout_ffn = nn.ModuleList([nn.Dropout(p=0.1) for _ in range(num_layers)])
        self.ffn = nn.ModuleList([FeedForward(em_dim, dff=512) for _ in range(num_layers)])
        self.final = nn.Linear(em_dim, vocab_size)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.embedding(x)
        x_mask = (x != 0)
        x = self.position_em(x)
        x = self.dropout_em(x)
        for i in range(self.num_layers):
            attn_out = self.mhai
            attn_out = self.dropout_attni
            x = x + attn_out
            x = nn.LayerNorm(x.size(-1))(x)
            ffn_out = self.ffni
            ffn_out = self.dropout_ffni
            x = x + ffn_out
            x = nn.LayerNorm(x.size(-1))(x)
        enc_out = self.final(x)
        return enc_out


class MultiHeadAttention(nn.Module):
    def __init__(self, latent_dim, heads, mask_right=False):
        super(MultiHeadAttention, self).__init__()
        self.latent_dim = latent_dim
        self.heads = heads
        self.mask_right = mask_right
        assert self.latent_dim % heads == 0, "The latent_dim must be divisible by heads!"
        self.depth = self.latent_dim // heads
        self.q_dense = nn.Linear(self.latent_dim, self.latent_dim, bias=False)
        self.k_dense = nn.Linear(self.latent_dim, self.latent_dim, bias=False)
        self.v_dense = nn.Linear(self.latent_dim, self.latent_dim, bias=False)

    def forward(self, inp, v_mask=None, q_mask=None):
        q, k, v = inp[:3]
        if len(inp) > 3:
            v_mask = inp[3]
            if len(inp) > 4:
                q_mask = inp[4]
        wq = self.q_dense(q)
        wk = self.k_dense(k)
        wv = self.v_dense(v)
        # (batch_size, seq_len, latent_dim) =>(batch_size, seq_len, heads, depth)
        wq = wq.view(wq.size(0), wq.size(1), self.heads, self.depth)
        wk = wk.view(wk.size(0), wk.size(1), self.heads, self.depth)
        wv = wv.view(wv.size(0), wv.size(1), self.heads, self.depth)
        # (batch_size, seq_len, heads, depth) => (batch_size, heads, seq_len, depth)
        wq = wq.transpose(1, 2)
        wk = wk.transpose(1, 2)
        wv = wv.transpose(1, 2)
        # => (batch_size, heads, seq_len_q, seq_len_k)
        scores = torch.matmul(wq, wk.transpose(-2, -1))
        # 缩放因子
        dk = self.depth.float()
        # scores[:, i, j] means the simility of the q[j] with k[j]
        scores = scores / torch.sqrt(dk)

        if v_mask is not None:
            # v_mask:(batch_size, seq_len_k)
            v_mask = v_mask.float()
            # (batch_size, seq_len_k) => (batch_size, 1, 1, seq_len_k)
            for _ in range(scores.dim() - v_mask.dim()):
                v_mask = v_mask.unsqueeze(1)
            scores -= (1 - v_mask) * 1e9
        # 解码端，自注意力时使用。预测第三个词仅使用前两个词
        if (self.mask_right is not False) or (self.mask_right is not None):
            if self.mask_right:
                # [1,1,seq_len_q,seq_len_k]
                ones = torch.ones_like(scores[:1, :1])
                # 不包含对角线的上三角矩阵，每个元素是1e9
                mask_ahead = (ones - torch.triu(ones, diagonal=1)) * 1e9
                # 遮掉所有未预测的词
                scores = scores - mask_ahead
            else:
                # 这种情况下，mask_right是外部传入的0/1矩阵，shape=[q_len, k_len]
                mask_ahead = (1 - torch.tensor(self.mask_right)) * 1e9
                mask_ahead = mask_ahead.unsqueeze(0)
                self.mask_ahead = mask_ahead
                scores = scores - mask_ahead

        attn_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, wv)
        out = out.transpose(1, 2).contiguous().view(out.size(0), out.size(1), self.latent_dim)

        if q_mask:
            # q_mask:(batch_size, seq_len_q)
            q_mask = q_mask.float()
            # (batch_size, seq_len_q) => (batch_size, seq_len_q, 1)
            for _ in range(out.dim() - q_mask.dim()):
                q_mask = q_mask.unsqueeze(-1)
            out *= q_mask

        return out


class PositionalEncoding(nn.Module):
    def __init__(self, latent_dim, maximum_position=1000):
        super(PositionalEncoding, self).__init__()
        self.maximum_position = maximum_position
        self.latent_dim = latent_dim

        # Create a position array with shape (maximum_position, 1)
        position = np.arange(self.maximum_position).reshape((self.maximum_position, 1))
        # Create a range array for latent_dim with shape (1, latent_dim)
        d_model = np.arange(self.latent_dim).reshape((1, self.latent_dim))
        # Calculate the angle rates for the positional encoding
        angle_rates = 1 / np.power(10000, (2 * (d_model // 2)) / np.float32(self.latent_dim))
        # Calculate the angle radians for each position and dimension
        angle_rads = position * angle_rates
        # 将 sin 应用于数组中的偶数索引（indices）；2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        # 将 cos 应用于数组中的奇数索引；2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        # (1, maximum_position, latent_dim)
        self.pos_encoding = torch.tensor(angle_rads[np.newaxis, ...], dtype=torch.float32)

    def forward(self, x):
        # Get the sequence length from the input
        self.seq_len = x.size(1)
        # Slice the positional encoding to match the input sequence length
        self.position_encoding = self.pos_encoding[:, :self.seq_len, :]
        # Add the positional encoding to the input
        return x + self.position_encoding


class FeedForward(nn.Module):
    def __init__(self, d_model, dff=512):
        super(FeedForward, self).__init__()
        self.d_model = d_model
        self.dense_1 = nn.Linear(d_model, dff)
        self.dense_2 = nn.Linear(dff, d_model)

    def forward(self, inp):
        out = self.dense_1(inp)
        out = torch.relu(out)
        out = self.dense_2(out)
        return out
