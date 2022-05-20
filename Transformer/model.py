from operator import attrgetter
from turtle import position
import torch
import torch.nn as nn
import numpy as np
import sentencepiece as spm
import matplotlib.pyplot as plt

## position embed
def get_sinusoid_encoding_table(n_seq, d_hidden):
    
    def cal_angle(position, i_hidden):
        return position / np.power(10000, 2*(i_hidden // 2) / d_hidden)

    def get_posi_ang_vec(position):
        return [cal_angle(position, i_hidden) for i_hidden in range(d_hidden)]

    sinusoid_table = np.array([get_posi_ang_vec(i_seq) for i_seq in range(n_seq)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:,0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:,1::2])

    return sinusoid_table

## pad_mask
def get_attn_pad_mask(seq_q, seq_k, i_pad):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(i_pad).unsqueeze(1).expand(batch_size, len_q, len_k)
    return pad_attn_mask

## decoder pad mask
def get_attn_decoder_mask(seq):
    subsequent_mask = torch.ones_like(seq).unsqueeze(-1).expand(seq.size(0), seq.size(1), seq.size(1))
    subsequent_mask = subsequent_mask.triu(diagonal = 1)
    return subsequent_mask

## scale dot product attention
class ScalarDotProductAttention(nn.Module):
    def __init__(self, config) -> None:
        super(ScalarDotProductAttention, self).__init__()
        self.config = config
        self.dropout = nn.Dropout(config.dropout)
        self.scaler = 1 / (self.config.d_head ** 0.5)

    def forward(self, Q, K, V, attn_mask):

        score = torch.matmul(Q, K.transpose(-1,-2)).mul_(self.scaler)
        score.masked_fill_(attn_mask, -1e9)
        attn_prob = nn.Softmax(dim = -1)(score)
        attn_prob = self.dropout(attn_prob)
        context = torch.matmul(attn_prob, V)

        return context, attn_prob

## multi head attention
class MultiHeadAttention(nn.Module):
    def __init__(self, config) -> None:
        super(MultiHeadAttention, self).__init__()
        self.config = config

        self.W_Q = nn.Linear(self.config.d_hidden, self.config.n_head * self.config.d_head)
        self.W_K = nn.Linear(self.config.d_hidden, self.config.n_head * self.config.d_head)
        self.W_V = nn.Linear(self.config.d_hidden, self.config.n_head * self.config.d_head)
        self.scaled_dot_attn = ScalarDotProductAttention(self.config)
        self.linear = nn.Linear(self.config.n_head * self.config.d_head, self.config.d_hidden)
        #self.dropout = nn.Dropout(config.dropout)

    def forward(self, Q, K, V, attn_mask):
        b_size = Q.size(0)
        q_s = self.W_Q(Q).view(b_size, -1, self.config.n_head, self.config.d_head).transpose(1, 2)
        k_s = self.W_K(K).view(b_size, -1, self.config.n_head, self.config.d_head).transpose(1, 2)
        v_s = self.W_V(V).view(b_size, -1, self.config.n_head, self.config.d_head).transpose(1, 2)

        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.config.n_head, 1, 1)

        context, attn_prob = self.scaled_dot_attn(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(b_size, -1, self.config.n_head * self.config.d_head)
        
        output = self.linear(context)
        return output, attn_prob

## PointwiseFeedForward
class PointWiseFeedForward(nn.Module):
    def __init__(self, config) -> None:
        super(PointWiseFeedForward, self).__init__()
        self.config = config

        self.conv1 = nn.Conv1d(self.config.d_hidden, self.config.d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(self.config.d_ff, self.config.d_hidden, kernel_size= 1)
        self.active = nn.GELU()
        self.dropout = nn.Dropout(self.config.dropout)

    def forward(self, inputs):
        output = self.active(self.conv1(inputs.transpose(1,2)))
        output = self.conv2(output).transpose(1,2)
        output = self.dropout(output)
        return output

# Encoder layer
class Encoderlayer(nn.Module):
    def __init__(self, config) -> None:
        super(Encoderlayer, self).__init__()
        self.config = config

        self.self_attn = MultiHeadAttention(self.config)
        
        self.layer_norm1 = nn.LayerNorm(self.config.d_hidden, eps = self.config.layer_norm_epsilon)
        self.pos_ffn = PointWiseFeedForward(self.config)
        self.layer_norm2 = nn.LayerNorm(self.config.d_hidden, eps = self.config.layer_norm_epsilon)

    def forward(self, inputs, attn_mask):
        attn_outputs, attn_prob = self.self_attn(inputs, inputs, inputs, attn_mask)
        attn_outputs = self.layer_norm1(inputs + attn_outputs)
        ffn_outputs = self.pos_ffn(attn_outputs)
        ffn_outputs = self.layer_norm2(ffn_outputs + attn_outputs)
        return ffn_outputs, attn_outputs

# Decoder layer
class Decoderlayer(nn.Module):
    def __init__(self, config) -> None:
        super(Decoderlayer, self).__init__()
        self.config = config

        self.self_attn = MultiHeadAttention(self.config)
        self.layer_norm1 = nn.LayerNorm(self.config.d_hidden, self.config.layer_norm_epsilon)
        self.dec_enc_attn = MultiHeadAttention(self.config)
        self.layer_norm2 = nn.LayerNorm(self.config.d_hidden, eps = self.config.layer_norm_epsilon)
        self.pos_ffn = PointWiseFeedForward(self.config)
        self.layer_norm3 = nn.LayerNorm(self.config.d_hidden, eps = self.config.layer_norm_epsilon)

    def forward(self, dec_inputs, enc_outputs, self_attn_mask, dec_enc_attn_mask):
        self_attn_outputs , self_attn_prob = self.self_attn(dec_inputs, dec_inputs, dec_inputs, self_attn_mask)
        self_attn_outputs = self.layer_norm1(dec_inputs + self_attn_outputs)
        dec_enc_attn_outputs, dec_enc_attn_prob = self.dec_enc_attn(self_attn_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_enc_attn_outputs = self.layer_norm2(self_attn_outputs + dec_enc_attn_outputs)
        ffn_outputs = self.pos_ffn(dec_enc_attn_outputs)
        ffn_outputs = self.layer_norm3(dec_enc_attn_outputs + ffn_outputs)

        return ffn_outputs, self_attn_prob, dec_enc_attn_prob
    
# Encoder
class Encoder(nn.Module):
    def __init__(self, config) -> None:
        super(Encoder, self).__init__()
        self.config = config
        self.enc_emb = nn.Embedding(self.config.n_enc_vocab, self.config.d_hidden)
        sinuoid_table = torch.FloatTensor(get_sinusoid_encoding_table(self.config.n_enc_seq+1, self.config.d_hidden))
        self.pos_emb = nn.Embedding.from_pretrained(sinuoid_table, freeze= True)
        self.layers = nn.ModuleList([Encoderlayer(self.config) for _ in range(self.config.n_layer)])

    def forward(self, inputs):
        positions = torch.arange(inputs.size(1), device=inputs.device, dtype= inputs.dtype).expand(inputs.size(0), inputs.size(1)).contiguous() + 1
        pos_mask = inputs.eq(self.config.i_padding)
        positions.masked_fill_(pos_mask, 0)

        outputs = self.enc_emb(inputs) + self.pos_emb(positions)

        attn_mask = get_attn_pad_mask(inputs, inputs, self.config.i_padding)

        attn_probs= []
        for layer in self.layers:
            outputs, attn_prob = layer(outputs, attn_mask)
            attn_probs.append(attn_prob)
        
        return outputs, attn_probs

# Decoder
class Decoder(nn.Module):
    def __init__(self, config) -> None:
        super(Decoder, self).__init__()
        self.config = config
        self.dec_emb = nn.Embedding(self.config.n_dec_vocab, self.config.d_hidden)
        sinusoid_table = torch.FloatTensor(get_sinusoid_encoding_table(self.config.n_dec_seq+1, self.config.d_hidden))
        self.pos_emb = nn.Embedding.from_pretrained(sinusoid_table, freeze = True)
        self.layers = nn.ModuleList([Decoderlayer(self.config) for _ in range(self.config.n_layer)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        positions = torch.arange(dec_inputs.size(1), device= dec_inputs.device, dtype= dec_inputs.dtype).expand(dec_inputs.size(0), dec_inputs.size(1)).contiguous() + 1
        pos_mask = dec_inputs.eq(self.config.i_padding)
        positions.masked_fill_(pos_mask, 0)

        dec_outputs = self.dec_emb(dec_inputs) + self.pos_emb(positions)

        dec_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs, self.config.i_padding)
        dec_attn_decoder_mask = get_attn_decoder_mask(dec_inputs)
        dec_self_attn_mask = torch.gt((dec_attn_pad_mask + dec_attn_decoder_mask), 0)
        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs, self.config.i_padding)

        self_attn_probs, dec_enc_attn_probs = [], []
        for layer in self.layers:
            dec_outputs, self_attn_prob, dec_enc_attn_prob = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
            self_attn_probs.append(self_attn_prob)
            dec_enc_attn_probs.append(dec_enc_attn_prob)
        
        return dec_outputs, self_attn_probs, dec_enc_attn_prob

class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()
        self.config = config

        self.encoder = Encoder(self.config)
        self.decoder = Decoder(self.config)

    def forward(self, enc_inputs, dec_inputs):
        enc_outputs, enc_self_attn_probs = self.encoder(enc_inputs)

        dec_outputs, dec_self_attn_probs, decc_enc_attn_probs = self.decoder(dec_inputs, enc_inputs, enc_outputs)

        return dec_outputs, enc_self_attn_probs, dec_self_attn_probs, decc_enc_attn_probs

class MovieClassification(nn.Module):
    def __init__(self, config):
        super(MovieClassification, self).__init__()
        self.config = config

        self.transformer = Transformer(self.config)
        self.projection = nn.Linear(self.config.d_hidden, self.config.n_output, bias = False)

    def forward(self, enc_inputs, dec_inputs):
        dec_outputs, enc_self_attn_probs, dec_self_attn_probs, dec_enc_attn_probs = self.transformer(enc_inputs, dec_inputs)
        dec_outputs, _ = torch.max(dec_outputs, dim = 1)
        logits = self.projection(dec_outputs)

        return logits, enc_self_attn_probs, dec_self_attn_probs, dec_enc_attn_probs

    def save(self, epoch, loss, score, path):
        torch.save({
            "epoch": epoch,
            "loss": loss,
            "score": score,
            "state_dict": self.state_dict()
        }, path)

    def load(self, path):
        save = torch.load(path)
        self.load_state_dict(save["state_dict"])
        return save["epoch"], save["loss"], save["score"]
