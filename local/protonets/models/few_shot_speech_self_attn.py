"""
This is the speaker embeding model based on self attention machanism (transformer)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


from protonets.models import register_model
from .utils import euclidean_dist

import numpy as np

from pyannote.audio.embedding.utils import to_condensed, pdist

import logging
# logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
sh = logging.StreamHandler()
sh.setLevel(logging.DEBUG)
sh.setFormatter(formatter)
logger.addHandler(sh) 



# implementation of self attention encoder is based on http://nlp.seas.harvard.edu/2018/04/03/attention.html
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn
    
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class SpeechEncoder(nn.Module):
    """
    We only keep the encoder from a standard Encoder-Decoder architecture.
    """
    def __init__(self, encoder, src_embed):
        super(SpeechEncoder, self).__init__()
        self.encoder = encoder
        self.src_embed = src_embed
        
    def forward(self, src, src_mask):
        "Take in and process masked src and target sequences."
        return self.encode(src, src_mask)
    
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)

def make_model(src_vocab, tgt_vocab, N=6, 
               d_model=512, d_ff=2048, h=8, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = SpeechEncoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        )
    
    # This was important from their code. 
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model


class TemoralAvgPooling(nn.Module):
    def __init__(self):
        super(TemoralAvgPooling, self).__init__()

    def forward(self, x):
        # x[0] unpadded sequences, x[1] lengths
        # here, we are using pack_padded_sequence, .data is needed for PackedSequence
        return x[0].data.sum(dim=1) / x[1].unsqueeze(1)
        # return x[0].data.sum(dim=1)

class UnitNormalize(nn.Module):
    def __init__(self):
        super(UnitNormalize, self).__init__()

    def forward(self, x):
        # batch_size, emb_dimension
        norm = torch.norm(x, 2, 1, keepdim=True)
        return x / norm

class FewShotSpeech(nn.Module):
    def __init__(self, encoder_rnn, encoder_linear):
        super(FewShotSpeech, self).__init__()

        self.encoder_rnn = encoder_rnn
        self.encoder_linear = encoder_linear

    # training with various length segmets 
    def loss(self, batch):
        xq = batch['xq_padded'] # n_class * n_query * max_len * mfcc_dim
        xs = batch['xs_padded'] # n_class * n_support * max_len * mfcc_dim
        xq_len = batch['xq_len'] # n_class * n_query 
        xs_len = batch['xs_len'] # n_class * n_support
        
        assert xq.shape[0] == xq_len.shape[0]
        assert xs.shape[0] == xs_len.shape[0]
        
        n_class = xq_len.shape[0]
        n_query = xq_len.shape[1]
        n_support = xs_len.shape[1]

        target_inds = torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, n_query, 1).long()
        target_inds = Variable(target_inds, requires_grad=False)

        if xq.is_cuda:
            target_inds = target_inds.cuda()
            
        seq_len = torch.cat([xq_len.view(n_class * n_query, -1).squeeze(-1),
                            xs_len.view(n_class * n_support, -1).squeeze(-1)], 0)
        seq_len = Variable(seq_len, requires_grad=False)

        x = torch.cat([xs.view(n_class * n_support, *xs.size()[2:]), 
                        xq.view(n_class * n_query, *xq.size()[2:])],
                        0)
        
        _len, perm_idx = seq_len.sort(0, descending=True)

        x = x[perm_idx]

        packed_input = pack_padded_sequence(x, _len.cpu().numpy().astype(dtype=np.int32), batch_first=True)

        packed_output, _ = self.encoder_rnn.forward(packed_input)

        z, _ = pad_packed_sequence(packed_output, batch_first=True)

        _, unperm_idx = perm_idx.sort(0)
        z = z[unperm_idx]

        #z, _ = self.encoder_rnn.forward(x)

        z = self.encoder_linear.forward((z, seq_len))

        z_dim = z.size(-1)
        z_proto = z[:n_class*n_support].view(n_class, n_support, z_dim).mean(1)
        zq = z[n_class*n_support:]


        dists = euclidean_dist(zq, z_proto)

        log_p_y = F.log_softmax(-dists, dim=1).view(n_class, n_query, -1)

        loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()

        _, y_hat = log_p_y.max(2)
        acc_val = torch.eq(y_hat, target_inds.squeeze(-1)).float().mean()

        logger.info(f'loss: {loss_val.item()}, acc: {acc_val.item()}')

        return loss_val, {
            'loss': loss_val.item(),
            'acc': acc_val.item()
        }

    def evaluate(self, batch):
        return self.loss(batch)
        

@register_model('few_shot_speech_var_len')
def load_few_short_speech(**kwargs):
    in_dim = kwargs['in_dim'][-1]
    out_dim = kwargs['out_dim']
    n_rnn = kwargs['n_rnn']
    # subsampling

    # self attention
    encoder_rnn = nn.LSTM(in_dim, out_dim, n_rnn,
                    bidirectional=True,
                    batch_first=True)

    # statistic pooling
    encoder_linear = nn.Sequential(
            TemoralAvgPooling(),
            nn.Linear(out_dim * 2, out_dim, bias=True),
            nn.Tanh(),
            nn.Linear(out_dim, out_dim, bias=True),
            nn.Tanh(),
            nn.Sigmoid()
            # UnitNormalize()
        )
    
    gpu_num = kwargs['gpu_num']
    if gpu_num > 1:
        logger.info(f'Using multiple GPUs: {gpu_num}')
        device_ids = list(range(gpu_num))

        encoder_rnn = nn.DataParallel(
            encoder_rnn,
            device_ids=device_ids
        )
        encoder_linear = nn.DataParallel(
            encoder_linear,
            device_ids=device_ids
        )
        return FewShotSpeech(encoder_rnn, encoder_linear)

    else:
        logger.info('Using single GPU')
        return FewShotSpeech(encoder_rnn, encoder_linear)

