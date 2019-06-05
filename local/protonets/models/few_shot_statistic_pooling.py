import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from protonets.models import register_model
from .utils import euclidean_dist

import numpy as np

from pyannote.audio.embedding.utils import to_condensed, pdist

from ..utils import log

logger = log.setup_custom_logger(__name__)


class FrameLevelEmbedding(nn.Module):
    def __init__(self, kernel_sizes, layer_sizes, input_feature_dim, dropout_keep_prob):
        # temporal pooling is defined by kernal size and layer size
        super(FrameLevelEmbedding, self).__init__()
        self.kernel_sizes = kernel_sizes
        self.layer_sizes = layer_sizes
        self.dropout_keep_prob = dropout_keep_prob

        # temporal pooking layers
        self.conv_layers = nn.Sequential()
        pre_dim = input_feature_dim
        for i, (kernal_size, layer_size) in enumerate(zip(kernel_sizes, layer_sizes)):
            self.conv_layers.add_module(f"conv_{i+1}",nn.Conv1d(pre_dim, layer_size, kernal_size, padding=kernal_size//2))
            self.conv_layers.add_module(f"relu_{i+1}", nn.ReLU())
            if i != len(kernel_sizes) - 1:
                self.conv_layers.add_module(f"dropout_{i+1}", torch.nn.Dropout(self.dropout_keep_prob))
            #TODO: batch nornalization?

            pre_dim = layer_size

    def forward(self, x):
        # x: sequence of audio features, N * T * dim
        # output: sequence of hidden representation, N * T * hid
        return self.conv_layers.forward(x.permute(0, 2, 1)).permute(0, 2, 1)

class StatisticsPooling(nn.Module):
    def __init__(self):
        super(StatisticsPooling, self).__init__()
    
    def forward(self, x, lengths):
        # x: batch * T * hid_dim
        # lengths: batch * 1
        x = x.cpu()
        lengths = lengths.cpu()
        lengths = lengths
        h_mean = x.sum(dim=1) / lengths[:, None].float()

        h_std = (x - h_mean[:, None, :])
        mask = torch.arange(x.size(1))[None, :] < lengths[:, None].float()
        mask = mask.float()
        h_std = h_std * mask[:, :, None].expand(*mask.shape, h_std.shape[-1])
        h_std = h_std ** 2
        h_std = h_std.sum(dim=1) / lengths[:, None].float()
        h_std = torch.sqrt(h_std)
        return torch.cat([h_mean, h_std], 1).cuda()

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

class SpeechEmbeddingModel(nn.Module):
    def __init__(self, seq_encoder, pooling_layer, linear_layers):
        super(SpeechEmbeddingModel, self).__init__()

        # self.seq_encoder = nn.DataParallel(seq_encoder)
        # self.pooling_layer = nn.DataParallel(pooling_layer)
        # self.linear_layers = nn.DataParallel(linear_layers)

        self.seq_encoder = seq_encoder
        self.pooling_layer = pooling_layer
        self.linear_layers = linear_layers

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
        
        x = self.seq_encoder.forward(x)

        z = self.pooling_layer.forward(x, seq_len)
        z = self.linear_layers.forward(z)

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
    # layer_sizes = [512, 512, 512, 512, 3 * 512]
    # kernel_sizes = [5, 5, 7, 1, 1]
    # embedding_sizes = [512, 200], since in x-vector, after LDA, 200 dimension 
    # is used for PLDA. Here we output the same dimension.
    layer_sizes = kwargs['layer_sizes']
    kernel_sizes = kwargs['kernel_sizes']
    embedding_sizes = kwargs['embedding_sizes']

    feature_dim = kwargs['feature_dim']
    dropout_keep_prob = kwargs['dropout_keep_prob']
    seq_encoder = FrameLevelEmbedding(kernel_sizes, layer_sizes, feature_dim, dropout_keep_prob)
    pooling_layer = StatisticsPooling()

    linear_layers = nn.Sequential()
    pre_dim = layer_sizes[-1] * 2
    for i, emb_size in enumerate(embedding_sizes):
        linear_layers.add_module(f"linear_{i+1}", nn.Linear(pre_dim, emb_size, bias=True))
        linear_layers.add_module(f"linear_relu_{i+1}", nn.ReLU())
        pre_dim = emb_size

        if i != len(embedding_sizes) - 1:
            linear_layers.add_module(f"linear_dropout_{i+1}", torch.nn.Dropout(dropout_keep_prob))


    return SpeechEmbeddingModel(seq_encoder, pooling_layer, linear_layers)