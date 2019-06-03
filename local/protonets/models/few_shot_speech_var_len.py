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
            self.conv_layers.add_module(f"conv_{i+1}",nn.Conv1d(pre_dim, layer_size, kernal_size))
            self.conv_layers.add_module(f"relu_{i+1}", nn.ReLU())
            if i != len(kernel_sizes) - 1:
                self.conv_layers.add_module(f"dropout_{i+1}", torch.nn.Dropout(self.dropout_keep_prob))
            #TODO: batch nornalization?

            pre_dim = layer_size

    def forward(self, x):
        # x: sequence of audio features, N * T * dim
        # output: sequence of hidden representation, N * T * hid
        return self.conv_layers.forward(x.transpose(1,2)).transpose(1,2)

class StatisticsPooling(nn.Module):
    def __init__(self):
        super(StatisticsPooling, self).__init__()
    
    def forward(self, x):
        h_mean = x.mean(dim=1)
        h_std = x.std(dim=1)
        return torch.cat([h_mean, h_std], 1)

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

    def pdist(self, fX):
        """Compute pdist à-la scipy.spatial.distance.pdist

        Parameters
        ----------
        fX : (n, d) torch.Tensor
            Embeddings.

        Returns
        -------
        distances : (n * (n-1) / 2,) torch.Tensor
            Condensed pairwise distance matrix
        """

        n_sequences, _ = fX.size()
        distances = []

        for i in range(n_sequences - 1):
            d = 1. - F.cosine_similarity(
                fX[i, :].expand(n_sequences - 1 - i, -1),
                fX[i+1:, :], dim=1, eps=1e-8)

            distances.append(d)

        return torch.cat(distances)

    
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
    # layer_sizes = [512, 512, 512, 512, 3 * 512]
    # kernel_sizes = [5, 5, 7, 1, 1]
    # embedding_sizes = [512, 512]
    layer_sizes = kwargs['layer_sizes']
    kernel_sizes = kwargs['kernel_sizes']
    embedding_sizes = kwargs['embedding_sizes']

    feature_dim = kwargs['feature_dim']
    dropout_keep_prob = kwargs['dropout_keep_prob']

    encoder = nn.Sequential(
            FrameLevelEmbedding(kernel_sizes, layer_sizes, feature_dim, dropout_keep_prob),
            StatisticsPooling(),
            # UnitNormalize()
        )
    pre_dim = layer_sizes[-1] * 2
    for i, emb_size in enumerate(embedding_sizes):
        encoder.add_module(f"linear_{i+1}", nn.Linear(pre_dim, emb_size, bias=True))
        encoder.add_module(f"linear_relu_{i+1}", nn.ReLU())

        if i != len(embedding_sizes) - 1:
            encoder.add_module(f"linear_dropout_{i+1}", torch.nn.Dropout(dropout_keep_prob))

    return encoder