from torch import nn, Tensor
from typing import Dict
import torch


class MLP(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_dim=1,
                 num_layers=1,
                 dropout_p=0.,
                 use_norm=False,
                 emb_size=1
                 ):
        super(MLP, self).__init__()
        assert hidden_dim > 0, "Set up correct hidden dimension"
        assert num_layers > 0, "Need at least 1 layers"
        self.dims = [input_dim] + (num_layers - 1) * [hidden_dim] + [output_dim]
        self.dropout = nn.Dropout(p=dropout_p)

        self._fea_value = 2  # How many value feature can get
        self.embedding = nn.ModuleList([nn.Embedding(self._fea_value, emb_size) for i in range((input_dim // 2))])

        self.log_softmax = nn.LogSoftmax(dim=-1)

        self.layers = nn.ModuleList()
        self.norm = nn.Identity()

        if use_norm:
            self.norm = nn.BatchNorm1d(self.dims[0])

        for i in range(len(self.dims) - 1):
            self.layers.append(nn.Linear(self.dims[i], self.dims[i + 1]))
            self.layers.append(nn.ReLU())

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """input dim: batch_size * input_dim"""
        h_list = []
        h = self.norm(x['h'])

        """
            emb_half, common_half = torch.tensor_split(h, dim=0)
            emb_half = [emb(elem) for emb, elem in zip(self.emb, emb_half.T)]
            h = torch.cat(emb_half + common_half, dim=0)
        """

        for i, layer in enumerate(self.layers):
            h = layer(h)
            h_list.append(h)

        log_preds = self.log_softmax(h)

        return {"logits": h, "log_preds": log_preds}
