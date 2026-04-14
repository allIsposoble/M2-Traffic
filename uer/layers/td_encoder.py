import torch
import torch.nn as nn


class TdEncoder(nn.Module):
    """
    Temporal Dynamics encoder.
    Multi-scale 1D-CNN stack with BN + GELU + residual connection.
    Input/Output shape: [batch_size, seq_length, emb_size]
    """

    def __init__(self, emb_size, kernel_sizes=(3, 5, 7), dropout=0.1):
        super(TdEncoder, self).__init__()
        self.emb_size = emb_size
        self.kernel_sizes = kernel_sizes

        self.convs = nn.ModuleList([
            nn.Conv1d(emb_size, emb_size, kernel_size=k, padding=k // 2)
            for k in kernel_sizes
        ])
        self.bns = nn.ModuleList([nn.BatchNorm1d(emb_size) for _ in kernel_sizes])
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fuse = nn.Linear(emb_size * len(kernel_sizes), emb_size)

    def forward(self, emb, seg=None):
        # emb: [B, L, D] -> [B, D, L]
        x = emb.transpose(1, 2).contiguous()
        feats = []
        for conv, bn in zip(self.convs, self.bns):
            h = conv(x)
            h = bn(h)
            h = self.act(h)
            feats.append(h)

        h = torch.cat(feats, dim=1)
        h = h.transpose(1, 2).contiguous()
        h = self.fuse(h)
        h = self.dropout(h)
        h = self.dropout(h + emb)
        return h
