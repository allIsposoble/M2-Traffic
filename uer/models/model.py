import torch
import torch.nn as nn
from uer.layers.td_encoder import TdEncoder


class Model(nn.Module):
    """
    Pretraining models consist of three parts:
        - embedding
        - encoder
        - target
    """
    def __init__(self, args, embedding, encoder, target):
        super(Model, self).__init__()
        self.embedding = embedding
        self.encoder = encoder
        self.target = target

        if args.target in ['bert', 'bertflow', 'mlm'] and args.tie_weights:
            self.target.mlm_linear_2.weight = self.embedding.word_embedding.weight
        elif args.target in ['lm', 't5'] and args.tie_weights:
            self.target.output_layer.weight = self.embedding.word_embedding.weight

        if args.target == 't5' and args.share_embedding:
            self.target.embedding.word_embedding.weight = self.embedding.word_embedding.weight

        self.is_moe = args.is_moe
        self.use_td_encoder = getattr(args, "use_td_encoder", False)
        if self.use_td_encoder:
            td_kernel_sizes = tuple(int(k) for k in str(getattr(args, "td_kernel_sizes", "3,5,7")).split(","))
            td_dropout = getattr(args, "td_dropout", getattr(args, "dropout", 0.1))
            self.td_encoder = TdEncoder(args.emb_size, td_kernel_sizes, td_dropout)
            self.td_alpha = nn.Parameter(torch.tensor(float(getattr(args, "td_alpha", 1.0))))
            self.td_layer_norm = nn.LayerNorm(args.emb_size)

    def forward(self, src, tgt, seg, proto=None):
        emb = self.embedding(src, seg)
        if self.use_td_encoder:
            td_hidden = self.td_encoder(emb, seg)
            emb = self.td_layer_norm(emb + self.td_alpha * td_hidden)

        if self.is_moe:
            output, gate_loss = self.encoder(emb, seg, src, proto)
            loss_info = self.target(output, tgt) + (gate_loss,)
        else:
            output = self.encoder(emb, seg)
            loss_info = self.target(output, tgt)
        return loss_info
