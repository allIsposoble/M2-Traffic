"""
This script provides an exmaple to wrap UER-py for classification.
"""
import os
import sys
sys.path.append(os.getcwd())
import random
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from uer.layers import *
from uer.encoders import *
from uer.layers.td_encoder import TdEncoder
from uer.utils.constants import *
from uer.utils import *
from uer.utils.optimizers import *
from uer.utils.config import load_hyperparam
from uer.utils.seed import set_seed
from uer.model_saver import save_model
from uer.opts import finetune_opts
import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score


class Classifier(nn.Module):
    def __init__(self, args):
        super(Classifier, self).__init__()
        self.embedding = str2embedding[args.embedding](args, len(args.tokenizer.vocab))
        self.encoder = str2encoder[args.encoder](args)
        self.labels_num = args.labels_num
        self.pooling = args.pooling
        self.soft_targets = args.soft_targets
        self.soft_alpha = args.soft_alpha
        self.use_td_encoder = args.use_td_encoder
        self.use_hybrid_pooling = args.use_hybrid_pooling
        self.use_scm_arcface = args.use_scm_arcface

        if self.use_td_encoder:
            td_kernel_sizes = tuple(int(k) for k in str(args.td_kernel_sizes).split(","))
            self.td_encoder = TdEncoder(args.emb_size, td_kernel_sizes, args.td_dropout)
            self.td_alpha = nn.Parameter(torch.tensor(float(args.td_alpha)))
            self.td_layer_norm = nn.LayerNorm(args.emb_size)

        if self.use_hybrid_pooling:
            self.feature_dim = args.hidden_size * 4
            self.fusion = nn.Linear(self.feature_dim, args.hidden_size)
        else:
            self.feature_dim = args.hidden_size

        self.output_layer_1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.output_layer_2 = nn.Linear(args.hidden_size, self.labels_num)

        if self.use_scm_arcface:
            self.arcface_s = args.arcface_s
            self.arcface_m_base = args.arcface_m_base
            self.arcface_m_lambda = args.arcface_m_lambda
            self.arcface_weight = nn.Parameter(torch.empty(self.labels_num, args.hidden_size))
            nn.init.xavier_uniform_(self.arcface_weight)
            self.pce = nn.Sequential(
                nn.Linear(args.hidden_size, args.pce_hidden_size),
                nn.GELU(),
                nn.Linear(args.pce_hidden_size, 1)
            )

    def _pool_single(self, h):
        if self.pooling == "mean":
            return torch.mean(h, dim=1)
        if self.pooling == "max":
            return torch.max(h, dim=1)[0]
        if self.pooling == "last":
            return h[:, -1, :]
        return h[:, 0, :]

    def _hybrid_pool(self, h):
        return torch.cat([torch.max(h, dim=1)[0], torch.mean(h, dim=1)], dim=-1)

    def _arcface_loss(self, features, labels, td_hidden):
        norm_features = F.normalize(features, p=2, dim=-1)
        norm_weight = F.normalize(self.arcface_weight, p=2, dim=-1)
        cosine = torch.matmul(norm_features, norm_weight.t()).clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        td_global = torch.mean(td_hidden, dim=1)
        sigma = torch.sigmoid(self.pce(td_global)).squeeze(-1)
        dynamic_margin = self.arcface_m_base + self.arcface_m_lambda * sigma

        theta_y = torch.acos(cosine.gather(1, labels.view(-1, 1)).squeeze(1))
        target_cos = torch.cos(theta_y + dynamic_margin)

        logits = cosine.clone()
        logits.scatter_(1, labels.view(-1, 1), target_cos.unsqueeze(1))
        logits = logits * self.arcface_s
        loss = nn.CrossEntropyLoss()(logits, labels.view(-1))
        return loss, logits

    def forward(self, src, tgt, seg, soft_tgt=None):
        emb = self.embedding(src, seg)
        td_hidden = emb
        if self.use_td_encoder:
            td_hidden = self.td_encoder(emb, seg)
            emb = self.td_layer_norm(emb + self.td_alpha * td_hidden)

        rc_hidden = self.encoder(emb, seg)

        if self.use_hybrid_pooling:
            v_td = self._hybrid_pool(td_hidden)
            v_rc = self._hybrid_pool(rc_hidden)
            flow_feature = self.fusion(torch.cat([v_td, v_rc], dim=-1))
        else:
            flow_feature = self._pool_single(rc_hidden)

        hidden = torch.tanh(self.output_layer_1(flow_feature))
        logits = self.output_layer_2(hidden)

        if tgt is None:
            return None, logits

        if self.use_scm_arcface:
            loss, logits = self._arcface_loss(hidden, tgt, td_hidden)
            return loss, logits

        if self.soft_targets and soft_tgt is not None:
            loss = self.soft_alpha * nn.MSELoss()(logits, soft_tgt) + \
                   (1 - self.soft_alpha) * nn.NLLLoss()(nn.LogSoftmax(dim=-1)(logits), tgt.view(-1))
        else:
            loss = nn.NLLLoss()(nn.LogSoftmax(dim=-1)(logits), tgt.view(-1))
        return loss, logits


def count_labels_num(path):
    labels_set, columns = set(), {}
    with open(path, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            if line_id == 0:
                for i, column_name in enumerate(line.strip().split("\t")):
                    columns[column_name] = i
                continue
            line = line.strip().split("\t")
            label = int(line[columns["label"]])
            labels_set.add(label)
    return len(labels_set)


def load_or_initialize_parameters(args, model):
    if args.pretrained_model_path is not None:
        print("Initialize with pretrained model.")
        model.load_state_dict(torch.load(args.pretrained_model_path, map_location={'cuda:1': 'cuda:0', 'cuda:2': 'cuda:0', 'cuda:3': 'cuda:0'}), strict=False)
    else:
        print("Initialize with normal distribution.")
        for n, p in list(model.named_parameters()):
            if "gamma" not in n and "beta" not in n:
                p.data.normal_(0, 0.02)


def build_optimizer(args, model):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    ]
    if args.optimizer in ["adamw"]:
        optimizer = str2optimizer[args.optimizer](optimizer_grouped_parameters, lr=args.learning_rate, correct_bias=False)
    else:
        optimizer = str2optimizer[args.optimizer](optimizer_grouped_parameters, lr=args.learning_rate,
                                                  scale_parameter=False, relative_step=False)
    if args.scheduler in ["constant"]:
        scheduler = str2scheduler[args.scheduler](optimizer)
    elif args.scheduler in ["constant_with_warmup"]:
        scheduler = str2scheduler[args.scheduler](optimizer, args.train_steps * args.warmup)
    else:
        scheduler = str2scheduler[args.scheduler](optimizer, args.train_steps * args.warmup, args.train_steps)
    return optimizer, scheduler


def batch_loader(batch_size, src, tgt, seg, soft_tgt=None):
    instances_num = src.size()[0]
    for i in range(instances_num // batch_size):
        src_batch = src[i * batch_size: (i + 1) * batch_size, :]
        tgt_batch = tgt[i * batch_size: (i + 1) * batch_size]
        seg_batch = seg[i * batch_size: (i + 1) * batch_size, :]
        if soft_tgt is not None:
            soft_tgt_batch = soft_tgt[i * batch_size: (i + 1) * batch_size, :]
            yield src_batch, tgt_batch, seg_batch, soft_tgt_batch
        else:
            yield src_batch, tgt_batch, seg_batch, None
    if instances_num > instances_num // batch_size * batch_size:
        src_batch = src[instances_num // batch_size * batch_size:, :]
        tgt_batch = tgt[instances_num // batch_size * batch_size:]
        seg_batch = seg[instances_num // batch_size * batch_size:, :]
        if soft_tgt is not None:
            soft_tgt_batch = soft_tgt[instances_num // batch_size * batch_size:, :]
            yield src_batch, tgt_batch, seg_batch, soft_tgt_batch
        else:
            yield src_batch, tgt_batch, seg_batch, None


def read_dataset(args, path):
    dataset, columns = [], {}
    with open(path, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            if line_id == 0:
                for i, column_name in enumerate(line.strip().split("\t")):
                    columns[column_name] = i
                continue
            line = line[:-1].split("\t")
            tgt = int(line[columns["label"]])
            if args.soft_targets and "logits" in columns.keys():
                soft_tgt = [float(value) for value in line[columns["logits"]].split(" ")]
            if "text_b" not in columns:
                text_a = line[columns["text_a"]]
                src = args.tokenizer.convert_tokens_to_ids([CLS_TOKEN] + args.tokenizer.tokenize(text_a))
                seg = [1] * len(src)
            else:
                text_a, text_b = line[columns["text_a"]], line[columns["text_b"]]
                src_a = args.tokenizer.convert_tokens_to_ids([CLS_TOKEN] + args.tokenizer.tokenize(text_a) + [SEP_TOKEN])
                src_b = args.tokenizer.convert_tokens_to_ids(args.tokenizer.tokenize(text_b) + [SEP_TOKEN])
                src = src_a + src_b
                seg = [1] * len(src_a) + [2] * len(src_b)

            if len(src) > args.seq_length:
                src = src[: args.seq_length]
                seg = seg[: args.seq_length]
            while len(src) < args.seq_length:
                src.append(0)
                seg.append(0)
            if args.soft_targets and "logits" in columns.keys():
                dataset.append((src, tgt, seg, soft_tgt))
            else:
                dataset.append((src, tgt, seg))

    return dataset


def train_model(args, model, optimizer, scheduler, src_batch, tgt_batch, seg_batch, soft_tgt_batch=None):
    model.zero_grad()

    src_batch = src_batch.to(args.device)
    tgt_batch = tgt_batch.to(args.device)
    seg_batch = seg_batch.to(args.device)
    if soft_tgt_batch is not None:
        soft_tgt_batch = soft_tgt_batch.to(args.device)

    loss, _ = model(src_batch, tgt_batch, seg_batch, soft_tgt_batch)
    if torch.cuda.device_count() > 1:
        loss = torch.mean(loss)

    if args.fp16:
        with args.amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        loss.backward()

    optimizer.step()
    scheduler.step()

    return loss


def evaluate(args, dataset, print_confusion_matrix=False):
    src = torch.LongTensor([sample[0] for sample in dataset])
    tgt = torch.LongTensor([sample[1] for sample in dataset])
    seg = torch.LongTensor([sample[2] for sample in dataset])

    batch_size = args.batch_size
    correct = 0
    confusion = torch.zeros(args.labels_num, args.labels_num, dtype=torch.long)
    y_true, y_pred = [], []
    args.model.eval()

    for src_batch, tgt_batch, seg_batch, _ in batch_loader(batch_size, src, tgt, seg):
        src_batch = src_batch.to(args.device)
        tgt_batch = tgt_batch.to(args.device)
        seg_batch = seg_batch.to(args.device)
        with torch.no_grad():
            _, logits = args.model(src_batch, tgt_batch, seg_batch)
        pred = torch.argmax(nn.Softmax(dim=1)(logits), dim=1)
        gold = tgt_batch
        for j in range(pred.size()[0]):
            confusion[pred[j], gold[j]] += 1
            y_true.append(gold[j].cpu())
            y_pred.append(pred[j].cpu())
        correct += torch.sum(pred == gold).item()

    if print_confusion_matrix:
        print("Confusion matrix:")
        print(confusion)
        print("Report precision, recall, and f1:")
        eps = 1e-9
        for i in range(confusion.size()[0]):
            p = confusion[i, i].item() / (confusion[i, :].sum().item() + eps)
            r = confusion[i, i].item() / (confusion[:, i].sum().item() + eps)
            f1 = 0 if (p + r) == 0 else 2 * p * r / (p + r)
            print("Label {}: {:.3f}, {:.3f}, {:.3f}".format(i, p, r, f1))

    print("Acc. (Correct/Total): {:.4f} ({}/{}) ".format(correct / len(dataset), correct, len(dataset)))
    print("Macro precision: {:.4f}, Micro precision: {:.4f}, Weighted precision: {:.4f}".format(
        precision_score(y_true, y_pred, average='macro'), precision_score(y_true, y_pred, average='micro'), precision_score(y_true, y_pred, average='weighted')))
    print("Macro recall: {:.4f}, Micro recall: {:.4f}, Weighted recall: {:.4f}".format(
        recall_score(y_true, y_pred, average='macro'), recall_score(y_true, y_pred, average='micro'), recall_score(y_true, y_pred, average='weighted')))
    print("Macro f1: {:.4f}, Micro f1: {:.4f}, Weighted f1: {:.4f}".format(
        f1_score(y_true, y_pred, average='macro'), f1_score(y_true, y_pred, average='micro'), f1_score(y_true, y_pred, average='weighted')))

    return f1_score(y_true, y_pred, average='macro'), confusion


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    finetune_opts(parser)

    parser.add_argument("--pooling", choices=["mean", "max", "first", "last"], default="first", help="Pooling type.")
    parser.add_argument("--earlystop", type=int, default=5, help="early stop rounds.")
    parser.add_argument("--tokenizer", choices=["bert", "char", "space"], default="bert",
                        help="Specify the tokenizer."
                             "Original Google BERT uses bert tokenizer on Chinese corpus."
                             "Char tokenizer segments sentences into characters."
                             "Space tokenizer segments sentences into words according to space.")
    parser.add_argument("--soft_targets", action='store_true', help="Train model with logits.")
    parser.add_argument("--soft_alpha", type=float, default=0.5, help="Weight of the soft targets loss.")

    parser.add_argument("--use_hybrid_pooling", action="store_true", help="Use max+avg pooling for TD and RC streams.")
    parser.add_argument("--use_scm_arcface", action="store_true", help="Use side-channel modulated ArcFace loss.")
    parser.add_argument("--arcface_s", type=float, default=30.0, help="ArcFace feature scale.")
    parser.add_argument("--arcface_m_base", type=float, default=0.2, help="Base ArcFace margin.")
    parser.add_argument("--arcface_m_lambda", type=float, default=0.3, help="Dynamic ArcFace margin coefficient.")
    parser.add_argument("--pce_hidden_size", type=int, default=128, help="PCE hidden layer size.")

    parser.add_argument("--is_moe", action="store_true", help="adopt moe layer.")
    parser.add_argument("--vocab_size", type=int, required=False, help="Number of vocab.")
    parser.add_argument("--moebert_expert_dim", type=int, required=False, default=3072, help="Dim of expert,default is ffn.")
    parser.add_argument("--moebert_expert_num", type=int, required=False, help="Number of expert.")
    parser.add_argument("--moebert_route_method", choices=["gate-token", "gate-sentence", "hash-random", "hash-balance", "proto"], default="hash-random",
                        help="moebert route method.")
    parser.add_argument("--moebert_route_hash_list", default=None, type=str, help="Path of moebert hash list file.")
    parser.add_argument("--moebert_load_balance", type=float, default=0.0, help="gate loss weight.")

    args = parser.parse_args()
    args = load_hyperparam(args)
    set_seed(args.seed)

    if args.train_path is None:
        args.labels_num = 197
    else:
        args.labels_num = count_labels_num(args.train_path)

    args.tokenizer = str2tokenizer[args.tokenizer](args)
    model = Classifier(args)
    load_or_initialize_parameters(args, model)

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(args.device)

    if args.train_path is None:
        args.model = model
        args.labels_num = 197
        print("No train data, only evaluate..")
        evaluate(args, read_dataset(args, args.dev_path))
        return

    trainset = read_dataset(args, args.train_path)
    random.shuffle(trainset)
    instances_num = len(trainset)
    batch_size = args.batch_size

    src = torch.LongTensor([example[0] for example in trainset])
    tgt = torch.LongTensor([example[1] for example in trainset])
    seg = torch.LongTensor([example[2] for example in trainset])
    soft_tgt = torch.FloatTensor([example[3] for example in trainset]) if args.soft_targets else None

    args.train_steps = int(instances_num * args.epochs_num / batch_size) + 1

    print("Batch size: ", batch_size)
    print("The number of training instances:", instances_num)

    optimizer, scheduler = build_optimizer(args, model)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
        args.amp = amp

    if torch.cuda.device_count() > 1:
        print("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)
    args.model = model

    total_loss, best_result = 0.0, 0.0
    best_result_round = 0

    for epoch in tqdm.tqdm(range(1, args.epochs_num + 1)):
        model.train()
        for i, (src_batch, tgt_batch, seg_batch, soft_tgt_batch) in enumerate(batch_loader(batch_size, src, tgt, seg, soft_tgt)):
            loss = train_model(args, model, optimizer, scheduler, src_batch, tgt_batch, seg_batch, soft_tgt_batch)
            total_loss += loss.item()
            if (i + 1) % args.report_steps == 0:
                print("Epoch id: {}, Training steps: {}, Avg loss: {:.3f}".format(epoch, i + 1, total_loss / args.report_steps))
                total_loss = 0.0

        result = evaluate(args, read_dataset(args, args.dev_path))
        if result[0] > best_result:
            best_result = result[0]
            best_result_round = epoch
            save_model(model, args.output_model_path)
        elif epoch - best_result_round >= args.earlystop:
            print("early stopping...")
            break

    if args.test_path is not None:
        print("Test set evaluation.")
        if torch.cuda.device_count() > 1:
            model.module.load_state_dict(torch.load(args.output_model_path))
        else:
            model.load_state_dict(torch.load(args.output_model_path))
        evaluate(args, read_dataset(args, args.test_path), True)


if __name__ == "__main__":
    main()
