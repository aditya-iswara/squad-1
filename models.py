"""Top-level model classes.

Author:
    Chris Chute (chute@stanford.edu)
"""

import layers
import torch
import torch.nn as nn

from cnn import CNN

class BiDAFTransformer(nn.Module):
    """BiDAF transformer model with character-level embeddings for SQuAD

    Structure:
        - Embedding layer: Generate word embedding and character embedding
        - Encoder layer: Encode the embedded sequence
        # - Attention layer: Apply an attention mechanism to the encoded sequence.
        - Transformer layer:
        - Model encoder layer: Encode the sequence again.
        - Output layer: Simple layer (e.g., fc + softmax) to get final outputs.

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        char_vectors (torch.Tensor): Pre-trained char vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
    """
    def __init__(self, word_vectors, char_vectors, hidden_size=100, drop_prob=0.1):
        super(BiDAFTransformer, self).__init__()
        self.wordemb = layers.Embedding(word_vectors=word_vectors,
                                        hidden_size=word_vectors.shape[1],
                                        drop_prob=drop_prob)

        self.charemb = layers.CharEmbedding(char_vectors=char_vectors,
                                    hidden_size=word_vectors.shape[1],
                                    drop_prob=drop_prob)

        input_size = word_vectors.shape[1]*2

        self.emb = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True, bidirectional=True) # bidirectional=True

        # self.enc = layers.RNNEncoder(input_size=hidden_size,
        #                              hidden_size=hidden_size,
        #                              num_layers=1,
        #                              drop_prob=drop_prob)
        #
        # self.att = layers.BiDAFAttention(hidden_size=2 * hidden_size,
        #                                  drop_prob=drop_prob)

        self.gatedatt = layers.GatedAttention(enc_size=2*hidden_size)



        self.selfatt = layers.SelfAttention(input_size=2*hidden_size,
                                             hidden_size=hidden_size, drop_prob=drop_prob)

        # self.mod = layers.RNNEncoder(input_size=8 * hidden_size,
        #                              hidden_size=hidden_size,
        #                              num_layers=2,
        #                              drop_prob=drop_prob)
        #
        # self.out = layers.BiDAFOutput(hidden_size=hidden_size,
        #                               drop_prob=drop_prob)
        self.output = layers.RNETOutput(hidden_size=2*hidden_size)

    def forward(self, cw_idxs, qw_idxs, cc_idxs, qc_idxs):
        # cw_mask = torch.zeros_like(cw_idxs) != cw_idxs
        # cc_mask = torch.zeros_like(cc_idxs) != cc_idxs
        # print(cw_mask.shape)
        # print(cc_mask.shape)
        # c_mask = torch.cat((cw_mask, cc_mask), 2)
        # qw_mask = torch.unsqueeze(torch.zeros_like(qw_idxs) != qw_idxs, 2)
        # qc_mask = torch.zeros_like(qc_idxs) != qc_idxs
        # q_mask = torch.cat((qw_mask, qc_mask), 2)
        # c_len, q_len = c_mask.sum(-1).sum(-1), q_mask.sum(-1).sum(-1)

        # c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        # c_mask = torch.cat((c_mask, torch.ones_like(c_mask)), 1)
        # # cc_mask = torch.zeros_like(cc_idxs) != cc_idxs
        # q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        # q_mask = torch.cat((q_mask, torch.ones_like(q_mask)), 1)
        # # qc_mask = torch.zeros_like(qc_idxs) != qc_idxs
        # c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        c_wordemb = self.wordemb(cw_idxs)
        c_charemb = self.charemb(cc_idxs)
        q_wordemb = self.wordemb(qw_idxs)
        q_charemb = self.charemb(qc_idxs)

        # print("c_wordemb:", c_wordemb.shape)

        c_emb = torch.cat((c_wordemb, c_charemb), 2)
        q_emb = torch.cat((q_wordemb, q_charemb), 2)

        # print("c_emb:", c_emb.shape)
        # print("q_emb:", q_emb.shape)

        c_emb, _ = self.emb(c_emb)
        q_emb, _ = self.emb(q_emb)

        # print("c_emb:", c_emb.shape)
        # print("q_emb:", q_emb.shape)

        v = self.gatedatt(c_emb, q_emb)

        h = self.selfatt(v)

        out = self.output(q_emb, h)

        # c_enc = self.enc(c_emb, c_len)    # (batch_size, c_len, 2 * hidden_size)
        # q_enc = self.enc(q_emb, q_len)    # (batch_size, q_len, 2 * hidden_size)
        #
        # att = self.att(c_enc, q_enc,
        #                c_mask, q_mask)    # (batch_size, c_len, 8 * hidden_size)
        #
        # #selfatt = self.selfatt(att, )
        #
        # mod = self.mod(att, c_len)        # (batch_size, c_len, 2 * hidden_size)
        #
        # out = self.out(att, mod, c_mask)  # 2 tensors, each (batch_size, c_len)

        return out


class BiDAF(nn.Module):
    """Baseline BiDAF model for SQuAD.

    Based on the paper:
    "Bidirectional Attention Flow for Machine Comprehension"
    by Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi
    (https://arxiv.org/abs/1611.01603).

    Follows a high-level structure commonly found in SQuAD models:
        - Embedding layer: Embed word indices to get word vectors.
        - Encoder layer: Encode the embedded sequence.
        - Attention layer: Apply an attention mechanism to the encoded sequence.
        - Model encoder layer: Encode the sequence again.
        - Output layer: Simple layer (e.g., fc + softmax) to get final outputs.

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
    """
    def __init__(self, word_vectors, hidden_size, drop_prob=0.):
        super(BiDAF, self).__init__()
        self.emb = layers.Embedding(word_vectors=word_vectors,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob)

        self.enc = layers.RNNEncoder(input_size=hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     drop_prob=drop_prob)

        self.att = layers.BiDAFAttention(hidden_size=2 * hidden_size,
                                         drop_prob=drop_prob)

        self.mod = layers.RNNEncoder(input_size=8 * hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=2,
                                     drop_prob=drop_prob)

        self.out = layers.BiDAFOutput(hidden_size=hidden_size,
                                      drop_prob=drop_prob)

    def forward(self, cw_idxs, qw_idxs):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        c_emb = self.emb(cw_idxs)         # (batch_size, c_len, hidden_size)
        q_emb = self.emb(qw_idxs)         # (batch_size, q_len, hidden_size)

        c_enc = self.enc(c_emb, c_len)    # (batch_size, c_len, 2 * hidden_size)
        q_enc = self.enc(q_emb, q_len)    # (batch_size, q_len, 2 * hidden_size)

        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)    # (batch_size, c_len, 8 * hidden_size)

        mod = self.mod(att, c_len)        # (batch_size, c_len, 2 * hidden_size)

        out = self.out(att, mod, c_mask)  # 2 tensors, each (batch_size, c_len)

        return out
