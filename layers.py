"""Assortment of layers for use in models.py.

Author:
    Chris Chute (chute@stanford.edu)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from util import masked_softmax
from cnn import CNN


class Embedding(nn.Module):
    """Embedding layer used by BiDAF, without the character-level component.

    Word-level embeddings are further refined using a 2-layer Highway Encoder
    (see `HighwayEncoder` class for details).

    Args:
        word_vectors (torch.Tensor)i: Pre-trained word vectors.
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations
    """
    def __init__(self, word_vectors, hidden_size, drop_prob):
        super(Embedding, self).__init__()
        self.drop_prob = drop_prob
        self.embed = nn.Embedding.from_pretrained(word_vectors)
        # print("Word vectors:", word_vectors.size())
        self.proj = nn.Linear(word_vectors.size(1), hidden_size, bias=False)
        self.hwy = HighwayEncoder(2, hidden_size)

    def forward(self, x):
        emb = self.embed(x)   # (batch_size, seq_len, embed_size)
        emb = F.dropout(emb, self.drop_prob, self.training)
        emb = self.proj(emb)  # (batch_size, seq_len, hidden_size)
        emb = self.hwy(emb)   # (batch_size, seq_len, hidden_size)

        return emb

class CharEmbedding(nn.Module):
    """Embedding layer used by BiDAF including character embeddings.

    Word-level embeddings are further refined using a 2-layer Highway Encoder
    (see `HighwayEncoder` class for details).

    Args:
        word_vectors (torch.Tensor)i: Pre-trained wo rd vectors.
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations
    """
    # def __init__(self, char_vectors, hidden_size, drop_prob, char_embed_size, word_embed_size):
    def __init__(self, char_vectors, hidden_size, drop_prob, char_embed_size=64, word_embed_size=300):
        super(CharEmbedding, self).__init__()
        self.drop_prob = drop_prob
        # print("Char vectors:", char_vectors.shape)
        self.embed = nn.Embedding.from_pretrained(char_vectors, freeze=False)
        self.CNN = CNN(char_embed_size=char_embed_size, word_embed_size=word_embed_size)
        self.proj = nn.Linear(word_embed_size, hidden_size, bias=False)
        self.hwy = HighwayEncoder(2, hidden_size)

    def forward(self, x_char):
        # print("x_char shape:", x_char.shape)
        emb = self.embed(x_char).permute(0,1,3,2)
        # print("Embed Shape:", emb.shape)

        # emb = self.CNN(emb)
        # emb = F.dropout(emb, self.drop_prob, self.training)
        # emb = self.proj(emb)
        # emb = self.hwy(emb)
        # print("Char_Embedding Shape:", emb.shape)
        #
        # return emb

        output = []
        for batch in torch.split(emb, 1, dim=0):
            # print(torch.squeeze(batch, dim=0).shape)
            x_convout = self.CNN(torch.squeeze(batch, dim=0))
            # print(x_convout.shape)
            x_convout = F.dropout(x_convout, self.drop_prob, self.training)
            # print(x_convout.shape)
            x_proj = self.proj(x_convout)
            x_unshaped = self.hwy(x_proj)
            output.append(x_unshaped)
        return torch.stack(output)

        # char_emb = self.char_embed(x_char)   # (batch_size, seq_len, embed_size)
        # emb = self.CNN(char_emb)
        # emb = torch.cat((word_emb, emb), 1)
        # emb = F.dropout(emb, self.drop_prob, self.training)
        # emb = self.proj(emb)  # (batch_size, seq_len, hidden_size)
        # emb = self.hwy(emb)   # (batch_size, seq_len, hidden_size)

        # return emb
    #
    # def __init__(self, embed_size, vocab):
    #     """
    #     Init the Embedding layer for one language
    #     @param embed_size (int): Embedding size (dimensionality) for the output
    #     @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
    #     """
    #     super(ModelEmbeddings, self).__init__()
    #     self.embed_size = embed_size
    #     self.embeddings = nn.Embedding(len(vocab.char2id), 50, vocab.word2id['<pad>'])
    #     self.Cnn = CNN(char_embed_size=50, word_embed_size=embed_size)
    #     self.Highway = Highway(word_embed_size=embed_size)
    #     self.dropout = nn.Dropout(p=0.3)
    #     ### END YOUR CODE
    #
    # def forward(self, input):
    #     """
    #     Looks up character-based CNN embeddings for the words in a batch of sentences.
    #     @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
    #         each integer is an index into the character vocabulary
    #
    #     @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the
    #         CNN-based embeddings for each word of the sentences in the batch
    #     """
    #     ## A4 code
    #     # output = self.embeddings(input)
    #     # return output
    #     ## End A4 code
    #
    #     ### YOUR CODE HERE for part 1j
    #     output = []
    #     input_emb = self.embeddings(input).permute(0,1,3,2)
    #
    #     for batch in torch.split(input_emb, 1, dim=0):
    #         # input_reshaped = self.embeddings(batch).permute(0,1,3,2)
    #         x_convout = self.Cnn(torch.squeeze(batch, dim=0))
    #         x_highway = self.Highway(x_convout)
    #
    #         x_unshaped = self.dropout(x_highway)
    #         output.append(x_unshaped)
    #
    #     return torch.stack(output)


class HighwayEncoder(nn.Module):
    """Encode an input sequence using a highway network.

    Based on the paper:
    "Highway Networks"
    by Rupesh Kumar Srivastava, Klaus Greff, Jürgen Schmidhuber
    (https://arxiv.org/abs/1505.00387).

    Args:
        num_layers (int): Number of layers in the highway encoder.
        hidden_size (int): Size of hidden activations.
    """
    def __init__(self, num_layers, hidden_size):
        super(HighwayEncoder, self).__init__()
        self.transforms = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                         for _ in range(num_layers)])
        self.gates = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                    for _ in range(num_layers)])

    def forward(self, x):
        for gate, transform in zip(self.gates, self.transforms):
            # Shapes of g, t, and x are all (batch_size, seq_len, hidden_size)
            g = torch.sigmoid(gate(x))
            t = F.relu(transform(x))
            x = g * t + (1 - g) * x

        return x


class RNNEncoder(nn.Module):
    """General-purpose layer for encoding a sequence using a bidirectional RNN.

    Encoded output is the RNN's hidden state at each position, which
    has shape `(batch_size, seq_len, hidden_size * 2)`.

    Args:
        input_size (int): Size of a single timestep in the input.
        hidden_size (int): Size of the RNN hidden state.
        num_layers (int): Number of layers of RNN cells to use.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 drop_prob=0.):
        super(RNNEncoder, self).__init__()
        self.drop_prob = drop_prob
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True,
                           bidirectional=True,
                           dropout=drop_prob if num_layers > 1 else 0.)

    def forward(self, x, lengths):
        # Save original padded length for use by pad_packed_sequence
        orig_len = x.size(1)

        # Sort by length and pack sequence for RNN
        lengths, sort_idx = lengths.sort(0, descending=True)
        x = x[sort_idx]     # (batch_size, seq_len, input_size)
        x = pack_padded_sequence(x, lengths, batch_first=True)

        # Apply RNN
        x, _ = self.rnn(x)  # (batch_size, seq_len, 2 * hidden_size)

        # Unpack and reverse sort
        x, _ = pad_packed_sequence(x, batch_first=True, total_length=orig_len)
        _, unsort_idx = sort_idx.sort(0)
        x = x[unsort_idx]   # (batch_size, seq_len, 2 * hidden_size)

        # Apply dropout (RNN applies dropout after all but the last layer)
        x = F.dropout(x, self.drop_prob, self.training)

        return x


class BiDAFAttention(nn.Module):
    """Bidirectional attention originally used by BiDAF.

    Bidirectional attention computes attention in two directions:
    The context attends to the query and the query attends to the context.
    The output of this layer is the concatenation of [context, c2q_attention,
    context * c2q_attention, context * q2c_attention]. This concatenation allows
    the attention vector at each timestep, along with the embeddings from
    previous layers, to flow through the attention layer to the modeling layer.
    The output has shape (batch_size, context_len, 8 * hidden_size).

    Args:
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob=0.1):
        super(BiDAFAttention, self).__init__()
        self.drop_prob = drop_prob
        self.c_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.q_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.cq_weight = nn.Parameter(torch.zeros(1, 1, hidden_size))
        for weight in (self.c_weight, self.q_weight, self.cq_weight):
            nn.init.xavier_uniform_(weight)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, c, q, c_mask, q_mask):
        batch_size, c_len, _ = c.size()
        q_len = q.size(1)

        #c_len //= 2
        #q_len //= 2

        s = self.get_similarity_matrix(c, q)        # (batch_size, c_len, q_len)
        c_mask = c_mask.view(batch_size, c_len, 1)  # (batch_size, c_len, 1)
        q_mask = q_mask.view(batch_size, 1, q_len)  # (batch_size, 1, q_len)
        s1 = masked_softmax(s, q_mask, dim=2)       # (batch_size, c_len, q_len)
        s2 = masked_softmax(s, c_mask, dim=1)       # (batch_size, c_len, q_len)

        # (bs, c_len, q_len) x (bs, q_len, hid_size) => (bs, c_len, hid_size)
        a = torch.bmm(s1, q)
        # (bs, c_len, c_len) x (bs, c_len, hid_size) => (bs, c_len, hid_size)
        b = torch.bmm(torch.bmm(s1, s2.transpose(1, 2)), c)

        x = torch.cat([c, a, c * a, c * b], dim=2)  # (bs, c_len, 4 * hid_size)

        return x

    def get_similarity_matrix(self, c, q):
        """Get the "similarity matrix" between context and query (using the
        terminology of the BiDAF paper).

        A naive implementation as described in BiDAF would concatenate the
        three vectors then project the result with a single weight matrix. This
        method is a more memory-efficient implementation of the same operation.

        See Also:
            Equation 1 in https://arxiv.org/abs/1611.01603
        """
        c_len, q_len = c.size(1), q.size(1)
        c = F.dropout(c, self.drop_prob, self.training)  # (bs, c_len, hid_size)
        q = F.dropout(q, self.drop_prob, self.training)  # (bs, q_len, hid_size)

        # Shapes: (batch_size, c_len, q_len)
        s0 = torch.matmul(c, self.c_weight).expand([-1, -1, q_len])
        s1 = torch.matmul(q, self.q_weight).transpose(1, 2)\
                                           .expand([-1, c_len, -1])
        s2 = torch.matmul(c * self.cq_weight, q.transpose(1, 2))
        s = s0 + s1 + s2 + self.bias

        return s

class SelfAttention(nn.Module):
    """Bidirectional attention originally used by BiDAF.

    Bidirectional attention computes attention in two directions:
    The context attends to the query and the query attends to the context.
    The output of this layer is the concatenation of [context, c2q_attention,
    context * c2q_attention, context * q2c_attention]. This concatenation allows
    the attention vector at each timestep, along with the embeddings from
    previous layers, to flow through the attention layer to the modeling layer.
    The output has shape (batch_size, context_len, 8 * hidden_size).

    Args:
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, input_size, hidden_size, drop_prob):
        super(SelfAttention, self).__init__()
        self.hidden_size = hidden_size
        self.BiRNN = nn.GRU(input_size=4*hidden_size, hidden_size=hidden_size, batch_first=True, dropout=drop_prob, bidirectional=True)
        self.linear1 = nn.Linear(input_size, input_size)
        self.linear2 = nn.Linear(input_size, input_size)
        self.softmax = nn.Softmax(dim=2)
        self.gatedAttention = nn.Linear(4*hidden_size, 4*hidden_size)


    def forward(self, v):
        hidden_states = []
        self.prevHiddenState = Variable(torch.zeros(2, v.shape[0], self.hidden_size))        # hidden_states.append(self.prevHiddenState)

        v_perm = v.permute(1,0,2)
        for context_word in torch.split(v_perm, 1):
            currentword_mult = self.linear1(context_word)
            context_mult = self.linear2(v_perm)
            f = torch.tanh(currentword_mult + context_mult)
            f = f.permute(1,2,0) # 64, 200, 357
            pre_softmax = (context_word.permute(1,0,2)).matmul(f) #64, 1, 357
            attention_vec = self.softmax(pre_softmax)
            cell_state = attention_vec.matmul(v)

            preGateInput = torch.cat((context_word.permute(1,0,2), cell_state), dim=2)

            gate = torch.sigmoid(self.gatedAttention(preGateInput))
            gate_input = (preGateInput * gate) #64, 1, 400

            output, self.prevHiddenState = self.BiRNN(gate_input, self.prevHiddenState)
            h = self.prevHiddenState
            h = torch.cat((h[0, :, :], h[1, :, :]), dim=1)
            hidden_states.append(h)
        return torch.stack(hidden_states)

class GatedAttention(nn.Module):
    def __init__(self, enc_size):
        super(GatedAttention, self).__init__()
        self.enc_size = enc_size
        self.uq = nn.Linear(enc_size, enc_size)
        self.uc = nn.Linear(enc_size, enc_size)
        self.vq = nn.Linear(enc_size, enc_size)
        self.softmax = nn.Softmax(dim=2)
        self.gatedAttention = nn.Linear(2*enc_size, 2*enc_size)
        self.prevHiddenState = Variable(torch.zeros(self.enc_size,))
        self.rnn = nn.GRU(2*enc_size, enc_size, num_layers=1, batch_first=True)

    def forward(self, c, q):
        hidden_states = []
        self.prevHiddenState = Variable(torch.zeros(1, c.shape[0], self.enc_size))

        c_perm = c.permute(1,0,2)
        q_perm = q.permute(1,0,2)
        for context_word in torch.split(c_perm, 1):
            hidden_mult = self.vq(self.prevHiddenState)
            context_mult = self.uc(context_word)
            question_mult = self.uq(q_perm)
            f = torch.tanh(hidden_mult + context_mult + question_mult)
            # self.prevHiddenState = self.prevHiddenState.permute(1,0,2)
            f = f.permute(1,2,0)
            pre_softmax = (self.prevHiddenState.permute(1,0,2)).matmul(f)
            attention_vec = self.softmax(pre_softmax)
            # print("attention_vec shape:", attention_vec.shape)
            # cell_state = q.matmul(attention_vec)
            cell_state = attention_vec.matmul(q)

            preGateInput = torch.cat((context_word.permute(1,0,2), cell_state), dim=2)

            gate = torch.sigmoid(self.gatedAttention(preGateInput))
            gate_input = (preGateInput*gate)

            output, self.prevHiddenState = self.rnn(gate_input, self.prevHiddenState)
            # h = self.prevHiddenState
            # h = torch.cat((h[0,:,:],h[1,:,:]),dim=1)
            hidden_states.append(self.prevHiddenState.squeeze())
        return torch.stack(hidden_states).permute(1,0,2)

class RNETOutput(nn.Module):
    def __init__(self, hidden_size):
        super(RNETOutput, self).__init__()
        self.initial_linear = nn.Linear(hidden_size,hidden_size)
        self.h_linear = nn.Linear(hidden_size, hidden_size)
        self.ha_linear = nn.Linear(hidden_size, hidden_size)
        self.vt_linear = nn.Linear(hidden_size, 1)
        self.rnn = nn.GRU(hidden_size, hidden_size, num_layers=1, batch_first=True)



    def forward(self, ques, h):
        q = self.initial_linear(ques) #64,23,200
        a_t = F.softmax(self.vt_linear(torch.tanh(q)), dim=2)
        # r_q = torch.sum(a_t*q,dim=1)
        r_q = (q.permute(0,2,1)).matmul(a_t)
        self.hidden_ans = r_q

        h_val = self.h_linear(h.permute(1,0,2))
        h_a = self.ha_linear(self.hidden_ans.permute(0,2,1))
        output = self.vt_linear(torch.tanh(h_val + h_a))
        p_1 = F.log_softmax(output, dim=1)

        c_t = h.permute(1,2,0).matmul(output)
        trash, self.hidden_ans = self.rnn(c_t.permute(0,2,1), self.hidden_ans.permute(2,0,1))

        h_val = self.h_linear(h.permute(1,0,2))
        h_a = self.ha_linear(self.hidden_ans.permute(1,0,2))
        output = self.vt_linear(torch.tanh(h_val + h_a))
        p_2 = F.log_softmax(output, dim=1)

        return p_1, p_2


class BiDAFOutput(nn.Module):
    """Output layer used by BiDAF for question answering.

    Computes a linear transformation of the attention and modeling
    outputs, then takes the softmax of the result to get the start pointer.
    A bidirectional LSTM is then applied the modeling output to produce `mod_2`.
    A second linear+softmax of the attention output and `mod_2` is used
    to get the end pointer.

    Args:
        hidden_size (int): Hidden size used in the BiDAF model.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob):
        super(BiDAFOutput, self).__init__()
        self.att_linear_1 = nn.Linear(8 * hidden_size, 1)
        self.mod_linear_1 = nn.Linear(2 * hidden_size, 1)

        self.rnn = RNNEncoder(input_size=2 * hidden_size,
                              hidden_size=hidden_size,
                              num_layers=1,
                              drop_prob=drop_prob)

        self.att_linear_2 = nn.Linear(8 * hidden_size, 1)
        self.mod_linear_2 = nn.Linear(2 * hidden_size, 1)

    def forward(self, att, mod, mask):
        # Shapes: (batch_size, seq_len, 1)
        logits_1 = self.att_linear_1(att) + self.mod_linear_1(mod)
        mod_2 = self.rnn(mod, mask.sum(-1))
        logits_2 = self.att_linear_2(att) + self.mod_linear_2(mod_2)

        # Shapes: (batch_size, seq_len)
        log_p1 = masked_softmax(logits_1.squeeze(), mask, log_softmax=True)
        log_p2 = masked_softmax(logits_2.squeeze(), mask, log_softmax=True)

        return log_p1, log_p2
