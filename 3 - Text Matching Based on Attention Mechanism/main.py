import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F

np.random.seed(2)
t.manual_seed(2)
use_gpu = True

# Model 3: Use attention for last LSTM output of hypothesis.
#
#  This model use an attention mechanism, where the attention weights
#  are computed between the last output of the hypothesis LSTM and all
#  the outputs of the premise LTSM.
#
#  lstm_size: the size of the LSTM cell.
#  hidden_size: the size of the fully connected layers.
#  drop_rate: Dropout rate.
#  beta: the L2 regularizer parameter for the fully connected layers.
#  rep_1: the matrix of word embeddings for the premise sentence.
#  len_1: the true length of the premise sentence.
#  mask_1: binary vector specifying true words (1) and dummy words used for padding (0).
#  rep_2: the matrix of word embeddings for the hypothesis sentence.
#  len_2: the true length of the hypothesis sentence.

class TwoWayWordByWordAttentionLSTM(nn.Module):
    def __init__(self, lstm_size, hidden_size, drop_out, embeddings, class_num):
        super(TwoWayWordByWordAttentionLSTM, self).__init__()
        self.lstm_size = lstm_size
        self.hidden_size = hidden_size
        self.embeddings = embeddings
        input_size = self.embedding.weight.size()[1]
        self.drop_out = drop_out

        # The LSTMs: lstm1 - premise; lstm2 - hypothesis
        self.lstm1 = nn.LSTMCell(input_size, lstm_size)
        self.lstm2 = nn.LSTM(input_size, lstm_size, 1)

        # The fully connectedy layers
        self.linear1 = nn.Linear(lstm_size, hidden_size)

        # The fully connectedy layer for softmax
        self.linear2 = nn.Linear(hidden_size, class_num)

        # transformation of the states
        u_min = -0.5
        u_max = 0.5

        self.Wy = nn.Parameter(t.Tensor(lstm_size, lstm_size).uniform_(u_min, u_max))
        self.Wh = nn.Parameter(t.Tensor(lstm_size, lstm_size).uniform_(u_min, u_max))
        self.Wp = nn.Parameter(t.Tensor(lstm_size, lstm_size).uniform_(u_min, u_max))
        self.Wx = nn.Parameter(t.Tensor(lstm_size, lstm_size).uniform_(u_min, u_max))
        self.aW = nn.Parameter(t.Tensor(1, lstm_size).uniform_(u_min, u_max))

    # Initialize hidden states and cell states of LSTM
    def init_hidden(self, batch_size):
        return (t.zeros(1, batch_size, self.lstm_size, requires_grad=False).float(),
                t.zeros(1, batch_size, self.lstm_size, requires_grad=False).float())

    # Forward propagation
    def forward(self, rep1, len1, mask1, rep2, len2):
        # Compute context vectors using attention.
        def context_vector(h_t):
            WhH = t.matmul(h_t, self.Wh)

            # Use mask to ignore the outputs of the padding part in premise
            shape = WhH.size()
            WhH = WhH.view(shape[0], 1, shape[1])
            WhH = WhH.expand(shape[0], max_seq_len, shape[1])

            M1 = mask1.type(self.float_type)
            shape = M1.size()
            M = M1.view(shape[0], shape[1], 1).type(self.float_type)
            M = M.expand(shape[0], shape[1], self.lstm_size)

            WhH = WhH * M
            M = t.tanh(WyY + WhH)
            aW = self.aW.view(1, 1, -1)
            aW = aW.expand(batch_size, max_seq_len, aW.size()[2])

            # Compute batch dot: the first step of a softmax
            batch_dot = M * aW
            batch_dot = t.sum(batch_dot, 2)

            # Avoid overflow
            max_by_column, _ = t.max(batch_dot, 1)
            max_by_column = max_by_column.view(-1, 1)
            max_by_column = max_by_column.expand(max_by_column.size()[0], max_seq_len)

            batch_dot = t.exp(batch_dot - max_by_column) * M1

            # Partition function and attention:
            # the second step of a softmax, use mask to ignore the padding
            partition = t.sum(batch_dot, 1)
            partition = partition.view(-1, 1)
            partition = partition.expand(partition.size()[0], max_seq_len)
            attention = batch_dot / partition

            # compute context vector
            shape = attention.size()
            attention = attention.view(shape[0], shape[1], 1)
            attention = attention.expand(shape[0], shape[1], self.lstm_size)

            cv_t = outputs_1 * attention
            cv_t = t.sum(cv_t, 1)

            return cv_t

        # ################# Forward Propagation code ###################

        # Set batch size
        batch_size = rep1.size()[0]

        # Representation of input sentences
        sent1 = self.embedding(rep1)
        sent2 = self.embedding(rep2)

        # Transform sentences representations to:
        # (sequence length * batch size * feqture size)
        sent1 = sent1.transpose(1, 0)
        sent2 = sent2.transpose(1, 0)

        # ----------------- YOUR CODE HERE ----------------------
        # Run the two LSTM's, compute the context vectors,
        # compute the final representation of the sentence pair,
        # and run it through the fully connected layer, then
        # through the softmax layer.
        rep = t.cat((rep1, rep2), 0)
        # length = t.cat((len1, len2), 0)

        # Representation for input sentences
        batch_size = rep1.size()[0]
        sents = self.embedding(rep)
        (sents_premise, sents_hypothesis) = t.split(sents, batch_size)

        # (sequence length * batch size * feature size)
        sents_premise = sents_premise.transpose(1, 0)
        sents_hypothesis = sents_hypothesis.transpose(1, 0)

        # Initialize hidden states and cell states
        (hx, cx) = self.init_hidden(batch_size)
        hx = hx.view(batch_size, -1)
        cx = cx.view(batch_size, -1)
        hidden = (hx, cx)

        # Ouput of LSTM: sequence (length x mini batch x lstm size)
        outp = []
        hidden_states = []
        for inp in range(sents_premise.size(0)):
            hidden = self.lstm1(sents_premise[inp], hidden)
            outp += [hidden[0]]
            hidden_states += [hidden[1]]

        outp = t.stack(outp).transpose(0, 1)
        len1 = (len1 - 1).view(-1, 1, 1).expand(outp.size(0), 1, outp.size(2))
        out = t.gather(outp, 1, len1).transpose(1, 0)

        hidden_states = t.stack(hidden_states).transpose(0, 1)
        # len1 = (len1-1).view(-1, 1, 1).expand(hidden_states.size(0), 1, hidden_states.size(2))
        hidden_state = t.gather(hidden_states, 1, len1).transpose(1, 0)

        lstm_outs, hidden_hypothesis = self.lstm2(sents_hypothesis, (out, hidden_state))
        lstm_outs = lstm_outs.transpose(0, 1)

        len2 = (len2 - 1).view(-1, 1, 1).expand(lstm_outs.size(0), 1, lstm_outs.size(2))
        lstm_out = t.gather(lstm_outs, 1, len2)
        lstm_out = lstm_out.view(lstm_out.size(0), -1)

        #############################################
        outputs_1 = lstm_outs
        max_seq_len = rep1.size()[1]
        WyY = t.matmul(outputs_1, self.Wy)
        context_vec = context_vector(lstm_out)
        final = t.tanh(t.matmul(context_vec, self.Wp) + t.matmul(lstm_out, self.Wh))
        #############################################

        # Concatenate premise and hypothesis representations
        final = F.dropout(final, p=self.drop_out)

        # Output of fully connected layers
        fc_out = F.dropout(F.tanh(self.linear1(lstm_out)), p=self.drop_out)

        # Output of Softmax
        fc_out = self.linear2(fc_out)

        return F.log_softmax(fc_out, dim=1)
