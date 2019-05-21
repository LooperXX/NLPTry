#coding=utf-8
import argparse
import json
import pickle
import re
import sys
import time
from struct import unpack

import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from nltk import tokenize

np.random.seed(2)
t.manual_seed(2)

#################################################
# Hyper parameter
use_gpu = True

#################################################

# Read word embedding from binary file


class WordEmbedding(object):
    def __init__(self, input_file, vocabulary):
        self.word_to_id = {}
        self.id_to_word = {}
        self.vectors = self.read_embedding(input_file, vocabulary)

    # read words representation from given file
    def read_embedding(self, input_file, vocabulary):
        wid = 0
        em_list = []

        with open(input_file, 'rb') as f:
            cols = f.readline().strip().split()  # read first line
            vocab_size = int(cols[0].decode())  # get vocabulary size
            embedding_size = int(cols[1].decode())  # get word vector size

            # add embedding for the padding word
            em_list.append(np.zeros([1, embedding_size]))
            wid += 1

            # add embedding for out of vocabulary word
            self.word_to_id['<unk>'] = wid
            self.id_to_word[wid] = '<unk>'
            em_list.append(np.zeros([1, embedding_size]))
            wid += 1

            # set read format: get vector for one word in one read operation
            fmt = str(embedding_size) + 'f'

            for i in range(0, vocab_size, 1):
                # init one word with empty string
                vocab = b''

                # read char from the line till ' '
                ch = b''
                while ch != b' ':
                    vocab += ch
                    ch = f.read(1)

                # convert word from binary to string
                vocab = vocab.decode()

                # read one word vector
                word_vector = list(unpack(fmt, f.read(4 * embedding_size))),
                one_vec = np.asarray(word_vector, dtype=np.float32)

                # If your embedding file has '\n' at the end of each line,
                # uncomment the below line.
                # If your embedding file has no '\n' at the end of each line,
                # comment the below line
                # f.read(1)

                if vocab not in vocabulary:
                    if vocab == 'unk':
                        em_list[1] = one_vec
                    continue

                # stored the word, word id and word representation
                self.word_to_id[vocab] = wid
                self.id_to_word[wid] = vocab
                em_list.append(one_vec)
                wid += 1

        vectors = np.asarray(em_list, dtype=np.float32)
        vectors = vectors.reshape(vectors.shape[0], vectors.shape[2])
        return vectors


# Read sentence pairs from SNLI data set
class SNLI(object):
    def __init__(self, embedding, snli_path):
        cols = snli_path.split('/')
        train_file = snli_path + '_train.jsonl'
        dev_file = snli_path + '_dev.jsonl'
        test_file = snli_path + '_test.jsonl'

        if not embedding:
            self.vocab = set()
            self.collect_vocab([train_file, dev_file, test_file])
        else:
            self.word_to_id = embedding.word_to_id
            self.id_to_word = embedding.id_to_word
            self.vectors = embedding.vectors
            self.max_sent_len = 0
            self.label_dict = {'entailment': 0,
                               'neutral': 1,
                               'contradiction': 2}

            self.train_set = self.load_data(train_file)
            self.dev_set = self.load_data(dev_file)
            self.test_set = self.load_data(test_file)

    # tokenize the given text
    def tokenize_text(self, text):
        text = text.replace('\\', '')
        text = re.sub(r'\.+', '.', text)

        # split text into sentences
        sents = tokenize.sent_tokenize(text)

        for sent in sents:
            # split sent into words
            tokens = tokenize.word_tokenize(sent)

            # ignore empty sentences
            if not tokens:
                continue

            # create an iterator for tokenized words
            for token in tokens:
                ntokens = token.split('-')
                if len(ntokens) == 1:
                    yield token
                else:
                    for one in ntokens:
                        yield one

    # collect vocabulary of the SNLI set
    def collect_vocab(self, file_list):
        for one_file in file_list:
            for line in open(one_file, 'r'):
                one_dict = json.loads(line)

                # get word list for sentence 1
                for word in self.tokenize_text(one_dict['sentence1']):
                    self.vocab.add(word)

                # get word list for sentence 2
                for word in self.tokenize_text(one_dict['sentence2']):
                    self.vocab.add(word)

    # sentence pairs and their labels
    def load_data(self, input_file):
        sent1_list = []
        sent2_list = []
        label_list = []

        for line in open(input_file, 'r'):
            one_dict = json.loads(line)

            # read label
            label = one_dict['gold_label']
            if label == '-':
                continue
            label = self.label_dict[label]

            # get word list for sentence 1
            sentence1 = []
            for x in self.tokenize_text(one_dict['sentence1']):
                if x in self.word_to_id:
                    sentence1.append(self.word_to_id[x])
                else:
                    sentence1.append(1)
            self.max_sent_len = max(self.max_sent_len, len(sentence1))

            # get word list for sentence 2
            sentence2 = []
            for x in self.tokenize_text(one_dict['sentence2']):
                if x in self.word_to_id:
                    sentence2.append(self.word_to_id[x])
                else:
                    sentence2.append(1)
            self.max_sent_len = max(self.max_sent_len, len(sentence2))

            sent1_list.append(sentence1)
            sent2_list.append(sentence2)
            label_list.append(label)

        return [sent1_list, sent2_list, label_list]

    def list_to_array(self, sent_list, max_len):
        selist = []
        length = []
        for one in sent_list:
            length.append(len(one))
            if len(one) < max_len:
                one.extend(list(np.zeros(max_len - len(one),
                                         dtype=np.int32)))
            selist.append(one)

        selist = np.asarray(selist, dtype=np.int32)
        length = np.asarray(length, dtype=np.int32)

        return selist, length

    def create_padding(self, data_set):
        sent_1v, sent_1l = self.list_to_array(data_set[0], self.max_sent_len)
        sent_2v, sent_2l = self.list_to_array(data_set[1], self.max_sent_len)
        data = [sent_1v, sent_1l, sent_2v, sent_2l,
                np.asarray(data_set[2], dtype=np.int32)]
        return data

    def create_padding_set(self):
        train_set = self.create_padding(self.train_set)
        dev_set = self.create_padding(self.dev_set)
        test_set = self.create_padding(self.test_set)
        return train_set, dev_set, test_set


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
        self.embeddings = nn.Embedding(1000, 300)
        # self.embeddings = embeddings
        input_size = 300
        # input_size = self.embeddings.weight.size()[1]
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

        batch_size = rep1.size()[0]

        sents_premise = self.embeddings(rep1).transpose(1, 0)  # (sequence_length, batch_size, feature_size)
        sents_hypothesis = self.embeddings(rep2).transpose(1, 0)

        (hx, cx) = self.init_hidden(batch_size)
        hx = hx.view(batch_size, -1)
        cx = cx.view(batch_size, -1)
        hidden = (hx, cx)

        # Ouput of LSTM: sequence (seq_length, mini batch x lstm size)
        outp = []
        hidden_states = []
        for inp in range(sents_premise.size(0)):
            hidden = self.lstm1(sents_premise[inp], hidden)
            outp += [hidden[0]]
            hidden_states += [hidden[1]]

        outp = t.stack(outp).transpose(0, 1)  # (batch_size, seq_len, lstm_size)
        len1 = (len1 - 1).view(-1, 1, 1).expand(outp.size(0), 1, outp.size(2))  # (batch_size, 1, lstm_size)
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


class RNNNet(object):
    def __init__(self, mode):
        self.mode = mode

        # Set tensor type when using GPU
        if t.cuda.is_available():
            self.use_gpu = True
            self.float_type = t.cuda.FloatTensor
            self.long_type = t.cuda.LongTensor
        # Set tensor type when using CPU
        else:
            self.use_gpu = False
            self.float_type = t.FloatTensor
            self.long_type = t.LongTensor

    # Get a batch of data from given data set.
    def get_batch(self, data_set, s, e):
        sent_1 = data_set[0]
        len_1 = data_set[1]
        sent_2 = data_set[2]
        len_2 = data_set[3]
        label = data_set[4]
        return sent_1[s:e], len_1[s:e], sent_2[s:e], len_2[s:e], label[s:e]

    # Create mask for premise sentences.
    def create_mask(self, data_set, max_length):
        length = data_set[1]
        masks = []
        for one in length:
            mask = list(np.ones(one))
            mask.extend(list(np.zeros(max_length - one)))
            masks.append(mask)
        masks = np.asarray(masks, dtype=np.float32)
        return masks

    # Evaluate the trained model on test set
    def evaluate_model(self, pred_Y, Y):
        _, idx = t.max(pred_Y, dim=1)

        # move tensor from GPU to CPU when using GPU
        if self.use_gpu:
            idx = idx.cpu()
            Y = Y.cpu()

        idx = idx.data.numpy()
        Y = Y.data.numpy()
        accuracy = np.sum(idx == Y)
        return accuracy

    # Train and evaluate SNLI models
    def train_and_evaluate(self, FLAGS, embedding, train_set, dev_set, test_set):
        class_num = 3
        num_epochs = FLAGS.num_epochs
        batch_size = FLAGS.batch_size
        learning_rate = FLAGS.learning_rate

        beta = FLAGS.beta
        drop_rate = FLAGS.dropout_rate
        lstm_size = FLAGS.lstm_size
        hidden_size = FLAGS.hidden_size

        # Word embeding
        vectors = embedding.vectors

        # Max length of input sequence
        max_seq_len = train_set[0].shape[1]

        # Create mask for first sentence
        train_mask = self.create_mask(train_set, max_seq_len)
        dev_mask = self.create_mask(dev_set, max_seq_len)
        test_mask = self.create_mask(test_set, max_seq_len)

        # Train, validate and test set size
        train_size = train_set[0].shape[0]
        dev_size = dev_set[0].shape[0]
        test_size = test_set[0].shape[0]

        # Initialize embedding matrix
        embedding = t.nn.Embedding(vectors.shape[0], vectors.shape[1], padding_idx=0)
        embedding.weight = t.nn.Parameter(t.from_numpy(vectors))
        embedding.weight.requires_grad = False

        # uncomment the below three lines to force the code to use CPU
        # self.use_gpu = False
        # self.float_type = torch.FloatTensor
        # self.long_type = torch.LongTensor

        # Define models
        model = eval("Model_" + str(self.mode))(
            self.use_gpu, lstm_size, hidden_size, drop_rate, beta, embedding, class_num
        )

        # If GPU is available, then run experiments on GPU
        if self.use_gpu:
            model.cuda()

        # ======================================================================
        # define optimizer
        #
        optimizer = t.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=learning_rate)

        # ======================================================================
        # accuracy calculation
        #
        accuracy = 0

        for i in range(num_epochs):
            # put model to training mode
            model.train()

            print(20 * '*', 'epoch', i + 1, 20 * '*')
            start_time = time.time()
            s = 0
            while s < train_size:
                model.train()
                e = min(s + batch_size, train_size)

                batch_1v, batch_1l, batch_2v, batch_2l, batch_label = \
                    self.get_batch(train_set, s, e)
                mask = train_mask[s:e]

                rep1 = t.from_numpy(batch_1v).requires_grad_(mode=False).long()
                len1 = t.from_numpy(batch_1l).requires_grad_(mode=False).long()
                rep2 = t.from_numpy(batch_2v).requires_grad_(mode=False).long()
                len2 = t.from_numpy(batch_1l).requires_grad_(mode=False).long()
                mask = t.from_numpy(mask).requires_grad_(mode=False).long()
                label = t.from_numpy(batch_label).requires_grad_(mode=False).long()

                # Forward pass: predict labels
                pred_label = model(rep1, len1, mask, rep2, len2)

                # Loss function: compute negative log likelyhood
                loss = F.nll_loss(pred_label, label)

                # Zero gradients, perform a backward pass, and update the weights.
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                s = e

            end_time = time.time()
            print('the training took: %d(s)' % (end_time - start_time))

            # Put model in evaluation mode
            model.eval()

            # Evaluate the trained model on validation set
            s = 0
            total_correct = 0
            while s < dev_size:
                e = min(s + batch_size, dev_size)
                batch_1v, batch_1l, batch_2v, batch_2l, batch_label = \
                    self.get_batch(dev_set, s, e)
                mask = dev_mask[s:e]

                rep1 = t.from_numpy(batch_1v).requires_grad_(mode=False).long()
                len1 = t.from_numpy(batch_1l).requires_grad_(mode=False).long()
                rep2 = t.from_numpy(batch_2v).requires_grad_(mode=False).long()
                len2 = t.from_numpy(batch_1l).requires_grad_(mode=False).long()
                mask = t.from_numpy(mask).requires_grad_(mode=False).long()
                label = t.from_numpy(batch_label).requires_grad_(mode=False).long()

                # Forward pass: predict labels
                pred_label = model(rep1, len1, mask, rep2, len2)

                total_correct += self.evaluate_model(pred_label, label)

                s = e

            print('accuracy of the trained model on validation set %f' %
                  (total_correct / dev_size))
            print()

            # evaluate the trained model on test set
            s = 0
            total_correct = 0
            while s < test_size:
                e = min(s + batch_size, test_size)
                batch_1v, batch_1l, batch_2v, batch_2l, batch_label = \
                    self.get_batch(test_set, s, e)
                mask = test_mask[s:e]

                rep1 = t.from_numpy(batch_1v).requires_grad_(mode=False).long()
                len1 = t.from_numpy(batch_1l).requires_grad_(mode=False).long()
                rep2 = t.from_numpy(batch_2v).requires_grad_(mode=False).long()
                len2 = t.from_numpy(batch_1l).requires_grad_(mode=False).long()
                mask = t.from_numpy(mask).requires_grad_(mode=False).long()
                label = t.from_numpy(batch_label).requires_grad_(mode=False).long()

                # Forward pass: predict labels
                pred_label = model(rep1, len1, mask, rep2, len2)

                total_correct += self.evaluate_model(pred_label, label)

                s = e

        return total_correct / test_size


def main():
    # collect vocabulary of SNLI set
    # snli = SNLI(None, sys.argv[1])
    snli = SNLI(None, './data/snli_1.0')

    # read word embedding
    embedding = WordEmbedding(sys.argv[2], snli.vocab)
    pickle.dump(embedding, open(sys.argv[3], 'wb'))

    # create SNLI data set
    # snli = SNLI(embedding, sys.argv[1])
    snli = SNLI(embedding, './data/snli_1.0')
    train_set, dev_set, test_set = snli.create_padding_set()
    pickle.dump([train_set, dev_set, test_set], open(sys.argv[4], 'wb'))

    # Set parameters for RNN Exercise.
    parser = argparse.ArgumentParser('RNN Exercise.')
    parser.add_argument('--embedding_path',
                        type=str,
                        default='data/embedding.pkl',
                        help='Path of the pretrained word embedding.')
    parser.add_argument('--snli_data_dir',
                        type=str,
                        default='data/snli_padding.pkl',
                        help='Directory to put the snli data.')
    parser.add_argument('--log_dir',
                        type=str,
                        default='logs',
                        help='Directory to put logging.')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=10,
                        help='Number of epochs to run trainer.')
    parser.add_argument('--batch_size',
                        type=int,
                        default=512,
                        help='Batch size. Must divide evenly into the dataset sizes.')
    parser.add_argument('--learning_rate',
                        type=float,
                        default=0.001,
                        help='Initial learning rate.')
    parser.add_argument('--beta',
                        type=float,
                        default=0.001,
                        help='Decay rate of L2 regulization.')
    parser.add_argument('--dropout_rate',
                        type=float,
                        default=0.1,
                        help='Dropout rate.')
    parser.add_argument('--lstm_size',
                        type=int,
                        default=100,
                        help='Size of lstm cell.')
    parser.add_argument('--hidden_size',
                        type=int,
                        default=200,
                        help='Size of hidden layer of FFN.')

    FLAGS = None
    FLAGS, unparsed = parser.parse_known_args()
    mode = int(sys.argv[1])

    # ======================================================================
    #  STEP 0: Load pre-trained word embeddings and the SNLI data set
    #

    embedding = pickle.load(open(FLAGS.embedding_path, 'rb'))

    snli = pickle.load(open(FLAGS.snli_data_dir, 'rb'))
    train_set = snli[0]
    dev_set = snli[1]
    test_set = snli[2]

    # ====================================================================
    # Use a smaller portion of training examples (e.g. ratio = 0.1)
    # for debuging purposes.
    # Set ratio = 1 for training with all training examples.

    ratio = 1

    train_size = train_set[0].shape[0]
    idx = list(range(train_size))
    idx = np.asarray(idx, dtype=np.int32)

    # Shuffle the train set.
    for i in range(7):
        np.random.seed(i)
        np.random.shuffle(idx)

    # Get a certain ratio of the training set.
    idx = idx[0:int(idx.shape[0] * ratio)]
    sent1 = train_set[0][idx]
    leng1 = train_set[1][idx]
    sent2 = train_set[2][idx]
    leng2 = train_set[3][idx]
    label = train_set[4][idx]

    train_set = [sent1, leng1, sent2, leng2, label]

    # ======================================================================
    #  STEP 3: Train the third model.
    #  This model use an attention mechanism, where the attention weights
    #  are computed between the last output of the hypothesis LSTM and all
    #  the outputs of the premise LTSM.
    #
    #  Accuracy: 79.2%
    #

    if mode == 3:
        rnn = RNNNet(3)
        accuracy = rnn.train_and_evaluate(FLAGS, embedding, train_set, dev_set, test_set)

        # ======================================================================
        # output accuracy
        #
        print(20 * '*' + 'model 3' + 20 * '*')
        print('accuracy is %f' % (accuracy))
        print()

# model = TwoWayWordByWordAttentionLSTM(128, 256, 0.5, None, 3)
# rep1 = t.zeros([32, 50]).long()
# len1 = (t.ones(32) * 50).long()
# rep2 = t.zeros([32, 30]).long()
# len2 = (t.ones(32) * 30).long()
# mask1 = t.ones_like(rep1)
# out = model(rep1, len1, mask1, rep2, len2)


if __name__ == '__main__':
    main()