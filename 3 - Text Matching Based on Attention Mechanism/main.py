# coding=utf-8
import json
import pickle
import re
import time
from struct import unpack
import math
import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from nltk import tokenize
from tqdm import tqdm

np.random.seed(2)
t.manual_seed(2)

#################################################
# Hyper parameter
use_gpu = True
ratio = 1
epochs = 10
classes = 3
batch_size = 512
learning_rate = 0.001
beta = 0.001
dropout_rate = 0.1
lstm_size = 100
hidden_size = 200
freeze_index = None

# Other parameters
u_min = -0.5
u_max = 0.5
data_source_path = './data/snli_1.0'
data_path = './data/snli_1.0.pkl'
embedding_source_path = '../../GoogleNews-vectors-negative300.bin'
embedding_path = './data/embedding.pkl'
load_model_path = None


#################################################

# Read word embedding from binary file
class WordEmbedding(object):
    def __init__(self, input_file, vocabulary):
        self.word_to_id = {}
        self.id_to_word = {}
        self.embeddings = self.read_embedding(input_file, vocabulary)

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
            freeze_index = wid
            vocab_set = set(vocabulary)

            for vocab in vocab_set:
                if vocab in self.word_to_id:
                    continue
                one_vec = t.empty((1, embedding_size)).uniform_(u_min, u_max).float().numpy()
                em_list.append(one_vec)
                self.word_to_id[vocab] = wid
                self.id_to_word[wid] = vocab
                wid += 1

        embeddings = np.asarray(em_list, dtype=np.float32)
        embeddings = embeddings.reshape(embeddings.shape[0], embeddings.shape[2])
        return embeddings


# Read sentence pairs from SNLI data set
class SNLI(object):
    def __init__(self, embedding, snli_path):
        train_file = snli_path + '_train.jsonl'
        dev_file = snli_path + '_dev.jsonl'
        test_file = snli_path + '_test.jsonl'

        if not embedding:
            self.vocab = set()
            self.collect_vocab([train_file, dev_file, test_file])
        else:
            self.word_to_id = embedding.word_to_id
            self.id_to_word = embedding.id_to_word
            self.embeddings = embedding.embeddings
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

    # collect vocabulary of the SNLI
    def collect_vocab(self, file_list):
        for one_file in file_list:
            for line in open(one_file, 'r'):
                one_dict = json.loads(line)
                for word in self.tokenize_text(one_dict['sentence1']):
                    self.vocab.add(word)
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
                else:  # oov 初始化为unk
                    sentence1.append(1)
            self.max_sent_len = max(self.max_sent_len, len(sentence1))

            # get word list for sentence 2
            sentence2 = []
            for x in self.tokenize_text(one_dict['sentence2']):
                if x in self.word_to_id:
                    sentence2.append(self.word_to_id[x])
                else:  # oov 初始化为unk
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


class TwoWayWordByWordAttentionLSTM(nn.Module):
    def __init__(self, lstm_size, hidden_size, dropout_rate, embeddings, class_num, beta):
        super(TwoWayWordByWordAttentionLSTM, self).__init__()
        self.lstm_size = lstm_size
        self.hidden_size = hidden_size
        self.embeddings = embeddings
        embedding_size = self.embeddings.weight.size()[1]
        # self.embeddings = nn.Embedding(1000, 300)
        # embedding_size = 300
        self.dropout_rate = dropout_rate

        # The LSTMs: lstm1 - premise; lstm2 - hypothesis
        self.lstm1 = nn.LSTMCell(embedding_size, lstm_size)
        self.lstm2 = nn.LSTM(embedding_size, lstm_size, 1)

        # The fully connectedy layers
        self.linear1 = nn.Linear(lstm_size * 2, hidden_size)

        # The fully connectedy layer for softmax
        self.linear2 = nn.Linear(hidden_size, class_num)

        self.Wy = nn.Parameter(t.empty(lstm_size, lstm_size).uniform_(u_min, u_max))
        self.Wh = nn.Parameter(t.empty(lstm_size, lstm_size).uniform_(u_min, u_max))
        self.Wr = nn.Parameter(t.empty(lstm_size, lstm_size).uniform_(u_min, u_max))
        self.Wp = nn.Parameter(t.empty(lstm_size, lstm_size).uniform_(u_min, u_max))
        self.Wx = nn.Parameter(t.empty(lstm_size, lstm_size).uniform_(u_min, u_max))
        self.aW = nn.Parameter(t.empty(1, lstm_size).uniform_(u_min, u_max))

    # Initialize hidden states and cell states of LSTM
    def init_hidden(self, batch_size):
        return (t.zeros(1, batch_size, self.lstm_size, requires_grad=False).float(),
                t.zeros(1, batch_size, self.lstm_size, requires_grad=False).float())

    # Compute context vectors using attention.
    def context_vector(self, h_t, out, WyY, mask, batch_size, max_seq_len):
        WhH = t.matmul(h_t, self.Wh)

        # Use mask to ignore the outputs of the padding part in premise
        shape = WhH.size()
        WhH = WhH.view(shape[0], 1, shape[1])
        WhH = WhH.expand(shape[0], max_seq_len, shape[1])

        M1 = mask.float()
        shape = M1.size()
        M = M1.view(shape[0], shape[1], 1).float()
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

        cv_t = out * attention
        cv_t = t.sum(cv_t, 1)

        return cv_t

    # Word-by-Word Attention Compute context vectors using attention.
    def context_vector_(self, h_t, out, WyY, context_vector, mask, batch_size, max_seq_len):
        WhH = t.matmul(h_t, self.Wh)
        WrC = t.matmul(context_vector, self.Wr)
        WhH = WhH + WrC
        # Use mask to ignore the outputs of the padding part in premise
        shape = WhH.size()
        WhH = WhH.view(shape[0], 1, shape[1])
        WhH = WhH.expand(shape[0], max_seq_len, shape[1])

        M1 = mask.float()
        shape = M1.size()
        M = M1.view(shape[0], shape[1], 1).float()
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

        cv_t = out * attention
        cv_t = t.sum(cv_t, 1)

        return cv_t

    def get_representation(self, sen_1, sen_2, len_1, mask_1, len_2, batch_size, max_seq_len):
        (hx, cx) = self.init_hidden(batch_size)
        hx = hx.view(batch_size, -1).cuda()
        cx = cx.view(batch_size, -1).cuda()
        hidden = (hx, cx)
        outp = []
        hidden_states = []
        for inp in range(sen_1.size(0)):
            hidden = self.lstm1(sen_1[inp], hidden)
            outp += [hidden[0]]
            hidden_states += [hidden[1]]

        # ht
        outp = t.stack(outp).transpose(0, 1)  # (batch_size, seq_len, lstm_size)
        len1 = (len_1 - 1).view(-1, 1, 1).expand(outp.size(0), 1, outp.size(2))  # (batch_size, 1, lstm_size)
        out = t.gather(outp, 1, len1).transpose(1, 0)
        # ct
        hidden_states = t.stack(hidden_states).transpose(0, 1)
        hidden_state = t.gather(hidden_states, 1, len1).transpose(1, 0)

        lstm_outs, _ = self.lstm2(sen_2, (out, hidden_state))
        lstm_outs = lstm_outs.transpose(0, 1)  # (batch_size, seq_len, lstm_size)

        # Attention
        # len2 = (len2 - 1).view(-1, 1, 1).expand(lstm_outs.size(0), 1, lstm_outs.size(2))
        # lstm_out = t.gather(lstm_outs, 1, len2)
        # lstm_out = lstm_out.view(lstm_out.size(0), -1)
        #
        # WyY = t.matmul(outp, self.Wy)
        # context_vec = context_vector(lstm_out, outp, WyY)
        # final = t.tanh(t.matmul(context_vec, self.Wp) + t.matmul(lstm_out, self.Wh))

        WyY = t.matmul(outp, self.Wy)
        # How to Init contextvec ?
        context_vec_pre = t.zeros(lstm_outs.size(0),lstm_outs.size(2)).cuda()
        context_vec = []
        for i in range(lstm_outs.size(1)):
            context_vec_pre = self.context_vector_(lstm_outs[:, i, :], outp, WyY, context_vec_pre, mask_1, batch_size, max_seq_len)
            context_vec.append(context_vec_pre)

        context_vec = t.stack(context_vec).transpose(0, 1)
        len2 = (len_2 - 1).view(-1, 1, 1).expand(lstm_outs.size(0), 1, lstm_outs.size(2))
        context_vec = t.gather(context_vec, 1, len2)
        context_vec = context_vec.view(context_vec.size(0), -1)
        lstm_out = t.gather(lstm_outs, 1, len2)
        lstm_out = lstm_out.view(lstm_out.size(0), -1)
        return t.tanh(t.matmul(context_vec, self.Wp) + t.matmul(lstm_out, self.Wh))

    # Forward propagation
    def forward(self, rep1, len1, mask1, rep2, len2, mask2):
        batch_size = rep1.size()[0]
        max_seq_len = rep1.size()[1]
        sents_premise = self.embeddings(rep1).transpose(1, 0)  # (sequence_length, batch_size, feature_size)
        sents_hypothesis = self.embeddings(rep2).transpose(1, 0)
        final_1 = self.get_representation(sents_premise, sents_hypothesis, len1, mask1, len2, batch_size, max_seq_len)
        final_2 = self.get_representation(sents_hypothesis, sents_premise, len2, mask2, len1, batch_size, max_seq_len)
        final = t.cat((final_1, final_2), dim=1)
        fc_out = F.dropout(t.tanh(self.linear1(final)), p=self.dropout_rate)
        fc_out = self.linear2(fc_out)
        out = F.log_softmax(fc_out, dim=1)
        return out

    def load(self, path):
        self.load_state_dict(t.load(path))

    def save(self, name=None):
        if name is None:
            prefix = 'checkpoints/TM' + '_'
            name = time.strftime(prefix + '%m%d_%H_%M_%S.pth')
        t.save(self.state_dict(), name)
        return name


class ModelRun(object):
    def __init__(self):
        pass

    # Get a batch of data from given data set.
    def get_batch(self, data_set, s, e):
        return data_set[0][s:e], data_set[1][s:e], data_set[2][s:e], data_set[3][s:e], data_set[4][s:e]

    # Create mask for premise sentences.
    def create_mask(self, data_set, max_length, index):
        length = data_set[index]
        masks = []
        for one in length:
            mask = list(np.ones(one))
            mask.extend(list(np.zeros(max_length - one)))
            masks.append(mask)
        masks = np.asarray(masks, dtype=np.float32)
        return masks

    # Evaluate the trained model on test set
    def evaluate_model(self, pred_Y, Y):
        # TODO: dim?
        _, idx = t.max(pred_Y, dim=1)
        if use_gpu:
            idx = idx.cpu()
            Y = Y.cpu()

        idx = idx.numpy()
        Y = Y.data.numpy()
        accuracy = np.sum(idx == Y)
        return accuracy

    # Train and evaluate SNLI models
    def train_and_evaluate(self, embedding, train_set, dev_set):
        # train_set = [sent1, len1, sent2, len2, label]
        embeddings = embedding.embeddings
        max_seq_len = train_set[0].shape[1]

        # Create mask for first sentence
        train_mask_1 = self.create_mask(train_set, max_seq_len, 1)
        train_mask_2 = self.create_mask(train_set, max_seq_len, 3)
        dev_mask_1 = self.create_mask(dev_set, max_seq_len, 1)
        dev_mask_2 = self.create_mask(dev_set, max_seq_len, 3)

        # Train, validate and test set size
        train_size = train_set[0].shape[0]
        dev_size = dev_set[0].shape[0]

        # Initialize embedding matrix
        embedding = t.nn.Embedding(embeddings.shape[0], embeddings.shape[1], padding_idx=0)
        embedding.weight = t.nn.Parameter(t.from_numpy(embeddings))
        embedding.weight.requires_grad = False
        embedding.weight[freeze_index:].requires_grad = True

        # Define models
        self.model = TwoWayWordByWordAttentionLSTM(
            lstm_size, hidden_size, dropout_rate, embedding, classes, beta
        )
        if load_model_path is not None:
            self.model.load(load_model_path)
        if use_gpu:
            self.model = self.model.cuda()

        optimizer = t.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                 lr=learning_rate, weight_decay=beta)
        criterion = t.nn.NLLLoss()  # CrossEntropyLoss()=log_softmax() + NLLLoss()
        accuracy = 0

        for i in range(epochs):
            # put model to training mode
            self.model.train()

            print('epoch', i + 1)
            start_time = time.time()
            s = 0
            loss_train = 0
            times = 0
            index = math.ceil(train_size / batch_size)
            # while s < train_size:
            for i in tqdm(range(index)):
                e = min(s + batch_size, train_size)
                batch_1v, batch_1l, batch_2v, batch_2l, batch_label = self.get_batch(train_set, s, e)
                batch_mask_1 = train_mask_1[s:e]
                batch_mask_2 = train_mask_2[s:e]
                rep1 = t.from_numpy(batch_1v).requires_grad_(False).long()
                len1 = t.from_numpy(batch_1l).requires_grad_(False).long()
                rep2 = t.from_numpy(batch_2v).requires_grad_(False).long()
                len2 = t.from_numpy(batch_1l).requires_grad_(False).long()
                mask_1 = t.from_numpy(batch_mask_1).requires_grad_(False).long()
                mask_2 = t.from_numpy(batch_mask_2).requires_grad_(False).long()
                label = t.from_numpy(batch_label).requires_grad_(False).long()
                if use_gpu:
                    rep1 = rep1.cuda()
                    len1 = len1.cuda()
                    rep2 = rep2.cuda()
                    len2 = len2.cuda()
                    mask_1 = mask_1.cuda()
                    mask_2 = mask_2.cuda()
                    label = label.cuda()
                optimizer.zero_grad()
                t.autograd.set_detect_anomaly(True)
                predict_label = self.model(rep1, len1, mask_1, rep2, len2, mask_2)
                loss = criterion(predict_label, label)
                loss_train += loss.item()
                times += 1
                # Zero gradients, perform a backward pass, and update the weights.
                loss.backward()
                optimizer.step()
                s = e
            loss_train /= times
            end_time = time.time()

            # Evaluate the trained model on valset
            self.model.eval()
            s = 0
            total_correct = 0
            while s < dev_size:
                e = min(s + batch_size, dev_size)
                batch_1v, batch_1l, batch_2v, batch_2l, batch_label = \
                    self.get_batch(dev_set, s, e)
                batch_mask_1 = dev_mask_1[s:e]
                batch_mask_2 = dev_mask_2[s:e]
                rep1 = t.from_numpy(batch_1v).requires_grad_(False).long()
                len1 = t.from_numpy(batch_1l).requires_grad_(False).long()
                rep2 = t.from_numpy(batch_2v).requires_grad_(False).long()
                len2 = t.from_numpy(batch_1l).requires_grad_(False).long()
                mask_1 = t.from_numpy(batch_mask_1).requires_grad_(False).long()
                mask_2 = t.from_numpy(batch_mask_2).requires_grad_(False).long()
                label = t.from_numpy(batch_label).requires_grad_(False).long()
                if use_gpu:
                    rep1 = rep1.cuda()
                    len1 = len1.cuda()
                    rep2 = rep2.cuda()
                    len2 = len2.cuda()
                    mask_1 = mask_1.cuda()
                    mask_2 = mask_2.cuda()
                    label = label.cuda()

                predict_label = self.model(rep1, len1, mask_1, rep2, len2, mask_2)
                total_correct += self.evaluate_model(predict_label, label)
                s = e
            print("Epoch {}\tLoss {}\tAccuracy_Val {}\tTime {}".format(i + 1, loss_train, (total_correct / dev_size),
                                                                       (end_time - start_time)))
            self.model.save()

    def test(self, test_set):
        max_seq_len = test_set[0].shape[1]
        test_mask_1 = self.create_mask(test_set, max_seq_len, 1)
        test_mask_2 = self.create_mask(test_set, max_seq_len, 3)
        test_size = test_set[0].shape[0]
        # Evaluate the trained model on testset
        s = 0
        total_correct = 0
        while s < test_size:
            e = min(s + batch_size, test_size)
            batch_1v, batch_1l, batch_2v, batch_2l, batch_label = \
                self.get_batch(test_set, s, e)
            batch_mask_1 = test_mask_1[s:e]
            batch_mask_2 = test_mask_2[s:e]
            rep1 = t.from_numpy(batch_1v).requires_grad_(False).long()
            len1 = t.from_numpy(batch_1l).requires_grad_(False).long()
            rep2 = t.from_numpy(batch_2v).requires_grad_(False).long()
            len2 = t.from_numpy(batch_1l).requires_grad_(False).long()
            mask_1 = t.from_numpy(batch_mask_1).requires_grad_(False).long()
            mask_2 = t.from_numpy(batch_mask_2).requires_grad_(False).long()
            label = t.from_numpy(batch_label).requires_grad_(False).long()
            if use_gpu:
                rep1 = rep1.cuda()
                len1 = len1.cuda()
                rep2 = rep2.cuda()
                len2 = len2.cuda()
                mask_1 = mask_1.cuda()
                mask_2 = mask_2.cuda()
                label = label.cuda()

            predict_label = self.model(rep1, len1, mask_1, rep2, len2, mask_2)
            total_correct += self.evaluate_model(predict_label, label)
            s = e
        print('Accuracy %f' % (total_correct / test_size))


def main():
    ## Preprocess
    ## collect vocabulary of SNLI set
    # snli = SNLI(None, data_source_path)
    # print('init vocab finish')
    # # read word embedding
    # embedding = WordEmbedding(embedding_source_path, snli.vocab)
    # pickle.dump(embedding, open(embedding_path, 'wb'))
    # print('init embedding finish')
    # # create SNLI dataset
    # snli = SNLI(embedding, data_source_path)
    # train_set, dev_set, test_set = snli.create_padding_set()
    # pickle.dump([train_set, dev_set, test_set], open(data_path, 'wb'))
    # print('init dataset finish')
    # Load pre-trained word embeddings and the SNLI dataset
    embedding = pickle.load(open(embedding_path, 'rb'))
    dataset = pickle.load(open(data_path, 'rb'))
    train_set = dataset[0]
    dev_set = dataset[1]
    test_set = dataset[2]
    train_size = train_set[0].shape[0]
    idx = list(range(train_size))
    idx = np.asarray(idx, dtype=np.int32)
    np.random.shuffle(idx)
    # idx = idx[0:int(idx.shape[0] * ratio)]
    sent1 = train_set[0][idx]
    len1 = train_set[1][idx]
    sent2 = train_set[2][idx]
    len2 = train_set[3][idx]
    label = train_set[4][idx]

    train_set = [sent1, len1, sent2, len2, label]

    # Train
    model = ModelRun()
    model.train_and_evaluate(embedding, train_set, dev_set)
    model.test(test_set)


if __name__ == '__main__':
    main()
    # model = TwoWayWordByWordAttentionLSTM(lstm_size, hidden_size, 0.5, None, 3, beta)
    # rep1 = t.zeros([32, 50]).long()
    # len1 = (t.ones(32) * 50).long()
    # len1[0] = 30
    # rep2 = t.zeros([32, 30]).long()
    # len2 = (t.ones(32) * 30).long()
    # len2[0] = 25
    # mask1 = t.ones_like(rep1)
    # mask1[0, 30:] = 0
    # mask2 = t.ones_like(rep2)
    # mask2[0, 25:] = 0
    # out = model(rep1, len1, mask1, rep2, len2, mask2)
