from __future__ import print_function
from collections import OrderedDict
import torch
import torch.nn as nn
from torch import autograd
import time
import pickle
import urllib
import matplotlib.pyplot as plt
import os
import sys
import codecs
import re
import numpy as np
from tqdm import tqdm

plt.rcParams['figure.dpi'] = 80
plt.style.use('seaborn-pastel')

# parameters for the Model
parameters = OrderedDict()
parameters['train'] = "./data/conll2003-IOBES/eng.train.IOBES"  # Path to train file
parameters['dev'] = "./data/conll2003-IOBES/eng.testa.IOBES"  # Path to test file
parameters['test'] = "./data/conll2003-IOBES/eng.testb.IOBES"  # Path to dev file
parameters['tag_scheme'] = "BIOES"  # BIO or BIOES
parameters['lower'] = True  # Boolean variable to control lowercasing of words
parameters['zeros'] = True  # Boolean variable to control replacement of  all digits by 0
parameters['char_dim'] = 30  # Char embedding dimension
parameters['word_dim'] = 100  # Token embedding dimension
parameters['word_lstm_dim'] = 200  # Token LSTM hidden layer size
parameters['word_bidirect'] = True  # Use a bidirectional LSTM for words
# parameters['embedding_path'] = "./data/glove.6B.100d.txt" #Location of pretrained embeddings
parameters[
    'embedding_path'] = "E:\\workspace\\jupyter_notebook\\.vector_cache\\glove.6B.100d.txt"  # Location of pretrained embeddings
parameters['all_emb'] = 1  # Load all embeddings
parameters['crf'] = 1  # Use CRF (0 to disable)
parameters['dropout'] = 0.5  # Droupout on the input (0 = no dropout)
parameters['epoch'] = 50  # Number of epochs to run"
parameters['weights'] = ""  # path to Pretrained for from a previous run
parameters['name'] = "self-trained-model"  # Model name
parameters['gradient_clip'] = 5.0
parameters['char_mode'] = "CNN"
models_path = "./models/"  # path to saved models

# GPU
parameters['use_gpu'] = torch.cuda.is_available()  # GPU Check
use_gpu = parameters['use_gpu']

# parameters['reload'] = "./models/pre-trained-model"
parameters['reload'] = False

# Constants
START_TAG = '<START>'
STOP_TAG = '<STOP>'

# paths to files
mapping_file = './data/mapping.pkl'  # To stored mapping file
name = parameters['name']  # To stored model
model_name = models_path + name  # get_name(parameters)

if not os.path.exists(models_path):
    os.makedirs(models_path)


# Load data and preprocess
def zero_digits(s):
    """
    Replace every digit in a string by a zero.
    """
    return re.sub('\d', '0', s)


def load_sentences(path, zeros):
    """
    Load sentences. A line must contain at least a word and its tag.
    Sentences are separated by empty lines.
    """
    sentences = []
    sentence = []
    for line in codecs.open(path, 'r', 'utf8'):
        # zero all the digits
        line = zero_digits(line.rstrip()) if zeros else line.rstrip()
        if not line:
            if len(sentence) > 0:
                if 'DOCSTART' not in sentence[0][0]:
                    sentences.append(sentence)
                sentence = []
        else:
            word = line.split()
            assert len(word) >= 2
            sentence.append(word)
    if len(sentence) > 0:
        if 'DOCSTART' not in sentence[0][0]:
            sentences.append(sentence)
    return sentences


# Create Mappings for Words, Characters and Tags
def create_dico(item_list):
    """
    Create a dictionary of items from a list of list of items.
    """
    assert type(item_list) is list
    dico = {}
    for items in item_list:
        for item in items:
            if item not in dico:
                dico[item] = 1
            else:
                dico[item] += 1
    return dico


def create_mapping(dico):
    """
    Create a mapping (item to ID / ID to item) from a dictionary.
    Items are ordered by decreasing frequency.
    """
    sorted_items = sorted(dico.items(), key=lambda x: (-x[1], x[0]))
    id_to_item = {i: v[0] for i, v in enumerate(sorted_items)}
    item_to_id = {v: k for k, v in id_to_item.items()}
    return item_to_id, id_to_item


def word_mapping(sentences, lower):
    """
    Create a dictionary and a mapping of words, sorted by frequency.
    """
    words = [[x[0].lower() if lower else x[0] for x in s] for s in sentences]
    dico = create_dico(words)
    dico['<UNK>'] = 10000000  # UNK tag for unknown words
    word_to_id, id_to_word = create_mapping(dico)
    print("Found %i unique words (%i in total)" % (
        len(dico), sum(len(x) for x in words)
    ))
    return dico, word_to_id, id_to_word


def char_mapping(sentences):
    """
    Create a dictionary and mapping of characters, sorted by frequency.
    """
    chars = ["".join([w[0] for w in s]) for s in sentences]
    dico = create_dico(chars)
    char_to_id, id_to_char = create_mapping(dico)
    print("Found %i unique characters" % len(dico))
    return dico, char_to_id, id_to_char


def tag_mapping(sentences):
    """
    Create a dictionary and a mapping of tags, sorted by frequency.
    """
    tags = [[word[-1] for word in s] for s in sentences]
    dico = create_dico(tags)
    dico[START_TAG] = -1
    dico[STOP_TAG] = -2
    tag_to_id, id_to_tag = create_mapping(dico)
    print("Found %i unique named entity tags" % len(dico))
    return dico, tag_to_id, id_to_tag


# Preparing final dataset
def lower_case(x, lower=False):
    if lower:
        return x.lower()
    else:
        return x


def prepare_dataset(sentences, word_to_id, char_to_id, tag_to_id, lower=False):
    """
    Prepare the dataset. Return a list of lists of dictionaries containing:
        - word indexes
        - word char indexes
        - tag indexes
    """
    data = []
    for s in sentences:
        str_words = [w[0] for w in s]
        words = [word_to_id[lower_case(w, lower) if lower_case(w, lower) in word_to_id else '<UNK>']
                 for w in str_words]
        # Skip characters that are not in the training set
        chars = [[char_to_id[c] for c in w if c in char_to_id]
                 for w in str_words]
        tags = [tag_to_id[w[-1]] for w in s]
        data.append({
            'str_words': str_words,
            'words': words,
            'chars': chars,
            'tags': tags,
        })
    return data


# Load Word Embeddings
def load_wordvec(word_to_id, tag_to_id, char_to_id):
    all_word_embeds = {}
    for i, line in enumerate(codecs.open(parameters['embedding_path'], 'r', 'utf-8')):
        s = line.strip().split()
        if len(s) == parameters['word_dim'] + 1:
            all_word_embeds[s[0]] = np.array([float(i) for i in s[1:]])

    # Intializing Word Embedding Matrix
    word_embeds = np.random.uniform(-np.sqrt(0.06), np.sqrt(0.06), (len(word_to_id), parameters['word_dim']))

    for w in word_to_id:
        if w in all_word_embeds:
            word_embeds[word_to_id[w]] = all_word_embeds[w]
        elif w.lower() in all_word_embeds:
            word_embeds[word_to_id[w]] = all_word_embeds[w.lower()]

    print('Loaded %i pretrained embeddings.' % len(all_word_embeds))
    with open(mapping_file, 'wb') as f:
        mappings = {
            'word_to_id': word_to_id,
            'tag_to_id': tag_to_id,
            'char_to_id': char_to_id,
            'parameters': parameters,
            'word_embeds': word_embeds
        }
        pickle.dump(mappings, f)

    print('word_to_id: ', len(word_to_id))
    return word_embeds


# Initialization of weights
def init_embedding(input_embedding):
    """
    Initialize embedding
    """
    bias = np.sqrt(3.0 / input_embedding.size(1))
    nn.init.uniform_(input_embedding, -bias, bias)


def init_linear(input_linear):
    """
    Initialize linear transformation
    """
    bias = np.sqrt(6.0 / (input_linear.weight.size(0) + input_linear.weight.size(1)))
    nn.init.uniform_(input_linear.weight, -bias, bias)
    if input_linear.bias is not None:
        input_linear.bias.data.zero_()


def init_lstm(input_lstm):
    """
    Initialize lstm

    PyTorch weights parameters:

        weight_ih_l[k]: the learnable input-hidden weights of the k-th layer,
            of shape `(hidden_size * input_size)` for `k = 0`. Otherwise, the shape is
            `(hidden_size * hidden_size)`

        weight_hh_l[k]: the learnable hidden-hidden weights of the k-th layer,
            of shape `(hidden_size * hidden_size)`
    """

    # Weights init for forward layer
    for ind in range(0, input_lstm.num_layers):
        ## Gets the weights Tensor from our model, for the input-hidden weights in our current layer
        weight = eval('input_lstm.weight_ih_l' + str(ind))

        # Initialize the sampling range
        sampling_range = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))

        # Randomly sample from our samping range using uniform distribution and apply it to our current layer
        nn.init.uniform_(weight, -sampling_range, sampling_range)

        # Similar to above but for the hidden-hidden weights of the current layer
        weight = eval('input_lstm.weight_hh_l' + str(ind))
        sampling_range = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
        nn.init.uniform_(weight, -sampling_range, sampling_range)

    # We do the above again, for the backward layer if we are using a bi-directional LSTM (our final model uses this)
    if input_lstm.bidirectional:
        for ind in range(0, input_lstm.num_layers):
            weight = eval('input_lstm.weight_ih_l' + str(ind) + '_reverse')
            sampling_range = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
            nn.init.uniform_(weight, -sampling_range, sampling_range)
            weight = eval('input_lstm.weight_hh_l' + str(ind) + '_reverse')
            sampling_range = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
            nn.init.uniform_(weight, -sampling_range, sampling_range)

    # Bias initialization steps

    # We initialize them to zero except for the forget gate bias, which is initialized to 1
    if input_lstm.bias:
        for ind in range(0, input_lstm.num_layers):
            bias = eval('input_lstm.bias_ih_l' + str(ind))

            # Initializing to zero
            bias.data.zero_()

            # This is the range of indices for our forget gates for each LSTM cell
            bias.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1

            # Similar for the hidden-hidden layer
            bias = eval('input_lstm.bias_hh_l' + str(ind))
            bias.data.zero_()
            bias.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1

        # Similar to above, we do for backward layer if we are using a bi-directional LSTM
        if input_lstm.bidirectional:
            for ind in range(0, input_lstm.num_layers):
                bias = eval('input_lstm.bias_ih_l' + str(ind) + '_reverse')
                bias.data.zero_()
                bias.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1
                bias = eval('input_lstm.bias_hh_l' + str(ind) + '_reverse')
                bias.data.zero_()
                bias.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1


# Helper Functions
def log_sum_exp(vec):
    '''
    This function calculates the score explained above for the forward algorithm
    vec 2D: 1 * tagset_size
    '''
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


def argmax(vec):
    '''
    This function returns the max index in a vector
    '''
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)


def to_scalar(var):
    '''
    Function to convert pytorch tensor to a scalar
    '''
    return var.view(-1).data.tolist()[0]


class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim,
                 char_to_ix=None, pre_word_embeds=None, char_out_dimension=25, char_embedding_dim=25, use_gpu=False
                 , use_crf=True, char_mode='CNN'):
        '''
        Input parameters:

                vocab_size= Size of vocabulary (int)
                tag_to_ix = Dictionary that maps NER tags to indices
                embedding_dim = Dimension of word embeddings (int)
                hidden_dim = The hidden dimension of the LSTM layer (int)
                char_to_ix = Dictionary that maps characters to indices
                pre_word_embeds = Numpy array which provides mapping from word embeddings to word indices
                char_out_dimension = Output dimension from the CNN encoder for character
                char_embedding_dim = Dimension of the character embeddings
                use_gpu = defines availability of GPU,
                    when True: CUDA function calls are made
                    else: Normal CPU function calls are made
                use_crf = parameter which decides if you want to use the CRF layer for output decoding
        '''

        super(BiLSTM_CRF, self).__init__()

        # parameter initialization for the model
        self.use_gpu = use_gpu
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.use_crf = use_crf
        self.tagset_size = len(tag_to_ix)
        self.out_channels = char_out_dimension
        self.char_mode = char_mode

        if char_embedding_dim is not None:
            self.char_embedding_dim = char_embedding_dim

            # Initializing the character embedding layer
            self.char_embeds = nn.Embedding(len(char_to_ix), char_embedding_dim)
            init_embedding(self.char_embeds.weight)

            # Performing LSTM encoding on the character embeddings
            # if self.char_mode == 'LSTM':
            #     self.char_lstm = nn.LSTM(char_embedding_dim, char_lstm_dim, num_layers=1, bidirectional=True)
            #     init_lstm(self.char_lstm)

            # Performing CNN encoding on the character embeddings
            if self.char_mode == 'CNN':
                self.char_cnn3 = nn.Conv2d(in_channels=1, out_channels=self.out_channels,
                                           kernel_size=(3, char_embedding_dim), padding=(2, 0))

        # Creating Embedding layer with dimension of ( number of words * dimension of each word)
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        if pre_word_embeds is not None:
            # Initializes the word embeddings with pretrained word embeddings
            self.pre_word_embeds = True
            self.word_embeds.weight = nn.Parameter(torch.FloatTensor(pre_word_embeds))
        else:
            self.pre_word_embeds = False

        # Initializing the dropout layer, with dropout specificed in parameters
        self.dropout = nn.Dropout(parameters['dropout'])

        # Lstm Layer:
        # input dimension: word embedding dimension + character level representation
        # bidirectional=True, specifies that we are using the bidirectional LSTM
        # if self.char_mode == 'LSTM':
        #     self.lstm = nn.LSTM(embedding_dim + char_lstm_dim * 2, hidden_dim, bidirectional=True)
        if self.char_mode == 'CNN':
            self.lstm = nn.LSTM(embedding_dim + self.out_channels, hidden_dim, bidirectional=True)

        # Initializing the lstm layer using predefined function for initialization
        init_lstm(self.lstm)

        # Linear layer which maps the output of the bidirectional LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim * 2, self.tagset_size)

        # Initializing the linear layer using predefined function for initialization
        init_linear(self.hidden2tag)

        if self.use_crf:
            # Matrix of transition parameters.  Entry i,j is the score of transitioning *to* i *from* j.
            # Matrix has a dimension of (total number of tags * total number of tags)
            self.transitions = nn.Parameter(
                torch.zeros(self.tagset_size, self.tagset_size))

            # These two statements enforce the constraint that we never transfer
            # to the start tag and we never transfer from the stop tag
            self.transitions.data[tag_to_ix[START_TAG], :] = -10000
            self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

    # Helper function to calculate score
    def score_sentences(self, feats, tags):
        # tags is ground_truth, a list of ints, length is len(sentence)
        # feats is a 2D tensor, len(sentence) * tagset_size
        r = torch.LongTensor(range(feats.size()[0]))
        if self.use_gpu:
            r = r.cuda()
            pad_start_tags = torch.cat([torch.cuda.LongTensor([self.tag_to_ix[START_TAG]]), tags])
            pad_stop_tags = torch.cat([tags, torch.cuda.LongTensor([self.tag_to_ix[STOP_TAG]])])
        else:
            pad_start_tags = torch.cat([torch.LongTensor([self.tag_to_ix[START_TAG]]), tags])
            pad_stop_tags = torch.cat([tags, torch.LongTensor([self.tag_to_ix[STOP_TAG]])])

        score = torch.sum(self.transitions[pad_stop_tags, pad_start_tags]) + torch.sum(feats[r, tags])

        return score

    # Implementation of Forward Algorithm
    def forward_alg(self, feats):
        '''
        This function performs the forward algorithm explained above
        '''
        # calculate in log domain
        # feats is len(sentence) * tagset_size
        # initialize alpha with a Tensor with values all equal to -10000.

        # Do the forward algorithm to compute the partition function
        init_alphas = torch.Tensor(1, self.tagset_size).fill_(-10000.)

        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas.requires_grad_(True)
        if self.use_gpu:
            forward_var = forward_var.cuda()

        # Iterate through the sentence
        for feat in feats:
            # broadcast the emission score: it is the same regardless of
            # the previous tag
            emit_score = feat.view(-1, 1)

            # the ith entry of trans_score is the score of transitioning to
            # next_tag from i
            tag_var = forward_var + self.transitions + emit_score

            # The ith entry of next_tag_var is the value for the
            # edge (i -> next_tag) before we do log-sum-exp
            max_tag_var, _ = torch.max(tag_var, dim=1)

            # The forward variable for this tag is log-sum-exp of all the
            # scores.
            tag_var = tag_var - max_tag_var.view(-1, 1)

            # Compute log sum exp in a numerically stable way for the forward algorithm
            forward_var = max_tag_var + torch.log(torch.sum(torch.exp(tag_var), dim=1)).view(1, -1)  # ).view(1, -1)
        terminal_var = (forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]).view(1, -1)
        alpha = log_sum_exp(terminal_var)
        # Z(x)
        return alpha

    # Implementation of Viterbi Algorithm
    def viterbi_algo(self, feats):
        '''
        In this function, we implement the viterbi algorithm explained above.
        A Dynamic programming based approach to find the best tag sequence
        '''
        backpointers = []
        # analogous to forward

        # Initialize the viterbi variables in log space
        init_vvars = torch.Tensor(1, self.tagset_size).fill_(-10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        if self.use_gpu:
            forward_var = forward_var.cuda()
        for feat in feats:
            next_tag_var = forward_var.view(1, -1).expand(self.tagset_size, self.tagset_size) + self.transitions
            _, bptrs_t = torch.max(next_tag_var, dim=1)
            bptrs_t = bptrs_t.squeeze().data.cpu().numpy()  # holds the backpointers for this step
            next_tag_var = next_tag_var.data.cpu().numpy()
            viterbivars_t = next_tag_var[range(len(bptrs_t)), bptrs_t]  # holds the viterbi variables for this step
            viterbivars_t = torch.FloatTensor(viterbivars_t)
            if self.use_gpu:
                viterbivars_t = viterbivars_t.cuda()

            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = viterbivars_t + feat
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        terminal_var.data[self.tag_to_ix[STOP_TAG]] = -10000.
        terminal_var.data[self.tag_to_ix[START_TAG]] = -10000.
        best_tag_id = argmax(terminal_var.unsqueeze(0))
        path_score = terminal_var[best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)

        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def forward(self, sentence, chars, chars2_length, d):
        '''
        The function calls viterbi decode and generates the
        most probable sequence of tags for the sentence
        '''

        # Get the emission scores from the BiLSTM
        feats = self.get_lstm_features(sentence, chars, chars2_length, d)
        # viterbi to get tag_seq

        # Find the best path, given the features.
        if self.use_crf:
            score, tag_seq = self.viterbi_algo(feats)
        else:
            score, tag_seq = torch.max(feats, 1)
            tag_seq = list(tag_seq.cpu().data)

        return score, tag_seq

    # Details fo the Model
    def get_lstm_features(self, sentence, chars2, chars2_length, d):
        if self.char_mode == 'LSTM':

            chars_embeds = self.char_embeds(chars2).transpose(0, 1)

            packed = torch.nn.utils.rnn.pack_padded_sequence(chars_embeds, chars2_length)

            lstm_out, _ = self.char_lstm(packed)

            outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(lstm_out)

            outputs = outputs.transpose(0, 1)

            chars_embeds_temp = torch.FloatTensor(torch.zeros((outputs.size(0), outputs.size(2))))

            if self.use_gpu:
                chars_embeds_temp = chars_embeds_temp.cuda()

            for i, index in enumerate(output_lengths):
                chars_embeds_temp[i] = torch.cat(
                    (outputs[i, index - 1, :self.char_lstm_dim], outputs[i, 0, self.char_lstm_dim:]))

            chars_embeds = chars_embeds_temp.clone()

            for i in range(chars_embeds.size(0)):
                chars_embeds[d[i]] = chars_embeds_temp[i]

        if self.char_mode == 'CNN':
            chars_embeds = self.char_embeds(chars2).unsqueeze(1)

            ## Creating Character level representation using Convolutional Neural Netowrk
            ## followed by a Maxpooling Layer
            chars_cnn_out3 = self.char_cnn3(chars_embeds)
            chars_embeds = nn.functional.max_pool2d(chars_cnn_out3, kernel_size=(chars_cnn_out3.size(2), 1)).view(
                chars_cnn_out3.size(0), self.out_channels)

            ## Loading word embeddings
        embeds = self.word_embeds(sentence)

        ## We concatenate the word embeddings and the character level representation
        ## to create unified representation for each word
        embeds = torch.cat((embeds, chars_embeds), 1)

        embeds = embeds.unsqueeze(1)

        ## Dropout on the unified embeddings
        embeds = self.dropout(embeds)

        ## Word lstm
        ## Takes words as input and generates a output at each step
        lstm_out, _ = self.lstm(embeds)

        ## Reshaping the outputs from the lstm layer
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim * 2)

        ## Dropout on the lstm output
        lstm_out = self.dropout(lstm_out)

        ## Linear layer converts the ouput vectors to tag space
        lstm_feats = self.hidden2tag(lstm_out)

        return lstm_feats

    def get_neg_log_likelihood(self, sentence, tags, chars2, chars2_length, d):
        # sentence, tags is a list of ints
        # features is a 2D tensor, len(sentence) * self.tagset_size
        feats = self.get_lstm_features(sentence, chars2, chars2_length, d)

        if self.use_crf:
            forward_score = self.forward_alg(feats)
            gold_score = self.score_sentences(feats, tags)
            return forward_score - gold_score
        else:
            scores = nn.functional.cross_entropy(feats, tags)
            return scores


# Helper functions for evaluation
def get_chunk_type(tok, idx_to_tag):
    """
    The function takes in a chunk ("B-PER") and then splits it into the tag (PER) and its class (B)
    as defined in BIOES

    Args:
        tok: id of token, ex 4
        idx_to_tag: dictionary {4: "B-PER", ...}

    Returns:
        tuple: "B", "PER"

    """

    tag_name = idx_to_tag[tok]
    tag_class = tag_name.split('-')[0]
    tag_type = tag_name.split('-')[-1]
    return tag_class, tag_type


def get_chunks(seq, tags):
    """Given a sequence of tags, group entities and their position

    Args:
        seq: [4, 4, 0, 0, ...] sequence of labels
        tags: dict["O"] = 4

    Returns:
        list of (chunk_type, chunk_start, chunk_end)

    Example:
        seq = [4, 5, 0, 3]
        tags = {"B-PER": 4, "I-PER": 5, "B-LOC": 3}
        result = [("PER", 0, 2), ("LOC", 3, 4)]

    """

    # We assume by default the tags lie outside a named entity
    default = tags["O"]

    idx_to_tag = {idx: tag for tag, idx in tags.items()}

    chunks = []

    chunk_type, chunk_start = None, None
    for i, tok in enumerate(seq):
        # End of a chunk 1
        if tok == default and chunk_type is not None:
            # Add a chunk.
            chunk = (chunk_type, chunk_start, i)
            chunks.append(chunk)
            chunk_type, chunk_start = None, None

        # End of a chunk + start of a chunk!
        elif tok != default:
            tok_chunk_class, tok_chunk_type = get_chunk_type(tok, idx_to_tag)
            if chunk_type is None:
                # Initialize chunk for each entity
                chunk_type, chunk_start = tok_chunk_type, i
            elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
                # If chunk class is B, i.e., its a beginning of a new named entity
                # or, if the chunk type is different from the previous one, then we
                # start labelling it as a new entity
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
                chunk_type, chunk_start = tok_chunk_type, i
        else:
            pass

    # end condition
    if chunk_type is not None:
        chunk = (chunk_type, chunk_start, len(seq))
        chunks.append(chunk)

    return chunks


def evaluating(model, datas, best_F, tag_to_id, dataset="Train"):
    '''
    The function takes as input the model, data and calcuates F-1 Score
    It performs conditional updates
     1) Flag to save the model
     2) Best F-1 score
    ,if the F-1 score calculated improves on the previous F-1 score
    '''
    # Initializations
    prediction = []  # A list that stores predicted tags
    save = False  # Flag that tells us if the model needs to be saved
    new_F = 0.0  # Variable to store the current F1-Score (may not be the best)
    correct_preds, total_correct, total_preds = 0., 0., 0.  # Count variables

    for data in datas:
        ground_truth_id = data['tags']
        words = data['str_words']
        chars2 = data['chars']

        if parameters['char_mode'] == 'LSTM':
            chars2_sorted = sorted(chars2, key=lambda p: len(p), reverse=True)
            d = {}
            for i, ci in enumerate(chars2):
                for j, cj in enumerate(chars2_sorted):
                    if ci == cj and not j in d and not i in d.values():
                        d[j] = i
                        continue
            chars2_length = [len(c) for c in chars2_sorted]
            char_maxl = max(chars2_length)
            chars2_mask = np.zeros((len(chars2_sorted), char_maxl), dtype='int')
            for i, c in enumerate(chars2_sorted):
                chars2_mask[i, :chars2_length[i]] = c
            chars2_mask = torch.LongTensor(chars2_mask)

        if parameters['char_mode'] == 'CNN':
            d = {}

            # Padding the each word to max word size of that sentence
            chars2_length = [len(c) for c in chars2]
            char_maxl = max(chars2_length)
            chars2_mask = np.zeros((len(chars2_length), char_maxl), dtype='int')
            for i, c in enumerate(chars2):
                chars2_mask[i, :chars2_length[i]] = c
            chars2_mask = torch.LongTensor(chars2_mask)

        dwords = torch.LongTensor(data['words'])

        # We are getting the predicted output from our model
        if use_gpu:
            val, out = model(dwords.cuda(), chars2_mask.cuda(), chars2_length, d)
        else:
            val, out = model(dwords, chars2_mask, chars2_length, d)
        predicted_id = out

        # We use the get chunks function defined above to get the true chunks
        # and the predicted chunks from true labels and predicted labels respectively
        lab_chunks = set(get_chunks(ground_truth_id, tag_to_id))
        lab_pred_chunks = set(get_chunks(predicted_id, tag_to_id))

        # Updating the count variables
        correct_preds += len(lab_chunks & lab_pred_chunks)
        total_preds += len(lab_pred_chunks)
        total_correct += len(lab_chunks)

    # Calculating the F1-Score
    p = correct_preds / total_preds if correct_preds > 0 else 0
    r = correct_preds / total_correct if correct_preds > 0 else 0
    new_F = 2 * p * r / (p + r) if correct_preds > 0 else 0

    print("{}: new_F: {} best_F: {} ".format(dataset, new_F, best_F))

    # If our current F1-Score is better than the previous best, we update the best
    # to current F1 and we set the flag to indicate that we need to checkpoint this model

    if new_F > best_F:
        best_F = new_F
        save = True

    return best_F, new_F, save


# Helper function for performing Learning rate decay
def adjust_learning_rate(optimizer, lr):
    """
    shrink learning rate
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# Training Step

def main():
    # Load data and preprocess
    train_sentences = load_sentences(parameters['train'], parameters['zeros'])
    test_sentences = load_sentences(parameters['test'], parameters['zeros'])
    dev_sentences = load_sentences(parameters['dev'], parameters['zeros'])
    # Create Mappings for Words, Characters and Tags
    dico_words, word_to_id, id_to_word = word_mapping(train_sentences, parameters['lower'])
    dico_chars, char_to_id, id_to_char = char_mapping(train_sentences)
    dico_tags, tag_to_id, id_to_tag = tag_mapping(train_sentences)
    # Preparing final dataset
    train_data = prepare_dataset(train_sentences, word_to_id, char_to_id, tag_to_id, parameters['lower'])
    dev_data = prepare_dataset(dev_sentences, word_to_id, char_to_id, tag_to_id, parameters['lower'])
    test_data = prepare_dataset(test_sentences, word_to_id, char_to_id, tag_to_id, parameters['lower'])
    print("{} / {} / {} sentences in train / dev / test.".format(len(train_data), len(dev_data), len(test_data)))
    # Load Word Embeddings
    # word_embeds = load_wordvec(word_to_id, tag_to_id, char_to_id)
    with open(mapping_file, 'rb') as f:
        mappings = pickle.load(f)
        # word_to_id = mappings.word_to_id
        # tag_to_id = mappings.tag_to_id
        # char_to_id = mappings.char_to_id
        word_embeds = mappings['word_embeds']
    model = BiLSTM_CRF(vocab_size=len(word_to_id),
                       tag_to_ix=tag_to_id,
                       embedding_dim=parameters['word_dim'],
                       hidden_dim=parameters['word_lstm_dim'],
                       use_gpu=use_gpu,
                       char_to_ix=char_to_id,
                       pre_word_embeds=word_embeds,
                       use_crf=parameters['crf'],
                       char_mode=parameters['char_mode'])
    print("Model Initialized!!!")

    if parameters['reload']:
        if not os.path.exists(parameters['reload']):
            print("downloading pre-trained model")
            model_url = "https://github.com/TheAnig/NER-LSTM-CNN-Pytorch/raw/master/trained-model-cpu"
            urllib.request.urlretrieve(model_url, parameters['reload'])
        model.load_state_dict(torch.load(parameters['reload']))
        print("model reloaded :", parameters['reload'])

    if use_gpu:
        model.cuda()

    # Initializing the optimizer
    # The best results in the paper where achived using stochastic gradient descent (SGD)
    # learning rate=0.015 and momentum=0.9
    # decay_rate=0.05

    learning_rate = 0.015
    momentum = 0.9
    number_of_epochs = parameters['epoch']
    decay_rate = 0.05
    gradient_clip = parameters['gradient_clip']
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    # variables which will used in training process
    losses = []  # list to store all losses
    loss = 0.0  # Loss Initializatoin
    best_dev_F = -1.0  # Current best F-1 Score on Dev Set
    best_test_F = -1.0  # Current best F-1 Score on Test Set
    best_train_F = -1.0  # Current best F-1 Score on Train Set
    all_F = [[0, 0, 0]]  # List storing all the F-1 Scores
    eval_every = len(train_data)  # Calculate F-1 Score after this many iterations
    plot_every = 2000  # Store loss after this many iterations
    count = 0  # Counts the number of iterations

    if not parameters['reload']:
        tr = time.time()
        model.train(True)
        for epoch in range(1, number_of_epochs):
            for i, index in tqdm(enumerate(np.random.permutation(len(train_data)))):
                count += 1
                data = train_data[index]

                ##gradient updates for each data entry
                model.zero_grad()

                sentence_in = data['words']
                sentence_in = torch.LongTensor(sentence_in)
                tags = data['tags']
                chars2 = data['chars']

                if parameters['char_mode'] == 'LSTM':
                    chars2_sorted = sorted(chars2, key=lambda p: len(p), reverse=True)
                    d = {}
                    for i, ci in enumerate(chars2):
                        for j, cj in enumerate(chars2_sorted):
                            if ci == cj and not j in d and not i in d.values():
                                d[j] = i
                                continue
                    chars2_length = [len(c) for c in chars2_sorted]
                    char_maxl = max(chars2_length)
                    chars2_mask = np.zeros((len(chars2_sorted), char_maxl), dtype='int')
                    for i, c in enumerate(chars2_sorted):
                        chars2_mask[i, :chars2_length[i]] = c
                    chars2_mask = torch.LongTensor(chars2_mask)

                if parameters['char_mode'] == 'CNN':

                    d = {}

                    ## Padding the each word to max word size of that sentence
                    chars2_length = [len(c) for c in chars2]
                    char_maxl = max(chars2_length)
                    chars2_mask = np.zeros((len(chars2_length), char_maxl), dtype='int')
                    for i, c in enumerate(chars2):
                        chars2_mask[i, :chars2_length[i]] = c
                    chars2_mask = torch.LongTensor(chars2_mask)

                targets = torch.LongTensor(tags)

                # we calculate the negative log-likelihood for the predicted tags using the predefined function
                if use_gpu:
                    neg_log_likelihood = model.get_neg_log_likelihood(sentence_in.cuda(), targets.cuda(),
                                                                      chars2_mask.cuda(), chars2_length, d)
                else:
                    neg_log_likelihood = model.get_neg_log_likelihood(sentence_in, targets, chars2_mask, chars2_length,
                                                                      d)
                loss += neg_log_likelihood.item() / len(data['words'])
                neg_log_likelihood.backward()

                # we use gradient clipping to avoid exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                optimizer.step()

                # Storing loss
                if count % plot_every == 0:
                    loss /= plot_every
                    print(count, ': ', loss)
                    if losses == []:
                        losses.append(loss)
                    losses.append(loss)
                    loss = 0.0

                # Evaluating on Train, Test, Dev Sets
                if count % (eval_every) == 0 and count > (eval_every * 20) or \
                        count % (eval_every * 4) == 0 and count < (eval_every * 20):
                    model.train(False)
                    best_train_F, new_train_F, _ = evaluating(model, train_data, best_train_F, tag_to_id, "Train")
                    best_dev_F, new_dev_F, save = evaluating(model, dev_data, best_dev_F, tag_to_id, "Dev")
                    if save:
                        print("Saving Model to ", model_name)
                        torch.save(model.state_dict(), model_name)
                    best_test_F, new_test_F, _ = evaluating(model, test_data, best_test_F, tag_to_id, "Test")

                    all_F.append([new_train_F, new_dev_F, new_test_F])
                    model.train(True)

                # Performing decay on the learning rate
                if count % len(train_data) == 0:
                    adjust_learning_rate(optimizer, lr=learning_rate / (1 + decay_rate * count / len(train_data)))

        print(time.time() - tr)
        plt.plot(losses)
        plt.show()

    if not parameters['reload']:
        # reload the best model saved from training
        model.load_state_dict(torch.load(model_name))

    # Model Testing
    model_testing_sentences = ['Jay is from India', 'Donald is the president of USA']

    # parameters
    lower = parameters['lower']

    # preprocessing
    final_test_data = []
    for sentence in model_testing_sentences:
        s = sentence.split()
        str_words = [w for w in s]
        words = [word_to_id[lower_case(w, lower) if lower_case(w, lower) in word_to_id else '<UNK>'] for w in str_words]

        # Skip characters that are not in the training set
        chars = [[char_to_id[c] for c in w if c in char_to_id] for w in str_words]

        final_test_data.append({
            'str_words': str_words,
            'words': words,
            'chars': chars,
        })

    # prediction
    predictions = []
    print("Prediction:")
    print("word : tag")
    for data in final_test_data:
        words = data['str_words']
        chars2 = data['chars']

        d = {}

        # Padding the each word to max word size of that sentence
        chars2_length = [len(c) for c in chars2]
        char_maxl = max(chars2_length)
        chars2_mask = np.zeros((len(chars2_length), char_maxl), dtype='int')
        for i, c in enumerate(chars2):
            chars2_mask[i, :chars2_length[i]] = c
        chars2_mask = torch.LongTensor(chars2_mask)

        dwords = torch.LongTensor(data['words'])

        # We are getting the predicted output from our model
        if use_gpu:
            val, predicted_id = model(dwords.cuda(), chars2_mask.cuda(), chars2_length, d)
        else:
            val, predicted_id = model(dwords, chars2_mask, chars2_length, d)

        pred_chunks = get_chunks(predicted_id, tag_to_id)
        temp_list_tags = ['NA'] * len(words)
        for p in pred_chunks:
            temp_list_tags[p[1]] = p[0]

        for word, tag in zip(words, temp_list_tags):
            print(word, ':', tag)
        print('\n')


if __name__ == '__main__':
    main()
