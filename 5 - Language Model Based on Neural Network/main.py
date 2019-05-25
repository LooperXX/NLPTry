import os
import sys
import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from tqdm import tqdm
import matplotlib.pyplot as plt

filepath = './data/poetryFromTang.txt'
npzpath = './data/data.npz'
max_len = 1109
embedding_dim = 128
hidden_dim = 256
lr = 1e-3
weight_decay = 1e-4
use_gpu = True
epoch = 20
batch_size = 32
env = 'poetry'
max_gen_len = 1000  # 生成诗歌最长长度
model_path = None  # 预训练模型路径
model_prefix = 'checkpoints/'  # 模型保存路径
u_min = -0.05
u_max = 0.05

prefix_words = '细雨鱼儿出,微风燕子斜。'  # 不是诗歌的组成部分，用来控制生成诗歌的意境，即hidden_state
start_words = '闲云潭影日悠悠'  # 藏头诗
acrostic = False  # 是否是藏头诗


def pad_sequences(sequences, max_len, value):
    num_samples = len(sequences)
    data = np.ones((num_samples, max_len)) * value
    for index, sen in enumerate(sequences):
        data[index, :len(sen)] = sen
    return data


def get_data(file):
    if os.path.exists(npzpath):
        data = np.load(npzpath, allow_pickle=True)
        data, word2ix, ix2word = data['data'], data['word2ix'].item(), data['ix2word'].item()
        return data, word2ix, ix2word
    sens = []
    with open(file, 'r', encoding='utf8') as f:
        f.readline()
        sen = []
        for line in f:
            if line is '\n':
                sens.append(sen)
                sen = []
            else:
                sen.extend(line.strip())
    words = {word for sen in sens for word in sen}
    word2ix = {word: index for index, word in enumerate(words)}
    word2ix['<EOP>'] = len(word2ix)
    word2ix['<START>'] = len(word2ix)
    word2ix['</s>'] = len(word2ix)
    ix2word = {index: word for word, index in list(word2ix.items())}
    for i in range(len(sens)):
        sens[i] = ["<START>"] + sens[i] + ["<EOP>"]
    max_len = np.max([len(sen) for sen in sens])
    print(max_len)
    new_data = [[word2ix[word] for word in sen] for sen in sens]

    pad_data = pad_sequences(new_data, max_len, len(word2ix) - 1)

    # 保存成二进制文件
    np.savez(npzpath, data=pad_data, word2ix=word2ix, ix2word=ix2word)
    return pad_data, word2ix, ix2word


class PoetryModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(PoetryModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.embeddings.weight.data.uniform_(u_min, u_max)
        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim, num_layers=2)
        self.linear1 = nn.Linear(self.hidden_dim, vocab_size)

    def forward(self, input, hidden=None):
        seq_len, batch_size = input.size()
        if hidden is None:
            h_0 = input.data.new(2, batch_size, self.hidden_dim).fill_(0).float()
            c_0 = input.data.new(2, batch_size, self.hidden_dim).fill_(0).float()
            # (num_layers, batch_size, hidden_dim)
        else:
            h_0, c_0 = hidden
        if use_gpu:
            h_0 = h_0.cuda()
            c_0 = c_0.cuda()

        # size: (seq_len, batch_size, embedding_dim)
        embeds = self.embeddings(input)
        # size: (seq_len, batch_size, hidden_dim)
        output, hidden = self.lstm(embeds, (h_0, c_0))

        # size: (seq_len*batch_size,vocab_size)
        output = self.linear1(output.view(seq_len * batch_size, -1))
        return output, hidden


def generate(model, ix2word, word2ix):
    """
    给定几个词，根据这几个词接着生成一首完整的诗歌
    """
    results = list(start_words)
    start_word_len = len(start_words)
    input = t.tensor([word2ix['<START>']]).view(1, 1).long()
    if use_gpu:
        input = input.cuda()
    hidden = None

    if prefix_words:
        for word in prefix_words:
            _, hidden = model(input, hidden)
            input = input.data.new([word2ix[word]]).view(1, 1)

    for i in range(max_gen_len):
        output, hidden = model(input, hidden)
        if i < start_word_len:
            word = results[i]
            input = input.data.new([word2ix[word]]).view(1, 1)
        else:
            top_index = output.data[0].topk(1)[1][0].item()
            word = ix2word[top_index]
            results.append(word)
            input = input.data.new([top_index]).view(1, 1)
        if word == '<EOP>':
            del results[-1]
            break
    return results


def gen_acrostic(model, ix2word, word2ix):
    """
    生成藏头诗
    """
    results = []
    start_word_len = len(start_words)
    input = (t.tensor([word2ix['<START>']]).view(1, 1).long())
    if use_gpu: input = input.cuda()
    hidden = None

    index = 0
    pre_word = '<START>'

    if prefix_words:
        for word in prefix_words:
            _, hidden = model(input, hidden)
            input = (input.data.new([word2ix[word]])).view(1, 1)

    for i in range(max_gen_len):
        if pre_word in {u'。', u'！', '<START>'}:
            if index == start_word_len:
                break
            else:
                # 输入藏头词
                word = start_words[index]
                index += 1
        input = (input.data.new([word2ix[word]])).view(1, 1)
        results.append(word)
        pre_word = word

        output, hidden = model(input, hidden)
        top_index = output.data[0].topk(1)[1][0].item()
        word = ix2word[top_index]
    return results


def train(data, word2ix, ix2word):
    device = t.device('cuda') if use_gpu else t.device('cpu')
    data = t.from_numpy(data)
    dataloader = t.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=1)

    # 模型定义
    model = PoetryModel(len(word2ix), embedding_dim, hidden_dim)
    optimizer = t.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    if model_path:
        model.load_state_dict(t.load(model_path))
    model.to(device)
    for i in range(epoch):
        losses = []
        times = 0.
        for ii, data_ in tqdm(enumerate(dataloader)):
            # 训练
            data_ = data_.long().transpose(1, 0).contiguous()
            data_ = data_.to(device)
            optimizer.zero_grad()
            input_, target = data_[:-1, :], data_[1:, :]
            output, _ = model(input_)
            loss = criterion(output, target.view(-1))
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            times += 1.
        plt.plot(list(range(int(times))), losses)
        print("Epoch\t{}/{}...".format(i + 1, epoch),
              "Loss\t{:.6f}...".format(np.sum(losses) / times))
        t.save(model.state_dict(), '%s_%s.pth' % (model_prefix, i))


def gen(word2ix, ix2word):
    model = PoetryModel(len(word2ix), embedding_dim, hidden_dim)
    model.load_state_dict(t.load(model_path))
    if use_gpu:
        model.cuda()
    gen_poetry = gen_acrostic if acrostic else generate
    result = gen_poetry(model, ix2word, word2ix)
    print(''.join(result))


def get_perplexity(data, word2ix, ix2word):
    device = t.device('cuda') if use_gpu else t.device('cpu')
    data = t.from_numpy(data)
    dataloader = t.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=1)
    model = PoetryModel(len(word2ix), embedding_dim, hidden_dim)
    if model_path:
        model.load_state_dict(t.load(model_path))
    model.to(device)
    model.eval()
    Tn = 0.
    perplexity = 1.0
    for i, data_ in tqdm(enumerate(dataloader)):
        data_ = data_.long().transpose(1, 0).contiguous()
        data_ = data_.to(device)
        input_, target = data_[:-1, :], data_[1:, :]
        target = target.cpu().numpy()
        Tn += target.shape[1] * max_len
        index = list(range(target.shape[1]))
        hidden = None
        for ii in range(input_.size()[1]):
            output, hidden = model(input_[ii, :].unsqueeze(0), hidden)
            temp = output[index, target[ii, :]].detach()
            perplexity *= t.prod(temp).item() # float64 !!!
    return np.power(perplexity, - 1. / Tn)

def main():
    # t.set_default_dtype(t.float64) # RuntimeError: cuDNN error: CUDNN_STATUS_EXECUTION_FAILED
    data, word2ix, ix2word = get_data(filepath)
    # train(data, word2ix, ix2word)
    perplexity = get_perplexity(data, word2ix, ix2word)
    print(perplexity)

if __name__ == '__main__':
    main()
