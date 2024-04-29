import numpy as np
import torch
from scipy.fftpack import fft
import scipy.io.wavfile as wav
from torch.utils.data import Dataset


# 获取信号的语谱图
def compute_fbank(file):
    x = np.linspace(0, 400 - 1, 400, dtype=np.int64)
    w = 0.54 - 0.46 * np.cos(2 * np.pi * (x) / (400 - 1))
    fs, wavsignal = wav.read(file)
    time_window = 25
    window_length = fs / 1000 * time_window
    wav_arr = np.array(wavsignal)
    wav_length = len(wavsignal)

    range0_end = int(len(wavsignal) / fs * 1000 - time_window) // 10
    data_input = np.zeros((range0_end, 200), dtype=np.float64)
    data_line = np.zeros((1, 400), dtype=np.float64)
    for i in range(0, range0_end):
        p_start = i * 160
        p_end = p_start + 400
        data_line = wav_arr[p_start:p_end]
        data_line = data_line * w
        data_line = np.abs(fft(data_line))
        data_input[i] = data_line[0:200]
    data_input = np.log(data_input + 1)
    return data_input


class AcousticDataset(Dataset):
    def __init__(self, wav_lst, pin_lst, data_path, acoustic_vocab, batch_size):
        self.wav_lst = wav_lst
        self.pin_lst = pin_lst
        self.data_path = data_path
        self.acoustic_vocab = acoustic_vocab
        self.batch_size = batch_size

    def __len__(self):
        return len(self.wav_lst)

    def __getitem__(self, index):
        fbank = compute_fbank(self.data_path + self.wav_lst[index])
        pad_fbank = np.zeros((fbank.shape[0] // 8 * 8 + 8, fbank.shape[1]))
        pad_fbank[:fbank.shape[0], :] = fbank
        label = self.pin2id(self.pin_lst[index], self.acoustic_vocab)
        label_ctc_len = self.ctc_len(label)
        if pad_fbank.shape[0] // 8 >= label_ctc_len:
            return pad_fbank, label
        else:
            return None

    def collate_fn(self, batch):
        batch = [item for item in batch if item is not None]
        wav_data_lst, label_data_lst = zip(*batch)
        pad_wav_data, input_length = self.wav_padding(wav_data_lst)
        pad_label_data, label_length = self.label_padding(label_data_lst)
        inputs = {
            'the_inputs': torch.tensor(pad_wav_data, dtype=torch.float32),
            'the_labels': torch.tensor(pad_label_data, dtype=torch.long),
            'input_length': torch.tensor(input_length, dtype=torch.long),
            'label_length': torch.tensor(label_length, dtype=torch.long),
        }
        outputs = {'ctc': torch.zeros(pad_wav_data.shape[0], dtype=torch.float32)}
        return inputs, outputs

    def pin2id(self, line, vocab):
        return [vocab.index(pin) for pin in line]

    def ctc_len(self, label):
        add_len = 0
        label_len = len(label)
        for i in range(label_len - 1):
            if label[i] == label[i + 1]:
                add_len += 1
        return label_len + add_len

    def wav_padding(self, wav_data_lst):
        wav_lens = [len(data) for data in wav_data_lst]
        wav_max_len = max(wav_lens)
        wav_lens = np.array([leng // 8 for leng in wav_lens])
        new_wav_data_lst = np.zeros((len(wav_data_lst), wav_max_len, 200, 1))
        for i in range(len(wav_data_lst)):
            new_wav_data_lst[i, :wav_data_lst[i].shape[0], :, 0] = wav_data_lst[i]
        return new_wav_data_lst, wav_lens

    def label_padding(self, label_data_lst):
        label_lens = np.array([len(label) for label in label_data_lst])
        max_label_len = max(label_lens)
        new_label_data_lst = np.zeros((len(label_data_lst), max_label_len))
        for i in range(len(label_data_lst)):
            new_label_data_lst[i][:len(label_data_lst[i])] = label_data_lst[i]
        return new_label_data_lst, label_lens


class Loader:
    def __init__(self, data_path, batch_size, data_length):
        self.data_path = data_path
        self.data_length = data_length
        self.batch_size = batch_size
        self.source_init()

    def source_init(self):
        self.wav_lst = []
        self.pin_lst = []
        self.han_lst = []

        with open('../data/train_data.txt', 'r', encoding='utf-8') as f:
            data = f.readlines()

            for line in data:
                wav_file, han, pin = line.split('\t')
                self.wav_lst.append(wav_file)
                self.pin_lst.append(pin.strip('\n').split(' '))
                self.han_lst.append(han)

            if self.data_length:
                self.wav_lst = self.wav_lst[:self.data_length]
                self.pin_lst = self.pin_lst[:self.data_length]
                self.han_lst = self.han_lst[:self.data_length]

            self.acoustic_vocab = self.acoustic_model_vocab(self.pin_lst)
            self.pin_vocab = self.language_model_pin_vocab(self.pin_lst)
            self.han_vocab = self.language_model_han_vocab(self.han_lst)

    def get_language_model_batch(self):
        batch_num = len(self.pin_lst) // self.batch_size
        for k in range(batch_num):
            begin = k * self.batch_size
            end = begin + self.batch_size
            input_batch = self.pin_lst[begin:end]
            label_batch = self.han_lst[begin:end]
            max_len = max([len(line) for line in input_batch])
            input_batch = np.array(
                [self.pin2id(line, self.pin_vocab) + [0] * (max_len - len(line)) for line in input_batch])
            label_batch = np.array(
                [self.han2id(line, self.han_vocab) + [0] * (max_len - len(line)) for line in label_batch])
            yield input_batch, label_batch

    def pin2id(self, line, vocab):
        return [vocab.index(pin) for pin in line]

    def han2id(self, line, vocab):
        return [vocab.index(han) for han in line]

    def wav_padding(self, wav_data_lst):
        wav_lens = [len(data) for data in wav_data_lst]
        wav_max_len = max(wav_lens)
        wav_lens = np.array([leng // 8 for leng in wav_lens])
        new_wav_data_lst = np.zeros((len(wav_data_lst), wav_max_len, 200, 1))
        for i in range(len(wav_data_lst)):
            new_wav_data_lst[i, :wav_data_lst[i].shape[0], :, 0] = wav_data_lst[i]
        return new_wav_data_lst, wav_lens

    def label_padding(self, label_data_lst):
        label_lens = np.array([len(label) for label in label_data_lst])
        max_label_len = max(label_lens)
        new_label_data_lst = np.zeros((len(label_data_lst), max_label_len))
        for i in range(len(label_data_lst)):
            new_label_data_lst[i][:len(label_data_lst[i])] = label_data_lst[i]
        return new_label_data_lst, label_lens

    def acoustic_model_vocab(self, data):
        vocab = []
        for line in data:
            line = line
            for pin in line:
                if pin not in vocab:
                    vocab.append(pin)
        vocab.append('_')
        return vocab

    def language_model_pin_vocab(self, data):
        vocab = ['<PAD>']
        for line in data:
            for pin in line:
                if pin not in vocab:
                    vocab.append(pin)
        return vocab

    def language_model_han_vocab(self, data):
        vocab = ['<PAD>']
        for line in data:
            line = ''.join(line.split(' '))
            for han in line:
                if han not in vocab:
                    vocab.append(han)
        return vocab

    def ctc_len(self, label):
        add_len = 0
        label_len = len(label)
        for i in range(label_len - 1):
            if label[i] == label[i + 1]:
                add_len += 1
        return label_len + add_len
