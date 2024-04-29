import numpy as np
import torch
from torch.utils.data import Dataset
from feature import read_wav_data, get_spectrogram_feature


class SpeechDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.wav_lst, self.pin_lst, self.han_lst = self.read_txt()
        self.acoustic_model_vocab = self.get_vocab()

    def __len__(self):
        return len(self.wav_lst)

    def __getitem__(self, idx):
        wav_data, fs = read_wav_data('../data_thchs30/data/' + self.wav_lst[idx])
        spec_feature = get_spectrogram_feature(wav_data, fs)

        data_input = np.zeros((spec_feature.shape[0] // 8 * 8 + 8, spec_feature.shape[1]))
        data_input[:spec_feature.shape[0], :] = spec_feature
        input = torch.tensor(data_input, dtype=torch.float32)

        label = self.get_label(self.pin_lst[idx], self.acoustic_model_vocab)
        label_ctc_len = self.ctc_len(label)
        if data_input.shape[0] // 8 >= label_ctc_len:
            return input, label
        else:
            return None

    def collate_fn(self, batch):
        batch = [item for item in batch if item is not None]
        wav_data_lst, label_data_lst = zip(*batch)

        wav_lens = [len(data) for data in wav_data_lst]
        # max_wav_len = max(wav_lens)
        input_lens = np.array([leng // 8 for leng in wav_lens])
        wav_data = np.zeros((len(wav_data_lst), 1, 1600, 200))
        for i in range(len(wav_data_lst)):
            wav_data[i, 0, :wav_data_lst[i].shape[0], :] = wav_data_lst[i]

        label_lens = np.array([len(label) for label in label_data_lst])
        # max_label_len = max(label_lens)
        label_data = np.zeros((len(label_data_lst), 64))
        for i in range(len(label_data_lst)):
            label_data[i][:len(label_data_lst[i])] = label_data_lst[i]

        inputs = {
            'the_inputs': torch.tensor(wav_data, dtype=torch.float32),
            'the_labels': torch.tensor(label_data, dtype=torch.long),
            'input_length': torch.tensor(input_lens, dtype=torch.long),
            'label_length': torch.tensor(label_lens, dtype=torch.long),
        }
        outputs = {'ctc': torch.zeros(wav_data.shape[0], dtype=torch.float32)}
        return inputs, outputs

    def read_txt(self):
        wav_lst, pin_lst, han_lst = [], [], []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = f.readlines()

            for line in data:
                wav_file, han, pin = line.split('\t')
                wav_lst.append(wav_file)
                han_lst.append(han)
                pin_lst.append(pin.strip('\n').split(' '))
        return wav_lst, pin_lst, han_lst

    def get_vocab(self):
        """
        label列表
        """
        pin_lst = []
        with open('../data/data.txt', 'r', encoding='utf-8') as f:
            data = f.readlines()

            for line in data:
                wav_file, han, pin = line.split('\t')
                pin_lst.append(pin.strip('\n').split(' '))
        vocab = []
        for line in pin_lst:
            for pin in line:
                if pin not in vocab:
                    vocab.append(pin)
        vocab.append('_')
        return vocab

    def get_label(self, line, vocab):
        """
        拼音label 转为 数字label
        """
        return [vocab.index(pin) for pin in line]

    def ctc_len(self, label):
        add_len = 0
        label_len = len(label)
        for i in range(label_len - 1):
            if label[i] == label[i + 1]:
                add_len += 1
        return label_len + add_len
