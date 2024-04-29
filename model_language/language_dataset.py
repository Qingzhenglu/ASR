import json
import numpy as np
import torch
from torch.utils.data import Dataset


def dic_build(pny2id=None, han2id=None):
    if pny2id is None:
        pny2id = {"pad": 0, "unk": 1, "go": 2, "eos": 3, "": 4}
    if han2id is None:
        han2id = {"pad": 0, "unk": 1, "go": 2, "eos": 3, "": 4}

    with open('../data/data.txt', 'r', encoding='utf-8') as f:
        data = f.readlines()
        for line in data:
            wav, hans, pin = line.split('\t')
            pny_lst = pin.strip('\n').split(' ')
            if len(pny_lst) == len(han2id):
                for pny, han in zip(pny_lst, hans):
                    if pny not in pny2id:
                        pny2id[pny] = len(pny2id)
                    if han not in han2id:
                        han2id[han] = len(han2id)
    with open("../data/pny2id.json", "w", encoding="utf-8") as f:
        json.dump(pny2id, f)
    with open("../data/han2id.json", "w", encoding="utf-8") as f:
        json.dump(han2id, f)


def seq_padding(x):
    length = [len(item) for item in x]
    max_len = max(length)
    x = [item+[0]*(max_len-len(item)) for item in x]
    return torch.tensor(np.array(x))


class LmDataset(Dataset):
    def __init__(self, pny_vocab, han_vocab, data_path="../data/train_data.txt"):
        self.pin_lst, self.han_lst = self.read_txt()
        self.pny2id = pny_vocab
        self.han2id = han_vocab
        self.data_path = data_path

    def __len__(self):
        return len(self.pin_lst)

    def __getitem__(self, idx):
        if len(self.pin_lst[idx]) == len(self.han_lst[idx]):
            pny = [self.pny2id.get(p, 1) for p in self.pin_lst[idx]]
            input = torch.tensor(pny)
            han = [self.han2id.get(h, 1) for h in self.han_lst[idx]]
            output = torch.tensor(han)
            return input, output
        else:
            return None

    def collate_fn(self, batch):
        batch = [item for item in batch if item is not None]
        pny_lst, han_lst = zip(*batch)

        inputs = seq_padding(pny_lst)
        outputs = seq_padding(han_lst)

        return inputs, outputs


    def read_txt(self):
        pin_lst, han_lst = [], []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = f.readlines()

            for line in data:
                _, han, pin = line.split('\t')
                han_lst.append(han)
                pin_lst.append(pin.strip('\n').split(' '))
        return pin_lst, han_lst



