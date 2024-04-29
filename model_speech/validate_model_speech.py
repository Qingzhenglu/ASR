import torch
import torch.nn as nn
from model_speech.model_speech_cnn_ctc import DFCNN
from model_speech.speech_dataset import SpeechDataset
from torch.utils.data import DataLoader


def decode_ctc(softmax_output, num2word):
    _, decoded = torch.max(softmax_output, dim=2)
    decoded = decoded.squeeze(0).flatten()  # 去除batch维度
    decoded = decoded.tolist()  # 转换为Python列表

    # 将数字标签映射回文本
    text = [num2word[j] for j in decoded]

    return decoded, text


if __name__ == '__main__':

    batch_size = 1
    train_dataset = SpeechDataset('../data/train_data.txt')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=train_dataset.collate_fn, shuffle=True)
    test_dataset = SpeechDataset('../data/test.txt')
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=test_dataset.collate_fn, shuffle=False)

    num_classes = 1209
    print("输出类别：" + str(num_classes))

    model = DFCNN(num_classes)
    checkpoint = torch.load('../acoustic_models/model_speech_dfcnn_ctc4.pth.tar')
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.eval()

    input, target = test_dataset.__getitem__(1)
    print(target)
    input = input.unsqueeze(0)
    input = input.unsqueeze(0)
    output = model(input)
    print(output.size())

    decoded_result, text_result = decode_ctc(output, train_dataset.acoustic_model_vocab)
    print("Decoded result (numeric):", decoded_result)
    print("Decoded result (text):", text_result)
    lst = []
    for i, idx in enumerate(target):
        lst.append(train_dataset.acoustic_model_vocab[idx])
    print(lst)








