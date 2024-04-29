import shutil
import time
import torch

import torch.optim as optim

from torch.nn import CTCLoss
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from model_speech.model_speech_cnn_ctc import DFCNN
from model_speech.speech_dataset import SpeechDataset


def save_checkpoint(state, is_best, filename='checkpoint_4.pth.tar'):
    """
    保存模型策略： 根据is_best，保存valid acc 最好的模型
    """
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best_' + filename)


def train(train_dataloader, model, loss_fn, optimizer, epoch, writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # 训练模式
    model.train()

    end = time.time()
    for i, (inputs, outputs) in enumerate(train_dataloader):
        # 计算数据加载时间
        data_time.update(time.time() - end)

        input = inputs['the_inputs'].cuda()
        targets = inputs['the_labels'].cuda()
        input_lengths = torch.full((input.size(0),), 200, dtype=torch.long)

        # 预测label , 计算loss
        # (T, N, C):T 是输入序列的长度，N 是批次大小，C 是类别数（包括空白标签）
        log_probs = model(input).log_softmax(2).permute(1, 0, 2).requires_grad_()

        # 计算CTCLoss
        loss = loss_fn(log_probs, targets, input_lengths, inputs['label_length'])
        # 记录loss
        losses.update(loss.item(), input.size(0))
        optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 反向传播，计算当前梯度
        optimizer.step()  # 根据梯度更新网络参数

        # 计算一个阶段经历的时间
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                epoch + 1, i, len(train_dataloader), batch_time=batch_time,
                data_time=data_time, loss=losses))
    writer.add_scalar('loss/train_loss', losses.val, global_step=epoch)


def validate(dev_dataloader, model, loss_fn, epoch, writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # 训练模式
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (inputs, outputs) in enumerate(dev_dataloader):
            # 计算数据加载时间
            data_time.update(time.time() - end)

            input = inputs['the_inputs'].cuda()
            targets = inputs['the_labels'].cuda()
            # 预测label , 计算loss (T, N, C):T 是输入序列的长度，N 是批次大小，C 是类别数（包括空白标签）
            log_probs = model(input).log_softmax(2).permute(1, 0, 2).requires_grad_()
            input_lengths = torch.full((input.size(0),), 200, dtype=torch.long)
            # 计算CTCLoss
            loss = loss_fn(log_probs, targets, input_lengths, inputs['label_length'])
            # 记录loss
            losses.update(loss.item(), input.size(0))

            # 计算一个阶段经历的时间
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0:
                print('Test-{0}: [{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                    epoch + 1, i, len(dev_dataloader),
                    batch_time=batch_time,
                    loss=losses))
    writer.add_scalar('loss/valid_loss', losses.val, global_step=epoch)


class AverageMeter(object):
    """
        计算单个变量的算术平均值
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    batch_size = 20
    epochs = 10
    learning_rate = 0.0001

    train_dataset = SpeechDataset('../data/dev_data.txt')
    dev_dataset = SpeechDataset('../data/test.txt')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=train_dataset.collate_fn, shuffle=True)
    dev_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=dev_dataset.collate_fn, shuffle=False)

    num_classes = len(train_dataset.acoustic_model_vocab)
    print("输出类别:" + str(num_classes))

    model = DFCNN(num_classes).cuda()
    # 定义损失函数
    ctc_loss = CTCLoss(blank=num_classes - 1, reduction='mean')
    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=10e-8, weight_decay=0)

    writer = SummaryWriter('../runs/dfcnn')
    print("训练次数:" + str(epochs))
    for epoch in range(epochs):
        train(train_loader, model, ctc_loss, optimizer, epoch, writer)
        validate(dev_loader, model, ctc_loss, epoch, writer)

    print("\n训练完成，保存模型")
    torch.save({
        'epoch': epochs,
        'arch': 'dfcnn+ctc',
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, '../acoustic_models/model_speech_dfcnn_ctc4.pth.tar')

    writer.close()

