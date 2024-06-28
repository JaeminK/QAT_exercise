import torch
import torch.nn as nn
from torch import optim
import torchvision
import torchvision.transforms as transforms

from model import CNN, QCNN
from dataloader import load_data

class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, fmt='%.4f'):
        self.fmt = fmt
        self.val = self.avg = self.sum = self.count = 0

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        s = self.fmt % self.avg
        return s

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def main():
    batch_size = 256
    learning_rate = 1e-3
    num_epochs = 10
    do_pretraining = True
    
    train_loader, test_loader = load_data('cifar10', batch_size)
    model = CNN().cuda()  # CNN 모델
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    if do_pretraining:
        for epoch in range(num_epochs):
            model.train()
            top1 = AverageMeter()
            losses = AverageMeter()
            for i, (images, labels) in enumerate(train_loader):
                images, labels = images.cuda(), labels.cuda()
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                acc = accuracy(outputs, labels, topk=(1,))
                top1.update(acc.item(), images.size(0))
                losses.update(loss.item(), images.size(0))
                if i % 100 == 99:
                    print(f'[Epoch {epoch + 1}, Batch {i + 1}] acc: {top1.avg:.4f} loss: {losses.avg:.3f}')
            
            with torch.no_grad():
                model.eval()
                top1 = AverageMeter()
                losses = AverageMeter()
                for i, (images, labels) in enumerate(test_loader):
                    images, labels = images.cuda(), labels.cuda()
                    outputs = model(images)
                    acc = accuracy(outputs, labels, topk=(1,))
                    top1.update(acc.item(), images.size(0))
                    losses.update(loss.item(), images.size(0))
                    
                    running_loss += loss.item()
                    if i % 100 == 99:
                        print(f'[Epoch {epoch + 1}, Batch {i + 1}] acc: {top1.avg:.4f} loss: {losses.avg:.3f}')
    else:
        model.load_state_dict(torch.load('cnn_model_weights.pth'))
        
        
    model_q = QCNN(model).cuda()  # QCNN 모델
    optimizer = optim.SGD(model_q.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model_q.train()
        top1 = AverageMeter()
        losses = AverageMeter()
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = model_q(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            acc = accuracy(outputs, labels, topk=(1,))
            top1.update(acc.item(), images.size(0))
            losses.update(loss.item(), images.size(0))
            if i % 100 == 99:
                print(f'[Epoch {epoch + 1}, Batch {i + 1}] acc: {top1.avg:.4f} loss: {losses.avg:.3f}')
        
        model_q.eval()
        with torch.no_grad():
            top1 = AverageMeter()
            losses = AverageMeter()
            for i, (images, labels) in enumerate(test_loader):
                images, labels = images.cuda(), labels.cuda()
                outputs = model_q(images)
                acc = accuracy(outputs, labels, topk=(1,))
                top1.update(acc.item(), images.size(0))
                losses.update(loss.item(), images.size(0))
                
                running_loss += loss.item()
                if i % 100 == 99:
                    print(f'[Epoch {epoch + 1}, Batch {i + 1}] acc: {top1.avg:.4f} loss: {losses.avg:.3f}')

    
    # 여기가 뭔가 Evaluation 할 수 있는 부분.
    # ex, fake quantize => real quantize 등

if __name__ == '__main__':
    main()