import torch
import torch.nn as nn
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim
import argparse
import sys, os
from LeNet5 import LeNet_5
from loss import *
#from tools import *

def get_data(args):
    download_root = "./data"
    my_transform = transforms.Compose([
        transforms.Resize([32,32]),
        transforms.ToTensor(),
        transforms.Normalize((0.5,),(1.0,))
    ])
    train_dataset = MNIST(
        root=download_root,
        transform=my_transform,
        train=True,
        download=args.download)
    eval_dataset = MNIST(
        root=download_root,
        transform=my_transform,
        train=False,
        download=args.download
    )
    test_dataset = MNIST(
        root=download_root,
        transform=my_transform,
        train=False,
        download=args.download
    )
    return train_dataset, eval_dataset, test_dataset


def make_dataloader(train_dataset, eval_dataset, test_dataset):
    train_loader = DataLoader(train_dataset,
                              batch_size=32,
                              num_workers=0,
                              pin_memory=True,
                              drop_last=True,
                              shuffle=True)
    eval_loader = DataLoader(eval_dataset,
                             batch_size=32,
                             num_workers=0,
                             pin_memory=True,
                             drop_last=False,
                             shuffle=False)
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             num_workers=0,
                             pin_memory=True,
                             drop_last=False,
                             shuffle=False)

    return train_loader, eval_loader, test_loader

def train_model(args,model,device, train_loader, epoch=3):
    model.to(device)
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    criterion = MNISTloss(device=device)

    for e in range(epoch):
        total_loss = 0
        for i, batch in enumerate(train_loader):
            img = batch[0]#torch.Size([32, 1, 32, 32])
            gt = batch[1]# tensor([0,1,3,2,3,5,...,3,2,4])
            img = img.to(device)
            gt = gt.to(device)

            out = model(img)
            loss_val = criterion(out, gt)
            loss_val.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss_val.item()
        mean_loss = total_loss/i
        scheduler.step()
        print("-> {} epoch mean loss: {}".format(e, mean_loss))
        torch.save(model.state_dict(), args.output_dir+"/model_epoch"+str(e)+".pt")
    print("===== END Train =====")

def eval_model(args, model, device, eval_loader):
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    accuracy = 0
    cnt_batch = 0
    for batch in eval_loader:
        img = batch[0]
        gt = batch[1]
        img = img.to(device)

        out = model(img)
        out = out.cpu()
        for i, out in enumerate(out):
            if torch.argmax(out) == gt[i]:
                accuracy += 1
        print("{} batch : {}/{}".format(cnt_batch, accuracy, i+1))
        cnt_batch += 1
        accuracy = 0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',
                        dest='mode',
                        default=None,
                        type=str)
    parser.add_argument('--download',
                        dest='download',
                        default=False,
                        type=bool)
    parser.add_argument('--output_dir',
                        dest = 'output_dir',
                        default = './output',
                        type = str)
    parser.add_argument('--checkpoint',
                        dest='checkpoint',
                        default=None,
                        type=str)
    args = parser.parse_args()
    return args

def main(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    train_dataset, eval_dataset, test_dataset = get_data(args)
    train_loader, eval_loader, test_loader = make_dataloader(train_dataset,eval_dataset,test_dataset)
    model = LeNet_5()

    if args.mode == "train":
        train_model(args,model, device, train_loader)
    elif args.mode == "eval":
        eval_model(args,model, device, eval_loader)

if __name__=="__main__":
    main(parse_args())