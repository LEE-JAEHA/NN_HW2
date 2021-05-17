from model.resnet import ResNet,ResNet18
import torch
import os
import argparse
from tqdm import tqdm # data 진행 속도 확인
from torchsummary import summary # model 구조 확인
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
import random
import numpy as np


def save_parameters(step, save_loc, network, optimizer, lr):
    # Generator
    state_dict = dict()
    state_dict['network'] = network.state_dict()
    # state_dict['optimizer'] = optimizer.state_dict()
    # state_dict['lr'] = lr
    # state_dict['step'] = step
    # print(state_dict)
    torch.save(state_dict, save_loc)
    # state_dict['inr_function'] = self.inr_function.state_dict()


def train():
    parser = argparse.ArgumentParser(description="helper")
    parser.add_argument("--save_model", default="./checkpoint", help="save")
    parser.add_argument("--lr",type = float, default=0.01, help="learning rate?")
    parser.add_argument("--batch", type=int, default=8, help="Number of batch?")
    parser.add_argument("--epoch", type=int, default=200, help="Number of Epoch")
    parser.add_argument("--n_gpu", default="0", help="Number of Epoch")
    parser.add_argument("--pretrained", default="./checkpoint/",
                        help="file name?")
    parser.add_argument("--mode", type=str, default="train", help="mode select")
    args = parser.parse_args()
    epochs = args.epoch

    os.environ["CUDA_VISIBLE_DEVICES"] = parser.n_gpu
    print('CUDA available: {}'.format(torch.cuda.is_available()))
    print(os.environ["CUDA_VISIBLE_DEVICES"])
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(0)
    print(args) # argument 들 확인


    #GPU 확인
    print('Available devices ', torch.cuda.device_count())
    print('Current cuda device ', torch.cuda.current_device())

    if torch.cuda.is_available():
        device = torch.device("cuda:0"); print("using cuda:0")
    else:
        device = torch.device("cpu") ; print("using CPU")
    print("Device ? : {0}".format(device))


    model = ResNet18()  # load model
    model.to(device) # load to GPU or CPU


    criterion = torch.nnCrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    # 학습률 스케줄러는 옵티.마이져를 통해 적용된다.
    optimizer = torch.optim.SGD(model.parameters(), lr= args.learning_rate,weight_decay=1e-4,nesterov=True,momentum=0.9)

    # 원하는 에폭마다, 이전 학습률 대비 변경폭에 따라 학습률을 감소시켜주는 방식
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, threshold=1, patience=1, mode='min')
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[35, 70], gamma=0.5)
    # scheduler = torch.optim.lr_scheduler`.ReduceLROnPlateau(optimizer, 'min', factor=0.5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    best_acc=-1
    best_idx=-1

    best_accuracy_test=-1;best_accuracy_train=-1
    train_loss = list();test_loss = list();train_accuracy = list();test_accuracy = list();

    for epoch in range(1, epochs + 1):
        for idx, data in enumerate(tqdm(data_.train_loader), 1):  # 한번의 for문 마다 전체 데이터를 나눈만큼 돈다
            model.train()
            imgs, labels = data
            imgs, labels = imgs.to(device), labels.to(device)
            pred = model(imgs)
            loss = criterion(pred, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()
        model.eval()
        model_name = ""
        torch.save(model.state_dict(), model_name)

        scheduler.step() # Optimizer만 사용하려면 얘 지우고 optimizer 부분 주석 제거
