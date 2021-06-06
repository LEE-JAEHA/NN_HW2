# from model.resnet import resnet18 as ResNet18
from model.resnet import  ResNet18
from dataloader import Dataset_face
import torch
import os
import argparse
from tqdm import tqdm # data 진행 속도 확인
import torch.backends.cudnn as cudnn
import random
import numpy as np
from torchsummary import summary as summary_
from draw_graph import plot_graph
import torchvision.models as models


def save_parameters(network,save_loc, optimizer):
    # Generator
    state_dict = dict()
    state_dict['network'] = network.state_dict()
    state_dict['optimizer'] = optimizer.state_dict()
    # state_dict['step'] = step
    # print(state_dict)
    torch.save(state_dict, save_loc)
    # state_dict['inr_function'] = self.inr_function.state_dict()

def load_parameters(network,path_):
    data = torch.load(path_)
    layer,optimizer = data['network'],data['optimizer']
    network.load_state_dict(layer,strict=True)



def train():
    parser = argparse.ArgumentParser(description="helper")
    parser.add_argument("--save_model", default="./checkpoint", help="save")
    parser.add_argument("--lr",type = float, default=0.001, help="learning rate?")
    parser.add_argument("--batch", type=int, default=512, help="Number of batch?")
    parser.add_argument("--epoch", type=int, default=200, help="Number of Epoch")
    parser.add_argument("--mode", type=str, default="train", help="GPU Select")
    parser.add_argument("--n_gpu", default="0", help="Number of Epoch")

    parser.add_argument("--pretrained", default=True, help="Will load pretrained model")
    parser.add_argument("--model_save", default="./checkpoint_c_B512_ep200", help="model save folder")
    parser.add_argument("--model_load", default="./checkpoint_b_B512_ep200/model_149.pth", help="model load folder")
    args = parser.parse_args()
    epochs = args.epoch

    os.environ["CUDA_VISIBLE_DEVICES"] = args.n_gpu
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

    model_save = args.model_save
    if os.path.exists(model_save) == False:
        print("Make Directory : {0}".format(model_save))
        os.mkdir(model_save)



    data = Dataset_face(root="./dataset",batch=args.batch)
    #
    train_loader = data.train_loader
    test_loader = data.test_loader
    # import pdb;pdb.set_trace()

    #GPU 확인
    print('Available devices ', torch.cuda.device_count())
    print('Current cuda device ', torch.cuda.current_device())

    if torch.cuda.is_available():
        device = torch.device("cuda:0"); print("using cuda:0")
    else:
        device = torch.device("cpu") ; print("using CPU")
    print("Device ? : {0}".format(device))
    model = ResNet18(pretrained=True)  # load model



    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 원하는 에폭마다, 이전 학습률 대비 변경폭에 따라 학습률을 감소시켜주는 방식
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, threshold=1, patience=1, mode='min')
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[35, 70], gamma=0.5)
    # scheduler = torch.optim.lr_scheduler`.ReduceLROnPlateau(optimizer, 'min', factor=0.5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    # load_parameters(model,optimizer,"./checkpoint/model_99.pth")
    # path_ ="./checkpoint/model_99.pth"

    # resnet18 = models.resnet18()
    model.to(device) # load to GPU or CPU
    # summary_(model, input_size=(3, 64, 64))
    # import pdb;
    # pdb.set_trace()

    # 학습률 스케줄러는 옵티.마이져를 통해 적용된다.
    # optimizer = torch.optim.SGD(model.parameters(), lr= args.lr,weight_decay=1e-4,nesterov=True,momentum=0.9)



    best_acc=-1
    best_idx=-1

    best_accuracy_test=-1;best_accuracy_train=-1
    train_loss = list();test_loss = list();train_accuracy = list();test_accuracy = list();
    best_test=(-1,-1);best_train=(-1,-1)
    for epoch in range(1, epochs + 1):
        model.train()
        sum_loss = 0;total=0;correct=0
        for idx, data in tqdm(enumerate(train_loader),total=len(train_loader),desc="train",smoothing=0.9):  # 한번의 for문 마다 전체 데이터를 나눈만큼 돈다
            imgs, labels = data
            imgs, labels = imgs.to(device), labels.to(device)
            pred = model(imgs)
            loss = criterion(pred, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()


            _, predicted_valid = torch.max(pred, 1)
            total += labels.size(0)
            correct += predicted_valid.eq(labels).sum().item()
        acc = 100 * correct / total
        if acc > best_train[1]:
            best_train = epoch, acc

        train_accuracy.append(acc)
        print("Epoch {0} | Accuracy of Training set : {1}".format(epoch, acc))


        # import pdb;pdb.set_trace()
        print("Epoch {0} | Train loss : {1}".format(epoch,sum_loss / (idx + 1)))
        train_loss.append(sum_loss/len(train_loader))

        model_name = "./model_{}.pth".format(epoch)
        # torch.save(model.state_dict(), model_name)
        scheduler.step()  # Optimizer만 사용하려면 얘 지우고 optimizer 부분 주석 제거

        running_loss_valid = 0
        # total=0
        # correct=0
        # with torch.no_grad():
        #     print("Test with Training set")
        #     for idx,item in tqdm(enumerate(train_loader),total=len(train_loader),desc="train",smoothing=0.9):  # 한번의 for문 마다 전체 데이터를 나눈만큼 돈다
        #         images,labels = item
        #         images,labels=images.to(device),labels.to(device)
        #         output = model(images)
        #         _, predicted_valid = torch.max(output, 1)
        #         total += labels.size(0)
        #         correct += predicted_valid.eq(labels).sum().item()
        #     acc=100 * correct / total
        #     if acc >best_train[1]:
        #         best_train = epoch,acc
        #
        #     train_accuracy.append(acc)
        #     print("Epoch {0} | Accuracy of Training set : {1}".format(epoch,acc))



        model.eval()
        correct=0
        running_loss_test=0
        total=0
        with torch.no_grad():
            print("Test start")
            for idx,data in tqdm(enumerate(test_loader),total=len(test_loader),desc="test",smoothing=0.9):
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                output = model(images)
                loss_valid = criterion(output, labels)
                running_loss_test += loss_valid.item()
                _, predicted_valid = torch.max(output, 1)
                total += labels.size(0)
                correct += predicted_valid.eq(labels).sum().item()
            acc = 100 * correct / total
            print("Epoch {0} | Accuracy of Testset : {1}".format(epoch,acc))
            if acc > best_acc:
                best_acc = acc
                best_test = epoch,acc
                print("** The best Accuracy of this network : {0}, epoch:{1}".format(acc,epoch))
            print("Epoch {0} | Test loss : {1}".format(epoch, running_loss_test / len(test_loader)))
            test_loss.append(running_loss_test / len(test_loader))
            test_accuracy.append(acc)
        if epoch >1 :
            save_name = "{0}/model_{1}.pth".format(model_save,epoch)
            save_parameters(model, save_name, optimizer)
    plot_graph(epochs,(train_loss,test_loss),(train_accuracy,test_accuracy),"")
    print("Best Accuracy of Trainset | epoch : {0} / acc : {1}".format(best_train[0],best_train[1]))
    print("Best Accuracy of Testset  | epoch : {0} / acc : {1}".format(best_test[0], best_test[1]))
    print(args)


if __name__ == '__main__':
    train()