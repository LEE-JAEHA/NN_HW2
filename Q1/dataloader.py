import torch
import torchvision
import torchvision.transforms as transforms

########################################################################
"""torchvision datasets은 pickle module로 객체 구조를 갖는 cifar10 데이터들을 serialize,
    pickle 설명> https://pythontips.com/2013/08/02/what-is-pickle-in-python/
    pillow library로 해당 데이터를 이미지화합니다.  
    데이터를 그냥 불러도 되지만, 대부분 데이터를 transform하죠! 그 이유는? Batch Normalization을
    생각하면 편할 것 같아요 :) 
     https://discuss.pytorch.org/t/understanding-transform-normalize/21730/2
     """

class Dataset_face():
    def __init__(self,root,batch=8):
        self.batch_size = batch
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.transform2 = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
                # transforms.RandomAffine(
                #     degrees=(10, 30),
                #     translate=(0.25, 0.5),
                #     scale=(1.2, 2.0),
                # ),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )
        self.root = root
        self.train_set = torchvision.datasets.ImageFolder(root=self.root+"/facescrub_train", transform=self.transform2)
        self.train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=batch, shuffle=True, num_workers=4)

        # self.validset = torchvision.datasets.ImageFolder(root=self.root + "/train", transform=self.transform)
        # self.validloader = torch.utils.data.DataLoader(self.validset, batch_size=128, shuffle=True, num_workers=4)

        self.test_set = torchvision.datasets.ImageFolder(root=self.root + "/facescrub_test", transform=self.transform)
        self.test_loader = torch.utils.data.DataLoader(self.test_set, batch_size=batch, shuffle=True, num_workers=4)



    def __getitem__(self, item):
        return self.x,self.y



