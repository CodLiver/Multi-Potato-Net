from torch.autograd import Variable
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from PIL import Image
import torch,numpy as np,torch.nn as nn,torch.nn.functional as F,torch.optim as optim,torchvision.models as models,torch.backends.cudnn as cudnn,torchvision.datasets as datasets,time,os,shutil,sys
import xml

from dataset_for_pascal import VOCDetection as VOC
import dataset_for_pascal as dsStuff

"""
Lines to change before running: (numbers with * are required to decide before running)
18*, 19*, 161, 162, 229, 275*, 281, 294, 307, 344, 348

"""

"Insert PATH of the datasets"
PATH_TRAIN = "D:/VOCdevkit/VOC2007/"
PATH_TEST = "D:/VOCtest/VOC2007/"

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

classes = ('__background__',  # always index 0
           'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
           'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')

grid=3

# myTrainDataset = VOC("D:/VOCdevkit/VOC2007/",'2007','train',False,transforms=transforms.Compose([
#     transforms.Resize((384,384)),transforms.ToTensor()#(384,384)
# ]))
#
# myValDataset = VOC("D:/VOCdevkit/VOC2007/",'2007','val',True,transforms=transforms.Compose([
#     transforms.Resize((384,384)),transforms.ToTensor()
# ]))

"dataset definition area"
myTrainValDataset = VOC(PATH_TRAIN,'2007','trainval',False,transforms=transforms.Compose([
    transforms.Resize((384,384)),transforms.ToTensor()
]))


myTestDataset = VOC(PATH_TEST,'2007','test',True,transforms=transforms.Compose([
    transforms.Resize((384,384)),transforms.ToTensor()
]))

"Multi-Potato Net"
class PotatoNet(nn.Module):
    def __init__(self):
        super(PotatoNet, self).__init__()
        self.C=20
        self.B=1

        self.pool2 = nn.MaxPool2d(2,2)
        self.conv1= nn.Conv2d(3, 32, 5)
        self.conv1_1= nn.Conv2d(32, 32, 3,padding=1)
        self.conv2= nn.Conv2d(32, 128, 5)
        self.conv2_1= nn.Conv2d(128, 128, 3,padding=1)
        self.conv3= nn.Conv2d(128, 512, 5)
        self.conv4= nn.Conv2d(512, 512, 5)
        self.conv4_1= nn.Conv2d(512, 512, 3,padding=1)
        self.conv5_1= nn.Conv2d(512, 1024, 3,padding=1)
        self.conv6= nn.Conv2d(1024, self.B*(1+4+self.C), 7)

        self.fc0 = nn.Linear(in_features=self.B*(1+4+self.C)*7*7, out_features=self.B*(1+4+self.C)*grid*grid)
        self.fc1 = nn.Linear(in_features=self.B*(1+4+self.C)*grid*grid, out_features=grid*grid*self.B*(1+4+self.C))

    def forward(self, x):
        "forward layer"

        "residual block 1"
        x=self.pool2(self.conv1(x))
        x=F.elu(self.conv1_1(x))
        out=self.pool2(F.dropout(self.conv2(x), p=0.4, training=True))
        x=F.relu(self.conv2_1(out))
        x=F.relu(self.conv2_1(x))
        x=self.pool2(F.dropout(self.conv3(x+out), p=0.5, training=True))

        "residual block 2"
        out=self.pool2(F.dropout(self.conv4(x), p=0.5, training=True))
        x=F.relu(self.conv4_1(out))
        x=F.relu(F.dropout(self.conv4_1(x), p=0.5, training=True))
        x=F.relu(self.conv5_1(x))
        x=self.pool2(F.dropout(self.conv6(x+F.elu(self.conv5_1(out))), p=0.5, training=True))

        "flatten"
        x = x.view(x.size(0), -1)
        x=self.fc0(x)
        x=self.fc1(x)

        return x


def crossEntropy(real, pred):
    """
    Calculates Cross_entropy via negative log likelihood.

    IN: ground truth final layer, model prediction final layer
    OUT: outputs

    Taken from
    https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html#cross-entropy


    """

    logs=-pred[(real == 1).nonzero()]+torch.log(torch.exp(pred).sum())
    return logs

def lossFunc(real,pred,classNo=20):
    """
    calculates loss.
    IN: ground truth final layer, model prediction final layer
    OUT: mean loss

    """

    for eachTensor in range(len(pred)):
        each=0
        while each < len(pred[eachTensor]):
            if real[eachTensor][each]==0:
                pred[eachTensor][each+1:each+4+classNo+1]=0
            each+=5+classNo

    loss=(pred-real)**2

    for eachTensor in range(len(loss)):
        each=0
        while each < len(loss[eachTensor]):

            if real[eachTensor][each]==1:
                wght=crossEntropy(real[eachTensor][each+1+4:each+4+classNo+1], pred[eachTensor][each+1+4:each+4+classNo+1])
                loss[eachTensor][each+1+4:each+4+classNo+1]=wght

            each+=5+classNo

    loss=loss.mean()


    if loss>10:
        print("RIP",loss)
        print(pred,real)
        exit()
    return loss

def validationMAP(N):
    """
    calculates validation mAP
    IN: model itself
    OUT: Bool, whether the model is up to standard or scheduling/mid-process final tuning is needed.
    """
    sumofiou=0
    sumofnonzero=0
    totaldetected=0
    maxof=0
    totObjSum=0
    mAP50=0
    mAP75=0
    classMatchCount=0

    startCount=0#len(myTrainValDataset)-500
    endCount=5#len(myTrainValDataset)

    print("total tested images:",endCount-startCount)
    for each in range(startCount,endCount):

        x,t, wh, imgRaw, totalObjCount = myTrainValDataset.__getitem__(each)
        x,t = x.to(device), torch.FloatTensor(t).to(device).unsqueeze(0)
        p = N(x.unsqueeze(0))
        p,t=p.squeeze(0),t.squeeze(0)

        totObjSum+=totalObjCount
        obtained=myTrainValDataset.iou(t,p.cpu().data.numpy())

        for each in obtained:
            sumofiou+=each[0]
            if each[0]>maxof:
                maxof=each[0]
            if each[0]>0.5:
                mAP50+=1
            if each[0]>0.75:
                mAP75+=1
            if each[3]:
                classMatchCount+=1
        totaldetected+=len(obtained)


    if totaldetected>0:
        print("total detected obj:",totaldetected,"| average per detected box %:",(sumofiou*100)/totaldetected)
        print("max iou %:",maxof*100)
        mAPP=sumofiou/totObjSum
        print("total object amount:",totObjSum,"| mAP %:",mAPP*100)
        print("mAP50 %:",mAP50/totObjSum,"| mAP75 %:",mAP75/totObjSum)
        print("correctClass %",classMatchCount/totObjSum)

        "uncomment to allow saving or decide the requirements or any."
        # save
        # if mAPP*100>8:
        #     torch.save(N.state_dict(), "./"+str(mAPP)[:6].strip(".")+"state.m")
        #     torch.save(N, "./"+str(mAPP)[:6].strip(".")+"model.m")

        if mAPP*100<7:
            return True
        else:
            return False

    else:
        print("none detected")
        return True


"final overload test"
def testMAP(N):

    """
    calculates test mAP
    IN: model itself
    OUT: None
    """
    sumofiou=0
    sumofnonzero=0
    totaldetected=0
    maxof=0
    totObjSum=0
    mAP50=0
    mAP75=0
    classMatchCount=0

    totalImage=5#len(myTestDataset)

    print("total tested:",totalImage)
    for each in range(totalImage):
        try:
            x,t, wh, imgRaw, totalObjCount = myTestDataset.__getitem__(each)
            x,t = x.to(device), torch.FloatTensor(t).to(device)
            p = N(x.unsqueeze(0)).squeeze(0)

            totObjSum+=totalObjCount
            obtained=myTestDataset.iou(t,p.cpu().data.numpy())

            for each in obtained:
                sumofiou+=each[0]
                if each[0]>maxof:
                    maxof=each[0]
                if each[0]>0.5:
                    mAP50+=1
                if each[0]>0.75:
                    mAP75+=1
                if each[3]:
                    classMatchCount+=1
            totaldetected+=len(obtained)

        except Exception as e:
            continue

    if totaldetected>0:
        print("total detected obj:",totaldetected,"| average per detected box %:",(sumofiou*100)/totaldetected)
        print("max iou %:",maxof*100)
        mAPP=sumofiou/totObjSum
        print("total object amount:",totObjSum,"| mAP %:",mAPP*100)
        print("mAP50 %:",mAP50/totObjSum,"| mAP75 %:",mAP75/totObjSum)
        print("correctClass %",classMatchCount/totObjSum)

    else:
        print("none detected")


"""
isTest: "True, if want to use test dataset. False, if train"

Note to lecturer: I ran this test method only once or twice to write the evaluation section.
I used the eval dataset instead.

"""
isTest=False

if isTest:
    "Test area"
    N = PotatoNet().to(device)
    "insert other models?"
    N.load_state_dict(torch.load("./Multi-PotatoNet_state.m"))
    N.eval()
    testMAP(N)

else:
    "training"

    N = PotatoNet().to(device)

    "hyper parameters"
    lr=0.01
    wd=0.0005
    momentum=0.05
    epochTOTAL=5#30
    optimiser = torch.optim.SGD(N.parameters(), lr=lr, weight_decay=wd,momentum=momentum)
    totalImageTrain=10#(len(myTrainValDataset)-500)

    epoch = 0
    while (epoch<epochTOTAL):
        print("epoch",epoch)

        # arrays for metrics
        logs = {}
        train_loss_arr = np.zeros(0)

        N = N.train()
        # train
        
        for i in range(totalImageTrain):
            x,t,b,r,n = myTrainValDataset.__getitem__(i)
            xNew,tNew=dsStuff.horzFlip(x.clone(),t.copy(),1)
            xNew,tNew=xNew.to(device).unsqueeze(0), torch.FloatTensor(tNew).to(device)
            x,t = x.to(device), torch.FloatTensor(t).to(device)

            optimiser.zero_grad()
            x = torch.cat((x.unsqueeze(0), xNew), 0)
            t=torch.cat((t.unsqueeze(0),tNew.unsqueeze(0)),0)
            p = N(x).squeeze(0)
            loss = lossFunc(t,p)


            loss.backward()
            optimiser.step()

            train_loss_arr = np.append(train_loss_arr, loss.cpu().data)


            if i==500 or i==1500 or i==2500 or i == 3500:
                print("i:",i)

        N = N.eval()

        "this is the eval I used"
        validationMAP(N)

        print('loss', train_loss_arr.mean())
        print("------------------------------------------------------")
        print()
        epoch = epoch+1


"some bounding boxes from test"
M = PotatoNet().to(device)
"insert any other state?"
M.load_state_dict(torch.load("./Multi-PotatoNet_state.m"))
M.eval()
print("------------")
print("------------")
for each in [0,10,50,100,200,500,1000,2000,4500]:
    x,t, wh, imgRaw, totalObjCount = myTestDataset.__getitem__(each)
    x,t = x.to(device), torch.FloatTensor(t).to(device)
    p = M(x.unsqueeze(0)).squeeze(0)

    myTestDataset.drawImages(t,p.cpu().data.numpy(),wh,imgRaw)

    obtained=myTestDataset.iou(t,p.cpu().data.numpy())
    try:
        print("IoU per detected box",obtained[0][0]*100)
        print("Top-1?",obtained[0][3])
        print("------------")
    except:
        continue
