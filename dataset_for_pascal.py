import os
import sys
import tarfile
import collections
from torch.utils.data import Dataset,DataLoader
from PIL import Image,ImageDraw
from torchvision import transforms
import numpy as np
import random as rd

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

classes = ('__background__',  # always index 0
           'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
           'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')



grid=3

classCount=len(classes)-1
block=(1+4+classCount)

def horzFlip(x,t,w):
    """
    Horizontal flip for data augmentation

    IN:
    x: original image PIL format
    t: target label
    w: width of the image

    OUT: flipped image, flipped target label

    """

    hFlipped=np.zeros((grid*grid*block))
    for each in range((grid*grid)):
        tempData=t[each*block:each*block+block]
        if tempData[0]==1:
            newXmin=w-tempData[3]
            newXmax=w-tempData[1]
            tempData[1]=newXmin
            tempData[3]=newXmax
            hFlipped[each*block:each*block+block]=tempData
        else:
            break
    return x.flip(2), hFlipped



def adder(customTarget,h,w,n):
    """
    Target filler for each object

    customTarget: object's information in JSON.
    h: height of the image
    w: width of the image

    OUT: label version of the object. (5+C) format.

    """
    boxes=customTarget["bndbox"]
    box,boy,boxx,boyy=boxes["xmin"],boxes["ymin"],boxes["xmax"],boxes["ymax"]
    boundaries=[int(box)/w,int(boy)/h,int(boxx)/w,int(boyy)/h]
    insertedClass=[0]*classCount
    insertedClass[classes.index(customTarget["name"])-1]=1
    tobeinserted=[1]+boundaries+insertedClass

    return tobeinserted

# https://github.com/pytorch/vision/blob/master/torchvision/datasets/voc.py

class VOCDetection(Dataset):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Detection Dataset.
    Args:
        root (string): Root directory of the VOC Dataset.
        year (string, optional): The dataset year, supports years 2007 to 2012.
        image_set (string, optional): Select the image_set to use, ``train``, ``trainval`` or ``val``
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
            (default: alphabetic indexing of VOC's 20 classes).
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, required): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.


            REFERENCE:

            /*************************************************
            * Title: voc.py
            * Author: pytorch
            * Date: 16 September 2019 (final commit)
            * Code version: Unknown
            * Availability: https://github.com/pytorch/vision/blob/master/torchvision/datasets/voc.py
            *************************************************/



    """

    def __init__(self,
                 root,
                 year,
                 image_set,
                 isEval,
                 download=False,
                 target_transform=None,
                 transforms=None):
        self.root=root
        self.transforms=transforms

        self.isEval=isEval

        voc_root = self.root
        image_dir = os.path.join(voc_root, 'JPEGImages')
        annotation_dir = os.path.join(voc_root, 'Annotations')

        if not os.path.isdir(voc_root):
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        splits_dir = os.path.join(voc_root, 'ImageSets/Main')

        split_f = os.path.join(splits_dir, image_set.rstrip('\n') + '.txt')

        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.annotations = [os.path.join(annotation_dir, x + ".xml") for x in file_names]
        assert (len(self.images) == len(self.annotations))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a dictionary of the XML tree.
        """
        img = Image.open(self.images[index]).convert('RGB')
        target = self.parse_voc_xml(
            ET.parse(self.annotations[index]).getroot())

        imgRaw=img.copy()

        if self.transforms is not None:
            img = self.transforms(img)

        objects=target["annotation"]["object"]
        sizes=target["annotation"]["size"]
        totalObjCount=len(objects)

        customTarget=objects.copy()
        custTarget=np.zeros((grid*grid*block))

        w,h=int(sizes["width"]),int(sizes["height"])

        "[isObj,bx,by,bxx,byy,c1,c2,c3...]"
        if type(customTarget)==type([1]):

            if totalObjCount<(grid*grid):#13
                for each in range(len(customTarget)):
                    custTarget[each*block:each*block+block]=adder(customTarget[each],h,w,totalObjCount)
            else:
                for each in range((grid*grid)):
                    custTarget[each*block:each*block+block]=adder(customTarget[each],h,w,totalObjCount)


        else:
            custTarget[0*block:0*block+block]=adder(customTarget,h,w,1)


        if self.isEval:
            return img, custTarget, [w,h], imgRaw,totalObjCount
        else:
            return img, custTarget, [w,h], imgRaw,totalObjCount

    def __len__(self):
        return len(self.images)

    def parse_voc_xml(self, node):
        voc_dict = {}
        children = list(node)
        if children:
            def_dic = collections.defaultdict(list)
            for dc in map(self.parse_voc_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            voc_dict = {
                node.tag:
                    {ind: v[0] if len(v) == 1 else v
                     for ind, v in def_dic.items()}
            }
        if node.text:
            text = node.text.strip()
            if not children:
                voc_dict[node.tag] = text
        return voc_dict


    def deNormalizer(self,target,pred):
        """
        Inputs labels and preds (final layers), and extracts the objects in it. (Preprocessing)

        IN:
        target, pred: ground truth, model output

        OUT:
        Extracted objects in a list.

        """
        deNormalizedMatrix=[]
        each=0
        pred=list(pred)
        while each < (len(pred)):
            if pred[each]>0.70:
                termMtx=[]
                for eacher in range(each,each+5):
                    termMtx.append(pred[eacher])

                getMax=max(pred[each+5:each+25])
                termMtx.append(pred[each+5:each+25].index(getMax))

                deNormalizedMatrix.append(termMtx)
            each+=25

        target=list(target)
        deNormalizedTarget=[]
        each=0
        while each < (len(target)):
            if target[each]==1:
                termMtx=[]
                for eacher in range(each,each+5):
                    termMtx.append(target[eacher])

                getMax=max(target[each+5:each+25])
                termMtx.append(target[each+5:each+25].index(getMax))

                deNormalizedTarget.append(termMtx)
                each+=25
            else:
                break

        return deNormalizedTarget,deNormalizedMatrix

    def drawImages(self, target, pred, b, imgRaw):
        """
        Draw images
        IN: ground truth, model prediction, image height/width, original image PIL
        OUT: Ground image with plotted boxes version.
        """

        deNormalizedTarget,deNormalizedMatrix=self.deNormalizer(target,pred)

        "drawer"
        draw=ImageDraw.Draw(imgRaw)
        counter=0
        for each in deNormalizedMatrix:
            draw.rectangle([each[1]*b[0], each[2]*b[1], each[3]*b[0], each[4]*b[1]],outline="red", width=2)
            draw.text((each[1]*b[0], each[2]*b[1]),fill="black",text=str(counter), width=2)
            draw.text((each[3]*b[0], each[4]*b[1]),fill="black",text=classes[each[5]+1], width=2)
            counter+=1

        counter=0
        for each in deNormalizedTarget:
            draw.rectangle([each[1]*b[0], each[2]*b[1], each[3]*b[0], each[4]*b[1]],outline="green", width=2)
            draw.text((each[1]*b[0], each[2]*b[1]),fill="black",text=str(counter), width=2)
            draw.text((each[3]*b[0], each[4]*b[1]),fill="black",text=classes[each[5]+1], width=2)
            counter+=1

        imgRaw.show()

    def iou(self, target, pred):
        """
        IoU calculator.

        IN: ground truth, model prediction
        OUT: model objects and ground truth objects are matched and IoU is calculated. For evaluation purposes.
        """

        deNormalizedTarget,deNormalizedMatrix=self.deNormalizer(target,pred)
        matchRates=[]
        for each in range(len(deNormalizedMatrix)):

            for eacher in range(len(deNormalizedTarget)):

                """
                Title: pytorch-0.4-yolov3

                Author: Young-Sun (Andy) Yun (GitHub: andy-yun)
                Date: June 1 2018
                Code version: 0.9.1

                Availability: https://github.com/andy-yun/pytorch-0.4-yolov3
                --------


                Only for IOU part due to complexity, but modified to fill the matchRates list.
                """

                x1_min = min(deNormalizedMatrix[each][1],deNormalizedTarget[eacher][1])
                x2_max = max(deNormalizedMatrix[each][3],deNormalizedTarget[eacher][3])
                y1_min = min(deNormalizedMatrix[each][2],deNormalizedTarget[eacher][2])
                y2_max = max(deNormalizedMatrix[each][4],deNormalizedTarget[eacher][4])


                w1, h1 = deNormalizedMatrix[each][3] - deNormalizedMatrix[each][1], deNormalizedMatrix[each][4] - deNormalizedMatrix[each][2]
                w2, h2 = deNormalizedTarget[eacher][3] - deNormalizedTarget[eacher][1], deNormalizedTarget[eacher][4] - deNormalizedTarget[eacher][2]



                w_union = x2_max - x1_min
                h_union = y2_max - y1_min
                w_cross = w1 + w2 - w_union
                h_cross = h1 + h2 - h_union
                carea = 0
                if w_cross <= 0 or h_cross <= 0:
                    matchRates.append([0.0,each,eacher,False,0])
                else:
                    area1 = w1 * h1
                    area2 = w2 * h2
                    carea = w_cross * h_cross
                    uarea = area1 + area2 - carea

                    matchRates.append([float(carea/uarea),each,eacher,deNormalizedMatrix[each][5]==deNormalizedTarget[eacher][5],deNormalizedTarget[eacher][5]])

        matchRates.sort(key=lambda x: x[0], reverse = True)

        totalKey=list(range(len(deNormalizedMatrix)))
        totalValue=list(range(len(deNormalizedTarget)))

        bestMatch=[]

        for each in matchRates:
            if each[1] in totalKey and each[2] in totalValue:
                bestMatch.append(each)
                totalKey.remove(each[1])
                totalValue.remove(each[2])

            if len(totalKey)==0:
                break

        return bestMatch
