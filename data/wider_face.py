import os
import os.path
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
import random

class WiderFaceDetection(data.Dataset):
    def __init__(self, txt_path, preproc=None, image_size=(640,640)):
        self.preproc = preproc
        self.imgs_path = []
        self.words = []
        self.image_size = image_size
        f = open(txt_path,'r')
        lines = f.readlines()
        isFirst = True
        labels = []
        for line in lines:
            line = line.rstrip()
            if line.startswith('#'):
                if isFirst is True:
                    isFirst = False
                else:
                    labels_copy = labels.copy()
                    self.words.append(labels_copy)
                    labels.clear()
                path = line[2:]
                path = txt_path.replace('label.txt','images/') + path
                self.imgs_path.append(path)
            else:
                line = line.split(' ')
                label = [float(x) for x in line]
                labels.append(label)

        self.words.append(labels)
        self.indices = range(len(self.imgs_path))

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, index):
        mosaic = random.random() <= 1
        if mosaic:
            img, target = load_mosaic(self, index)
        else:
            img = cv2.imread(self.imgs_path[index])
            height, width, _ = img.shape
            print(' sssss \n\n')

            labels = self.words[index]
            annotations = np.zeros((0, 15))
            if len(labels) == 0:
                return annotations
            for idx, label in enumerate(labels):
                annotation = np.zeros((1, 15))
                # bbox
                annotation[0, 0] = label[0]  # x1
                annotation[0, 1] = label[1]  # y1
                annotation[0, 2] = label[0] + label[2]  # x2
                annotation[0, 3] = label[1] + label[3]  # y2

                # landmarks
                annotation[0, 4] = label[4]    # l0_x
                annotation[0, 5] = label[5]    # l0_y
                annotation[0, 6] = label[7]    # l1_x
                annotation[0, 7] = label[8]    # l1_y
                annotation[0, 8] = label[10]   # l2_x
                annotation[0, 9] = label[11]   # l2_y
                annotation[0, 10] = label[13]  # l3_x
                annotation[0, 11] = label[14]  # l3_y
                annotation[0, 12] = label[16]  # l4_x
                annotation[0, 13] = label[17]  # l4_y
                if (annotation[0, 4]<0):
                    annotation[0, 14] = -1
                else:
                    annotation[0, 14] = 1

                annotations = np.append(annotations, annotation, axis=0)
            target = np.array(annotations)
        if self.preproc is not None:
            img, target = self.preproc(img, target)

        return torch.from_numpy(img), target

def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    imgs = []
    for _, sample in enumerate(batch):
        for _, tup in enumerate(sample):
            if torch.is_tensor(tup):
                imgs.append(tup)
            elif isinstance(tup, type(np.empty(0))):
                annos = torch.from_numpy(tup).float()
                targets.append(annos)

    return (torch.stack(imgs, 0), targets)


def load_mosaic(self, index):
        # loads images in a 4-mosaic

        labels4 = []
        img_resize = [self.image_size[0],self.image_size[1]] #h,w
        mosaic_border = [-img_resize[0] // 2, -img_resize[1] // 2]
        s = img_resize
        #yc, xc = [int(random.uniform(-x, 2 * s[0] + x)) for x in mosaic_border]  # mosaic center x, y
        yc = img_resize[0]*2 // 2
        xc = img_resize[1]*2 // 2
        indices = [index] + random.choices(self.indices, k=3)  # 3 additional image indices
        annotations = np.zeros((0, 15))

        for i, index in enumerate(indices):
            # Load image
            img, _, (h, w), ratio = load_image(self, index, img_resize)
            #img = cv2.imread(self.imgs_path[index])

            # place img in img4
            if i == 0:  # top left
                img4 = np.full((s[0] * 2, s[1] * 2, 3), 114, dtype=np.uint8)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s[1] * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s[0] * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s[1] * 2), min(s[0] * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b

            # Labels
            labels = self.words[index].copy()

            annotations = xywhn2xyxy(labels, annotations, w, h, padw, padh, ratio)  # normalized xywh to pixel xyxy format
        target = np.array(annotations)
        return img4, target

def xywhn2xyxy(labels, annotations, w=640, h=640, padw=0, padh=0, ratio=1):
        if len(labels) == 0:
            return annotations
        for idx, label in enumerate(labels):
            annotation = np.zeros((1, 15))
            # bbox
            annotation[0, 0] = ratio * label[0] + padw   # x1
            annotation[0, 1] = ratio * label[1] + padh  # y1
            annotation[0, 2] = ratio * (label[0] + label[2]) + padw  # x2
            annotation[0, 3] = ratio * (label[1] + label[3]) + padh  # y2

            # landmarks
            annotation[0, 4] = ratio * label[4] + padw if label[4]!=-1 else label[4]   # l0_x
            annotation[0, 5] = ratio * label[5] + padh if label[5]!=-1 else label[5]   # l0_y
            annotation[0, 6] = ratio * label[7] + padw if label[7]!=-1 else label[7]   # l1_x
            annotation[0, 7] = ratio * label[8] + padh if label[8]!=-1 else label[8]  # l1_y
            annotation[0, 8] = ratio * label[10] + padw if label[10]!=-1 else label[10]  # l2_x
            annotation[0, 9] = ratio * label[11] + padh if label[11]!=-1 else label[11]  # l2_y
            annotation[0, 10] = ratio * label[13] + padw if label[13]!=-1 else label[13] # l3_x
            annotation[0, 11] = ratio * label[14] + padh if label[14]!=-1 else label[14] # l3_y
            annotation[0, 12] = ratio * label[16] + padw if label[16]!=-1 else label[16] # l4_x
            annotation[0, 13] = ratio * label[17] + padh if label[17]!=-1 else label[17] # l4_y
            if (annotation[0, 4]<0):
                annotation[0, 14] = -1
            else:
                annotation[0, 14] = 1
            
            annotations = np.append(annotations, annotation, axis=0)

        return annotations

def load_image(self, index, img_size):

    small_img_h_target = img_size[0]
    small_img_w_target = img_size[1]
    img = cv2.imread(self.imgs_path[index])  # BGR
    #assert img is not None, 'Image Not Found ' + path
    h0, w0 = img.shape[:2]  # orig hw
    if h0 > w0:
        r = small_img_h_target / max(h0, w0)  # resize image to img_size
        #if r != 1:  # always resize down, only resize up if training with augmentation
        if small_img_w_target < r * w0:
            r = small_img_w_target / w0  # resize image to img_size
        interp = cv2.INTER_AREA 
        img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
    else:
        r = small_img_w_target / max(h0, w0)  # resize image to img_size
        if small_img_h_target < r * h0:
            r = small_img_h_target / h0  # resize image to img_size
        #if r != 1:  # always resize down, only resize up if training with augmentation
        interp = cv2.INTER_AREA
        img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
    return img, (h0, w0), img.shape[:2], r  # img, hw_original, hw_resized, ratio

