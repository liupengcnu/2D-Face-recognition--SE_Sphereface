from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
torch.backends.cudnn.bencmark = True
import os
import sys
import cv2
import random
import datetime
import argparse
import numpy as np
from PIL import Image
from dataset import ImageDataset
from matlab_cp2tform import get_similarity_transform_for_cv2

from networks import se_sphere

from loss.angleLoss import AngleLoss
from loss.amsoftmax import AMSoftmax


def alignment(src_img, src_pts):
    of = 2
    ref_pts = [[30.2946+of, 51.6963+of],
               [65.5318+of, 51.5014+of],
               [48.0252+of, 71.7366+of],
               [33.5493+of, 92.3655+of],
               [62.7299+of, 92.2041+of]]
    crop_size = (96+of*2, 112+of*2)

    s = np.array(src_pts).astype(np.float32)
    r = np.array(ref_pts).astype(np.float32)

    tfm = get_similarity_transform_for_cv2(s, r)
    face_img = cv2.warpAffine(src_img, tfm, crop_size)
    return face_img


def dataset_load(name, filename, pindex, cacheobj, zfile):
    split = filename.split('\t')
    img_dir = split[0]
    classid = int(split[1])
    src_pts = []
    if not os.path.exists(img_dir):
        return None
    for i in range(5):
        src_pts.append([int(split[2*i+2]), int(split[2*i+3])])
    img = Image.open(img_dir)
    img = np.array(img)
    img = alignment(img, src_pts)
    if ':train' in name:
        if random.random() > 0.5:
            img = cv2.flip(img, 1)
        if random.random() > 0.5:
            rx = random.randint(0, 2*2)
            ry = random.randint(0, 2*2)
            img = img[ry:ry+112, rx:rx+96, :]
        else:
            img = img[2:2+112, 2:2+96, :]
    else:
        img = img[2:2+112, 2:2+96, :]
    img = img.transpose(2, 0, 1).reshape((1, 3, 112, 96))
    img = (img - 127.5) / 128.0  # (-1, 1)
    label = np.zeros((1, 1), np.float32)
    label[0, 0] = classid
    return (img, label)


def save_model(model, filename):
    state = model.state_dict()
    for key in state:
        names = key.split('.')
        names = names.remove(names[0])
        key = ".".join(names)
        state[key] = state[key].clone().cpu()
    torch.save(state, filename)


def dt():
    return datetime.datetime.now().strftime('%H:%M:%S')


def train(epoch, args):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    batch_idx = 0
    ds = ImageDataset(imageroot=args.dataset,
                      callback=dataset_load,
                      imagelistfile=args.data_list,
                      name=args.net + ':train',
                      batchsize=args.batchsize,
                      shuffle=True,
                      nthread=args.nthread,
                      imagesize=128)

    batch_num = ds.imagenum // args.batchsize
    while True:
        img, label = ds.get()
        if img is None:
            break
        inputs = torch.from_numpy(img).float()
        targets = torch.from_numpy(label[:, 0]).long()
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        outputs = outputs[0]  # 0=cos_theta 1=phi_theta
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).sum().item()

        if batch_idx % 10 == 0:
            print(dt(), 'Epoch=%d batch: %d/%d Loss=%.4f | Acc=%.4f%%' %
                    (epoch, batch_idx, batch_num, train_loss/(batch_idx+1), correct*100.0/total))
        batch_idx += 1


def get_args():
    parser = argparse.ArgumentParser(description='PyTorch sphereface')
    parser.add_argument('--net', '-n', default='sphere36a', type=str)

    parser.add_argument('--dataset', type=str, default='../dataset/WebFace_224x192_V1')
    parser.add_argument('--data_list', type=str, default='./data/casia_landmark.txt')

    parser.add_argument('--pre_model', type=str, default='')
    parser.add_argument('--save_model', type=str, default='se_sphereface')

    parser.add_argument('--class_num', type=int, default=10574)
    parser.add_argument('--embedding_size', type=int, default=128)

    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--loss', default='angleLoss', type=str)
    parser.add_argument('--optimizer', default='sgd', type=str)
    parser.add_argument('--gamma', default=0.1, type=float, help='gamma')
    parser.add_argument('--batchsize', default=64, type=int)
    parser.add_argument('--nthread', default=4, type=int)
    parser.add_argument('--gpus', nargs='+', default=[0,1,2,3])

    parser.add_argument('--epoch_start', default=0, type=int)
    parser.add_argument('--epoch_end', default=250, type=int)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    if args.net == 'sphere36a':
        net = se_sphere.sphere36a(classnum=args.class_num, feature=False, embedding_size=args.embedding_size)
    else:
        pass

    if os.path.exists(args.pre_model):
        print('start load model: ', args.pre_model)
        net.load_state_dict(torch.load(args.pre_model))
    else:
        print('no file model: ', args.pre_model)

    gpu_str = ''
    for gpu in args.gpus:
        gpu_str = gpu_str + str(gpu) + ','
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        net.cuda()
    if len(args.gpus) > 1:
        net = torch.nn.DataParallel(net)

    if args.loss == 'amsoftmax':
        criterion = AMSoftmax()
    elif args.loss == 'angleLoss':
        criterion = AngleLoss()
    else:
        pass

    if args.optimizer == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=5e-4)
    else:
        pass
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=args.gamma)

    print('start: time={}'.format(dt()))
    for epoch in range(args.epoch_start, args.epoch_end):
        scheduler.step(epoch)
        train(epoch, args)
        if epoch % 50 == 0:
           save_model(net, '{}_{}_{}.pth'.format(args.net, args.save_model, epoch))
    print('finish: time={}\n'.format(dt()))
