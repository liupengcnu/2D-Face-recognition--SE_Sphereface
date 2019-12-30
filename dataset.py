from __future__ import print_function
import os
import time
import random
import numpy as np
import ctypes
import cv2
from PIL import Image
from multiprocessing import Process, Queue, Array

from matlab_cp2tform import get_similarity_transform_for_cv2


def numpy_to_share(index, image, label, nparrimage, nparrlabel):
    nparrimage[index, 0:image.size] = image.reshape(-1)[0:image.size]
    nparrlabel[index, 0:label.size] = label.reshape(-1)[0:label.size]


def return_batchdata(result, imagelist, labellist, pathlist, freearr, nparrimage, nparrlabel):
    index = freearr.get()  # wait for free index
    image = np.vstack(imagelist)
    label = np.vstack(labellist)
    numpy_to_share(index, image, label, nparrimage, nparrlabel)
    result.put((index, image.shape, label.shape, list(pathlist)))
    del imagelist[:]
    del labellist[:]
    del pathlist[:]


def dataset_handle(name, filelist, result, callback, bs, pindex, freearr, arrimage, arrlabel, zfile):
    cacheobj = type('', (), {})
    imagelist = []
    labellist = []
    pathlist = []
    nparrimage = np.frombuffer(arrimage.get_obj(), np.float32).reshape(10, int(len(arrimage)/10))
    nparrlabel = np.frombuffer(arrlabel.get_obj(), np.float32).reshape(10, int(len(arrlabel)/10))
    while True:
        filename = filelist.get()
        if filename.endswith('\n'):
            filename = filename[:-1]
        if filename == 'FINISH':
            break
        data = callback(name, filename, pindex, cacheobj, zfile)
        if data is not None:
            imagelist.append(data[0])
            labellist.append(data[1])
            pathlist.append(filename)
        if len(imagelist) == bs:
            return_batchdata(result, imagelist, labellist, pathlist, freearr, nparrimage, nparrlabel)
    if len(imagelist) > 0:
        return_batchdata(result, imagelist, labellist, pathlist, freearr, nparrimage, nparrlabel)
    result.put(('FINISH', 'FINISH', 'FINISH', 'FINISH'))


class ImageDataset(object):
    zipcache = {}

    def __init__(self, imageroot, callback, imagelistfile=None, batchsize=1, shuffle=False, nthread=4, name='',
                 imagesize=128, pathinfo=False, maxlistnum=None):
        self.callback = callback  # callback(name,filename,pindex,cacheobj) result=(image,label) in np.array
        self.batchsize = batchsize
        self.shuffle = shuffle
        self.nthread = nthread
        self.name = name

        self.arrimage = Array(ctypes.c_float, 10*batchsize*3*imagesize*imagesize)
        self.arrlabel = Array(ctypes.c_float, 10*batchsize*3*imagesize*imagesize)

        self.nparrimage = np.frombuffer(self.arrimage.get_obj(), np.float32).reshape(10, int(len(self.arrimage)/10))
        self.nparrlabel = np.frombuffer(self.arrlabel.get_obj(), np.float32).reshape(10, int(len(self.arrlabel)/10))

        self.filelist = Queue()
        self.result = Queue()
        self.freearr = Queue()
        self.imagenum = 0
        self.finishnum = 0
        self.zfile = None
        self.pathinfo = pathinfo

        for i in range(10):
            self.freearr.put(i)

        self.flist = []
        if imagelistfile is None and os.path.isdir(imageroot):
            for (dirpath, dirnames, filenames) in os.walk(imageroot):
                for filename in filenames:
                    self.flist.append(dirpath + '/' + filename)
        else:
            if os.path.isdir(imageroot):
                imageroot = imageroot + '/'
            else:
                imageroot = imageroot + ':'
                if '.zip:' in imageroot:
                    import zipfile
                    zipfilepath = imageroot.split(':')[0]
                    if zipfilepath in ImageDataset.zipcache:
                        self.zfile = ImageDataset.zipcache[zipfilepath]
                    else:
                        self.zfile = zipfile.ZipFile(zipfilepath)
                        ImageDataset.zipcache[zipfilepath] = self.zfile

            if '.zip:' in imageroot and imagelistfile is None:
                for zf in self.zfile.filelist:
                    self.flist.append(imageroot+zf.filename)
            elif '.zip:' in imagelistfile:
                with self.zfile.open(imagelistfile.split(':')[1]) as f:
                    lines = f.readlines()
                for line in lines:
                    self.flist.append(imageroot+line)  # zippath:filename classname
            else:
                with open(imagelistfile) as f:
                    lines = f.readlines()
                for line in lines:
                    self.flist.append(imageroot+line)  # root/filepath classname || zippath:filename classname

        self.imagenum = len(self.flist)
        if self.shuffle:
            random.shuffle(self.flist)
        for filepath in self.flist:
            self.filelist.put(filepath)
            if maxlistnum is not None:
                maxlistnum -= 1
            if maxlistnum == 0:
                break

        for i in range(nthread):
            self.filelist.put('FINISH')
            p = Process(target=dataset_handle,
                        args=(self.name, self.filelist, self.result, self.callback, self.batchsize, i,
                              self.freearr, self.arrimage, self.arrlabel, self.zfile))
            p.start()

    def get(self):
        while True:
            index, imageshape, labelshape, pathlist = self.result.get()
            if type(index) == str and index == 'FINISH':
                self.finishnum += 1
                if self.finishnum == self.nthread:
                    if self.pathinfo:
                        return (None, None, None)
                    else:
                        return (None, None)
                else:
                    continue
            imagesize = np.prod(imageshape)
            labelsize = np.prod(labelshape)
            image = np.empty(imageshape, np.float32)
            label = np.empty(labelshape, np.float32)
            image.reshape(imagesize)[:] = self.nparrimage[index, 0:imagesize]
            label.reshape(labelsize)[:] = self.nparrlabel[index, 0:labelsize]
            self.freearr.put(index)
            if self.pathinfo:
                return (image, label, pathlist)
            else:
                return (image, label)


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
    img = (img - 127.5) / 128.0
    label = np.zeros((1, 1), np.float32)
    label[0, 0] = classid
    return (img, label)


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


if __name__ == "__main__":
    dataset = '../dataset/WebFace_224x192_V1'
    net = 'sphere36a'
    ds = ImageDataset(imageroot=dataset,
                      callback=dataset_load,
                      imagelistfile='data/casia_landmark.txt',
                      name=net + ':train',
                      batchsize=256,
                      shuffle=True,
                      nthread=2,
                      imagesize=112)

    while True:
        time_start = time.time()
        img, label = ds.get()
        time_end = time.time()
        print('time: ', time_end-time_start, img.shape, label.shape)
