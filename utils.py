import json
import cv2
import numpy as np
from scipy.misc import imresize
from torch.autograd import Variable

FINAL_W=32
FINAL_H=32

def readFile( fname ):
    with open( fname, 'r') as f:
        return f.read()

def readJson( fname ):
    with open( fname, 'r' ) as f:
        return json.load( f )

def writeFile( fname, contents ):
    with open( fname, 'w' ) as f:
        f.write( contents )

def writeJson( fname, data ):
    with open( fname, 'w') as outfile:
        json.dump(data, outfile)

def fixedWidthImg( img ):
    ( currentH, currentW) = img.shape
    if( currentH > currentW ):
        newWidth = int( currentW * ( FINAL_H/currentH ) )
        newHeigh = FINAL_H
        offsetY=0
        offsetX = int( ( FINAL_W - newWidth )/2 )
    else:
        newHeigh = int( currentH * ( FINAL_W/currentW ) )
        newWidth = FINAL_W
        offsetX=0
        offsetY = int( ( FINAL_H - newHeigh )/2 )
    resizedImg = cv2.resize( img, ( newWidth, newHeigh ), interpolation=cv2.INTER_CUBIC )
    canvasImg = np.ones( ( FINAL_H, FINAL_W ), dtype=np.uint8)*255
    canvasImg[offsetY:offsetY+newHeigh, offsetX:offsetX+newWidth] = resizedImg
    return canvasImg




class averager(object):
    """Compute average for `torch.Variable` and `torch.Tensor`. """

    def __init__(self):
        self.reset()

    def add(self, v):
        if isinstance(v, Variable):
            count = v.data.numel()
            v = v.data.sum()
        elif isinstance(v, torch.Tensor):
            count = v.numel()
            v = v.sum()

        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res

class resizeNormalize(object):

    def __init__(self, size, interpolation='bilinear'):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = imresize( img, self.size, interp=self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img

class alignCollate(object):

    def __init__(self, imgH=32, imgW=100, keep_ratio=False, min_ratio=1):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio = keep_ratio
        self.min_ratio = min_ratio

    def __call__(self, batch):
        images, labels = zip(*batch)
        imgH = self.imgH
        imgW = self.imgW
        if self.keep_ratio:
            ratios = []
            for image in images:
                _, h, w = image.shape
                ratios.append(w / float(h))
            ratios.sort()
            max_ratio = ratios[-1]
            imgW = int(np.floor(max_ratio * imgH))
            imgW = max(imgH * self.min_ratio, imgW)  # assure imgH >= imgW

        import ipdb; ipdb.set_trace()
        transform = resizeNormalize((imgW, imgH))
        images = [transform(image) for image in images]
        images = torch.cat([t.unsqueeze(0) for t in images], 0)

        return images, labels
