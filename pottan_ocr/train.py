#!/usr/bin/env python3


import os
import argparse
import random
from datetime import datetime
import torch
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from warpctc_pytorch import CTCLoss

from pottan_ocr import utils
from pottan_ocr import string_converter as converter
from pottan_ocr import model as crnn
from pottan_ocr.dataset import TextDataset

imageHeight = utils.config['imageHeight']


parser = argparse.ArgumentParser()
parser.add_argument('--traindata', required=True, help='path to dataset')
parser.add_argument('--traindata_limit', type=int, default=51200, help='Limit the training dataset size')
parser.add_argument('--traindata_cache', default=None, help='Cache Directory for caching generated train data')
parser.add_argument('--valdata', required=True, help='path to dataset')
parser.add_argument('--valdata_limit', type=int, default=4096, help='Limit the validation dataset size')
parser.add_argument('--valdata_cache', default=None, help='Cache Directory for caching generated validation data')
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--nh', type=int, default=256, help='size of the lstm hidden state')
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate for Critic, default=0.00005')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--crnn', default='', help="path to crnn (to continue training)")
parser.add_argument('--outdir', default=None, help='Where to store samples and models')
parser.add_argument('--displayInterval', type=int, default=500, help='Interval to be displayed')
parser.add_argument('--n_test_disp', type=int, default=10, help='Number of samples to display when test')
parser.add_argument('--valInterval', type=int, default=500, help='Interval to be displayed')
parser.add_argument('--saveInterval', type=int, default=500, help='Interval to be displayed')
parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
parser.add_argument('--adadelta', action='store_true', help='Whether to use adadelta (default is rmsprop)')
opt = parser.parse_args()
print(opt)

if opt.outdir is None:
    opt.outdir = 'expr'
os.system('mkdir -p {0}'.format(opt.outdir))

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)


if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")



train_loader = TextDataset( opt.traindata, batchSize=opt.batchSize, limit=opt.traindata_limit, cache=opt.traindata_cache )
test_loader = TextDataset( opt.valdata, batchSize=opt.batchSize, limit=opt.valdata_limit, cache=opt.valdata_cache )

nclass = converter.totalGlyphs
print('Number of char class = %d' % nclass )


criterion = CTCLoss()


# custom weights initialization called on crnn
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


#  1 --> Number of channels
crnn = crnn.CRNN( imageHeight, 1, nclass, opt.nh)
crnn.apply(weights_init)
if opt.crnn != '':
    utils.loadTrainedModel( crnn, opt )
#  print(crnn)

image = torch.FloatTensor(opt.batchSize, 3,  imageHeight,  imageHeight)
text = torch.IntTensor(opt.batchSize * 5)
length = torch.IntTensor(opt.batchSize)

if opt.cuda:
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True
    crnn.cuda()
    crnn = torch.nn.DataParallel(crnn, device_ids=range(opt.ngpu))
    image = image.cuda()
    criterion = criterion.cuda()

image = Variable(image)
text = Variable(text)
length = Variable(length)

# loss averager
loss_avg = utils.averager()

# setup optimizer
if opt.adam:
    optimizer = optim.Adam(crnn.parameters(), lr=opt.lr,
                           betas=(opt.beta1, 0.999))
elif opt.adadelta:
    optimizer = optim.Adadelta(crnn.parameters(), lr=opt.lr)
else:
    optimizer = optim.RMSprop(crnn.parameters(), lr=opt.lr)

def loadData(v, data):
    v.data.resize_(data.size()).copy_(data)

def val(net, criterion, max_iter=10):
    print('Validating...')

    for p in crnn.parameters():
        p.requires_grad = False

    net.eval()

    n_correct = 0
    loss_avg = utils.averager()

    for i, data in enumerate( test_loader ):
        cpu_images, cpu_texts = data

        batchSize = cpu_images.size(0)
        txts, lengths = converter.encode(cpu_texts)
        #  image = Variable( cpu_images )
        #  text = Variable( txts )
        #  length = Variable( lengths )
        loadData(image, cpu_images)
        loadData(length, lengths)
        loadData(text, txts)

        preds = crnn(image)
        preds_size = Variable(torch.IntTensor( [preds.size(0)] * preds.size(1) ))
        cost = criterion(preds, text, preds_size, length) / batchSize
        loss_avg.add(cost)

        _, preds = preds.max(2)
        #  preds = preds.squeeze(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
        for pred, target in zip(sim_preds, cpu_texts):
            if pred == target:
                n_correct += 1

    raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:opt.n_test_disp]
    for raw_pred, pred, gt in zip(raw_preds, sim_preds, cpu_texts):
        print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))

    accuracy = n_correct / float( len( test_loader ) * opt.batchSize)
    print('Test loss: %f, accuray: %f' % (loss_avg.val(), accuracy))


def trainBatch( data ):
    print('Training start')
    cpu_images, cpu_texts = data
    cpu_images = torch.from_numpy( cpu_images )
    #  cpu_texts = torch.from_numpy( cpu_texts )

    import ipdb; ipdb.set_trace()
    batchSize = cpu_images.size(0)
    txts, lengths = converter.encode(cpu_texts)

    #  image = Variable( cpu_images )
    #  text = Variable( txts )
    #  length = Variable( lengths )
    loadData(image, cpu_images)
    loadData(length, lengths)
    loadData(text, txts)

    preds = crnn(image)
    preds_size = Variable(torch.IntTensor( [preds.size(0)] * preds.size(1) ))
    cost = criterion(preds, text, preds_size, length) / batchSize
    crnn.zero_grad()
    cost.backward()
    optimizer.step()
    return cost

def saveModel( epoch=0 ):
    print('Saving model...' )
    fname = '{0}/netCRNN_{2}_{1}.pth'.format(opt.outdir, epoch, datetime.now().strftime('%m-%d-%H-%M-%S') )
    torch.save( crnn.state_dict(), fname )
    print('Saved model "%s"' % fname )


def main( epoch ):
    for i,batchData in enumerate(train_loader):
        for p in crnn.parameters():
            p.requires_grad = True
        crnn.train()

        cost = trainBatch( batchData )
        loss_avg.add(cost)
        i += 1

        if i % opt.displayInterval == 0:
            print('[ %s ][%d/%d][%d/%d] Loss: %f' %
                  ( datetime.now().strftime('%H-%M-%S'), epoch, opt.niter, i, len(train_loader), loss_avg.val()))
            loss_avg.reset()

        if i % opt.valInterval == 0:
            val(crnn, criterion)

        if i % opt.saveInterval == 0:
            saveModel( epoch )

for epoch in range(opt.niter):
    try:
        main( epoch )
    except KeyboardInterrupt as e:
        print('Exiting.....')
        break

saveModel()

