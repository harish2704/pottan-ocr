
import torch
from torch.utils.data import Dataset
from data_gen import extractWords, renderText
from utils import readYaml


fontList = readYaml('./fontlist.yaml')
fontListFlat = []
for fnt, styles in fontList:
    for style in styles:
        fontListFlat.append([ fnt, style ])

totalVariations = len(fontListFlat)


def normaizeImg( img ):
    img = torch.FloatTensor( img.astype('f') )
    img = ((img*2)/255 ) -1
    return img


def alignCollate( batch ):
    images, labels = zip(*batch)
    images = [ normaizeImg(image) for image in images]
    images = torch.cat([t.unsqueeze(0) for t in images], 0)
    return images, labels




class TextDataset(Dataset):

    def __init__(self, txtFile):
        self.txtFile = txtFile
        self.words = extractWords( txtFile )
        self.itemCount = len( self.words )*totalVariations

    def __len__(self):
        return self.itemCount

    def __getitem__(self, index):
        assert index <= self.itemCount, 'index range error'
        wordIdx = int( index / totalVariations )
        font, style = fontListFlat[ index % totalVariations ]
        label = self.words[ wordIdx ]
        img = renderText( label, font=font, style=style )
        return ( img, label)
