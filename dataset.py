
import torch
from torch.utils.data import Dataset
from data_gen import extractWords, renderText



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

    def __len__(self):
        return len( self.words )

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        label = self.words[index]
        img = renderText( label )
        return ( img, label)
