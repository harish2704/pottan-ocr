import json
import numpy as np
import gzip
import yaml
from re import split


def showImg( im ):
    from matplotlib import pyplot
    pyplot.imshow( im )
    pyplot.show()

def normaizeImg( img, channel_axis=0 ):
    img = img.astype('f')
    img = ( img - 127.5 ) / 127.5
    img = np.expand_dims( img, channel_axis )
    return img

def normalizeBatch( batch, channel_axis=0 ):
    images, labels = zip(*batch)
    images = [ normaizeImg( image, channel_axis ) for image in images]
    images = np.stack( images )
    return images, np.array( labels )

def myOpen( fname, mode ):
    return open( fname, mode, encoding="utf-8" )

def readFile( fname ):
    opener, mode = ( gzip.open, 'rt' ) if fname[-3:] == '.gz' else ( open, 'r' )
    with opener( fname, mode ) as f:
        return f.read()

def readLines( fname ):
    return split('[\r\n]', readFile( fname ) )

def readJson( fname ):
    with myOpen( fname, 'r' ) as f:
        return json.load( f )

def writeFile( fname, contents ):
    with myOpen( fname, 'w' ) as f:
        f.write( contents )

def writeJson( fname, data ):
    with myOpen( fname, 'w') as outfile:
        json.dump(data, outfile)

def readYaml( fname ):
    with myOpen(fname, 'r') as fp:
        return yaml.load( fp )


config = readYaml('./config.yaml')


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



def loadTrainedModel( model, opt ):
    """Load a pretrained model into given model"""

    print('loading pretrained model from %s' % opt.crnn)
    if( opt.cuda ):
        stateDict = torch.load(opt.crnn )
    else:
        stateDict = torch.load(opt.crnn, map_location={'cuda:0': 'cpu'} )

    #  Handle the case of some old torch version. It will save the data as module.<xyz> . Handle it
    if( list( stateDict.keys() )[0][:7] == 'module.' ):
        for key in list(stateDict.keys()): 
            stateDict[ key[ 7:] ] = stateDict[key]
            del stateDict[ key ]
    model.load_state_dict( stateDict )
    print('Completed loading pre trained model')
