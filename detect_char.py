
import sys
import numpy as np
import torch
from torch.autograd import Variable
from model import Net
import torch.nn.functional as F
import cv2
#  from debug_utils import imshow

from utils import readJson

glyphs = readJson( './cache/glyph_labels.json')
SIZE=32

model = Net()
model.load_state_dict( torch.load( './cache/model-state', map_location=lambda storage, loc: storage ) )
model.cpu()
model.eval()


def detectGlyph( img ):
    if( img.ndim == 2 ):
        img = np.array( [ img ] )
    img = np.expand_dims( img.astype( np.float32 ), axis=1 )
    tensor = Variable( torch.from_numpy( img ), volatile=True)
    out = model( tensor )


    ch = out.data.max(1, keepdim=True)
    return [ glyphs[i][0] for i in ch[1][:,0] ]

    #  import ipdb; ipdb.set_trace()
    #  confidence, keys = out.data.topk(15)
    #  keys = [ glyphs[ i ][0] for i in keys[0] ]
    #  return keys[0], zip( keys, confidence )
    #  for v, k in zip( vals[0].cpu(), keys[0] ):
        #  print( "%s => %f"% ( glyphs[ k ], v ) )

def main( fname ):
    img = cv2.imread( fname, cv2.IMREAD_GRAYSCALE )
    img = cv2.resize( img, ( SIZE, SIZE ), interpolation=cv2.INTER_CUBIC )
    glyph, confList = detectGlyph( img )
    print( glyph )

if( __name__ == '__main__' ):
    main( sys.argv[1] )
