#!/usr/bin/env python3

from utils import readFile, writeFile
from string_converter import encodeStr, decodeStr
import re

from unicode_text_to_image_array.scribe import scribe
import numpy as np

WORD_LIST_FILE = './data/%s.txt'
DATA_FILE = './data/%s_data.npz'

# fonts=[
#         ('AnjaliOldLipi', ['regular', 'bold' ]),
#         ('Chilanka', ['regular', 'bold', 'italic' ]),
#         ('Dyuthi', ['regular', 'bold', 'italic', 'bold italic']),
#         ('Kalyani', ['regular', 'bold', 'italic', 'bold italic']),
#         ('Karumbi', ['regular', 'bold', 'italic', 'bold italic']),
#         #  It is already too tick. Rm bold
#         ('Keraleeyam', ['regular', 'italic' ]),
#         ('Lohit Malayalam', ['regular', 'bold', 'italic', 'bold italic']),
#         ('Manjari', ['regular', 'bold', 'italic', 'bold italic']),
#         ('Manjari,Manjari Thin', ['regular', 'italic']),
#         ('Meera', ['regular', 'bold', 'italic', 'bold italic']),
#         ('ML-NILA01', ['regular', 'bold', 'italic', 'bold italic']),
#         ('ML-NILA02', ['regular', 'bold', 'italic', 'bold italic']),
#         ('ML-NILA03', ['regular', 'bold', 'italic', 'bold italic']),
#         ('ML-NILA04', ['regular', 'bold', 'italic', 'bold italic']),
#         ('ML-NILA05', ['regular', 'bold', 'italic', 'bold italic']),
#         ('ML-NILA06', ['regular', 'bold', 'italic', 'bold italic']),
#         ('ML-NILA07', ['regular', 'bold', 'italic', 'bold italic']),
#         ('Noto Sans Malayalam', ['regular', 'bold', 'italic', 'bold italic']),
#         ('Noto Sans Malayalam UI', ['regular', 'bold', 'italic', 'bold italic']),
#         ('Noto Serif Malayalam', ['regular', 'bold', 'italic', 'bold italic']),
#         ('Rachana', ['regular', 'bold', 'italic', 'bold italic']),
#         ('RaghuMalayalam', ['regular', 'bold', 'italic', 'bold italic']),
#         ('Samyak Malayalam', ['regular', 'bold', 'italic', 'bold italic']),
#         ('Suruma', ['regular', 'bold', 'italic', 'bold italic']),
#         #  It is already too tick. Rm bold
#         ('Uroob', ['regular', 'italic' ])
#         ]
#

def extractWords( txtFile ):
    words = readFile( txtFile )
    words = filter( None, re.split('[\n\ ]', words ) )
    return list(set( words ))



zwjMapping = {
        'ല്‍': 'ൽ',
        'ന്‍': 'ൻ',
        'ണ്‍': 'ൺ',
        'ര്‍': 'ർ',
        'ക്‍': 'ൿ',
        }
zwnjChilluRe = re.compile( '(' + '|'.join(zwjMapping.keys()) + ')' )
zwnjRe = re.compile('[‌‍]')



#  Pre-process input text file.
#  Pre-processing includes the following steps
#  * Convert ZWNJ based chillu to atomic chillu.
#  * Remove/replace un-necessary chars
#  * Then try encode each word. If it fails becuase of the presents of ZWJ char, then remove that ZWJ
def preProcess( kind ):
    txtFile = WORD_LIST_FILE % kind
    words = readFile( txtFile )

    # Replace all zwnj chillus with atomic chillu
    words = zwnjChilluRe.sub( lambda g: zwjMapping[ g.group(0) ], words )

    words = filter( None, re.split('‌*[\n\ ]', words ) )
    words = list(set( words ))
    goodWords = []
    total = len( words )
    for idx, w in enumerate(words):
        if( idx % 10000 == 0 ):
            print( 'preProcess %4.2f %%' % ( idx*100/total ) )
        try:
            encodeStr( w )
            goodWords.append( w )
        except Exception as e:
            if( re.match( ".*u200[cd].* is not in list", e.args[0] ) ):
                #  print('Replacing zwj in "%s"' % w )
                goodWords.append( zwnjRe.sub('', w ) )
            else:
                pass
                #  print( 'Omiting "%s"' % w, e )
                #  import ipdb; ipdb.set_trace()
                #  raise e
    goodWords = list( filter( lambda x: len(x)< 27, goodWords ) )
    goodWords.sort(key=lambda x: len(x), reverse=True )
    writeFile( txtFile, '\n'.join( goodWords ))



#  Generate image as nupy array for a unicode text.
#  We are using fixed width because torch.DataLoader expect a fixed size array
# scribe function will fill the extra space with white/black color
def computeDataset( words, font='AnjaliOldLipi' ):
    imgs = []
    labels = []
    total = len( words )
    for idx, word in enumerate(words):
        if( idx % 10000 == 0 ):
            print( 'ComputeDataset %4.2f %%' % ( idx*100/total ) )
        img = scribe( word, '%s 16' % font, 400, 32, 0, 3, 0 )
        img = np.invert( img )
        #  import ipdb; ipdb.set_trace()
        # Convert into 1xWxH  
        img = np.expand_dims( img, axis=0 )
        imgs.append( img )
        labels.append( word )
    return imgs, labels



def createDataset( kind ):
    words = extractWords( WORD_LIST_FILE % kind )
    imgs, labels = computeDataset( words )
    np.savez_compressed(  DATA_FILE % kind, list(zip( imgs, labels )) )



# function for self testing.
# Encode each wrod, then decode it back
def testEncoding( kind='validate' ):
    words = extractWords( WORD_LIST_FILE % kind )
    for w in words:
        enc, encSize = encodeStr( w )
        dec  = decodeStr( enc )
        if( w != dec ):
            raise ValueError('Encoding failed "%s" != "%s"'% ( w, dec ) )


def main( opt ):
    kind = opt.name

    if( not opt.skip_preprocessing ):
        print('Pre-processing %s' % kind )
        preProcess( kind )

    if( not opt.skip_encodetest ):
        print('Testing encodability of  "%s" dataset' % kind )
        testEncoding( kind )

    if( not opt.skip_creation ):
        print( 'Create dataset %s' % kind )
        createDataset( kind )
    print('Completed generating traindata for "%s" dataset\n\n' % kind )



if( __name__ == '__main__' ):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-preprocessing', action='store_true', help='Skip preprocessing of wordlist')
    parser.add_argument('--skip-encodetest', action='store_true', help='Skip encodability test on wordlist')
    parser.add_argument('--skip-creation', action='store_true', help='Skip dataset creation')
    parser.add_argument('--name', required=True, help='name of dataset. ./data/<name>.txt should be present')
    opt = parser.parse_args()
    main( opt )
