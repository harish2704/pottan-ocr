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
        'ര്‍': 'ർ'
        }
zwnjChilluRe = re.compile( '(' + '|'.join(zwjMapping.keys()) + ')' )
zwnjRe = re.compile('‌')



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
    words = re.sub('[”“]', '"', words )
    # Rm English letters
    words = re.sub('[a-zA-Z]', '', words )

    words = filter( None, re.split('[\n\ ]', words ) )
    words = list(set( words ))
    for idx, w in enumerate(words):
        try:
            encodeStr( w )
        except Exception as e:
            if( e.args[0] == "'\\u200c' is not in list" ):
                print('Replacing zwj in "%s"' % w )
                words[idx] = zwnjRe.sub('', w )
            else:
                raise e
    writeFile( txtFile, ' '.join( words ))



#  Generate image as nupy array for a unicode text.
#  We are using fixed width because torch.DataLoader expect a fixed size array
# scribe function will fill the extra space with white/black color
def computeDataset( words, font='AnjaliOldLipi' ):
    imgs = []
    labels = []
    for word in words:
        img = scribe( word, '%s 19' % font, 400, 32, 0, 3, 0 )
        #  import ipdb; ipdb.set_trace()
        img = np.invert( img )
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
def testEncoding( kind='validate'):
    words = extractWords( WORD_LIST_FILE % kind )
    for w in words:
        enc, encSize = encodeStr( w )
        dec  = decodeStr( enc )
        if( w != dec ):
            raise ValueError('%s != %s'% ( w, dec ) )


def main( kind ):
    print('Pre-processing %s' % kind )
    preProcess( kind )

    print('Testing encodability of  "%s" dataset' % kind )
    testEncoding( kind )

    print( 'Create dataset %s' % kind )
    createDataset( kind )
    print('Completed generating traindata for "%s" dataset\n\n' % kind )



if( __name__ == '__main__' ):
    main( 'validate' )
    main( 'train' )
