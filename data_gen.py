#!/usr/bin/env python3

from utils import readFile, writeFile
from string_converter import encodeStr, decodeStr
import re
import numpy as np
import os
from PIL import Image
from dataset import TextDataset, extractWords




zwjMapping = {
        'ല്‍': 'ൽ',
        'ന്‍': 'ൻ',
        'ണ്‍': 'ൺ',
        'ര്‍': 'ർ',
        'ക്‍': 'ൿ',
        }
zwnjChilluRe = re.compile( '(' + '|'.join(zwjMapping.keys()) + ')' )
zwnjRe = re.compile('[‌‍]')


class DataGen:

    def __init__(self, infile, outfile ):
        self.WORD_LIST_FILE = infile
        self.DATA_FILE = outfile


    #  Pre-process input text file.
    #  Pre-processing includes the following steps
    #  * Convert ZWNJ based chillu to atomic chillu.
    #  * Remove/replace un-necessary chars
    #  * Then try encode each word. If it fails becuase of the presents of ZWJ char, then remove that ZWJ
    def preProcess( self ):
        txtFile = self.WORD_LIST_FILE
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
        writeFile( txtFile, '\n'.join( goodWords ))




    # function for self testing.
    # Encode each wrod, then decode it back
    def testEncoding( self ):
        words = extractWords( self.WORD_LIST_FILE )
        for w in words:
            try:
                enc, encSize = encodeStr( w )
            except Exception as e:
                print('Error encoding "%s"' % w, e )
                continue
            dec  = decodeStr( enc )
            if( w != dec ):
                raise ValueError('Encoding failed "%s" != "%s"'% ( w, dec ) )



    def createDataset( self, opts ):
        dataset = TextDataset( self.WORD_LIST_FILE )
        print( 'Dataset length=%d'%len(dataset))
        if( opts.format == 'numpy' ):
            np.savez_compressed( self.DATA_FILE, list( dataset ))
        else:
            os.system('mkdir -p %s' % self.DATA_FILE )
            for idx in range(len(dataset)):
                img, label = dataset[idx]
                img = Image.fromarray( img[0] )
                img.save( '%s/%s__%3d.png' %( self.DATA_FILE, label, idx ) )







def main( opt ):
    dg = DataGen( opt.input, opt.output )
    if( opt.preprocess ):
        print('Pre-processing' )
        dg.preProcess()

    if( opt.testencoding ):
        print('Testing encodability of  dataset' )
        dg.testEncoding()

    if( not opt.skip_creation ):
        print( 'Create dataset' )
        dg.createDataset( opt )
    print('Completed generating traindata for dataset\n\n' )



if( __name__ == '__main__' ):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--preprocess', action='store_true', help='preprocess wordlist and save it back to disk')
    parser.add_argument('--testencoding', action='store_true', help='do encodability test on each workd in the wordlist')
    parser.add_argument('--skip-creation', action='store_true', help='Skip dataset creation')
    parser.add_argument('--input', help='input text file contains words')
    parser.add_argument('--output', help='output numpy data file')
    parser.add_argument('--format', choices=[ 'numpy', 'images' ], default='numpy', help='Format of output. Numpy array vs Directory of images' )
    parser.add_argument('--name', help='name of dataset. ( Ie, input=./data/<name>.txt , output=./data/<name>_data.npz )')
    opt = parser.parse_args()
    if( opt.name ):
        opt.input = './data/%s.txt' % opt.name
        opt.output = './data/%s_data' % opt.name
    elif( opt.input and opt.output ):
        pass
    else:
        parser.error("Either '--name' or both '--input' and '--output' need to be specified")

    main( opt )

