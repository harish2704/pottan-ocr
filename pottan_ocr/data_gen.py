#!/usr/bin/env python3

import re
import numpy as np
import os
from PIL import Image
import multiprocessing
import random


from pottan_ocr.string_converter import encodeStr, decodeStr
from pottan_ocr.dataset import TextDataset, getTrainingTexts
from pottan_ocr.utils import readFile, writeFile

def processInThread( i ):
    datasetInOtherthread.__getitem__(i)

def threadInitializer( fname, batchSize, cache, limit  ):
    global datasetInOtherthread
    datasetInOtherthread = TextDataset( fname, batchSize=batchSize, cache=cache, limit=limit, overwriteCache=True )


class DataGen:

    def __init__(self, infile, outfile ):
        self.WORD_LIST_FILE = infile
        self.DATA_FILE = outfile



    def testEncoding( self, opt ):
        """
        function for self testing.
        Encode each wrod, then decode it back
        """
        goodWords = []
        words = getTrainingTexts( self.WORD_LIST_FILE )
        for w in words:
            try:
                enc, encSize = encodeStr( w )
                goodWords.append( w )
            except Exception as e:
                print('Error encoding "%s"' % w, e )
                continue
            dec  = decodeStr( enc )
            if( w != dec ):
                raise ValueError('Encoding failed "%s" != "%s"'% ( w, dec ) )
        if( opt.update ):
            writeFile( self.WORD_LIST_FILE, '\n'.join( goodWords ))



    def createDatasetSingleThread( self, opts ):
        """ Used for debugging . It is hard to debug a multi-threaded application """
        dataset = TextDataset( self.WORD_LIST_FILE, batchSize=opts.batchSize, cache=opts.output, limit=opts.count, overwriteCache=True )
        results = [ dataset.__getitem__(i) for i in range( len( dataset )) ]
        print( 'Total lines count=%d' % ( len( dataset )*opts.batchSize ) )
        #  dataset.printStats()

    def createDataset( self, opts ):
        dataset = TextDataset( self.WORD_LIST_FILE, batchSize=opts.batchSize, cache=opts.output, limit=opts.count, overwriteCache=True)
        pool = multiprocessing.Pool( 1, initializer=threadInitializer, initargs=( self.WORD_LIST_FILE, opts.batchSize, opts.output, opts.count  ) )
        results = [ pool.apply_async( processInThread, ( i, )  ) for i in range( len( dataset )) ]
        print( 'Total lines count=%d' % ( len( dataset )*opts.batchSize ) )
        for idx, result in enumerate(results):
            result.get()




def main( opt ):
    dg = DataGen( opt.input, opt.output )

    if( opt.testencoding ):
        print('Testing encodability of  dataset' )
        dg.testEncoding( opt )

    if( not opt.skip_creation ):
        print( 'Creating dataset' )
        #  dg.createDatasetSingleThread( opt )
        dg.createDataset( opt )
    print('Completed generating traindata for dataset\n\n' )



if( __name__ == '__main__' ):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--testencoding', action='store_true', help='do encodability test on each word in the wordlist')
    parser.add_argument('--update', action='store_true', help='Update input file with valid data after testencoding')
    parser.add_argument('--skip-creation', action='store_true', help='Skip dataset creation')
    parser.add_argument('--input', help='input text file contains words')
    parser.add_argument('--output', help='output numpy data file')
    parser.add_argument('--count', type=int, default=512, help='size of the dataset')
    parser.add_argument('--batchSize', type=int, default=64, help='traindata batch size')
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

