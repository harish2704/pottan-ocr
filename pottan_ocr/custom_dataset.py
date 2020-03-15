#!/usr/bin/env python

#  ഓം ബ്രഹ്മാർപ്പണം
#
#  File name: pottan_ocr/custom_dataset.py
#  Author: Harish.K<harish2704@gmail.com>
#  Copyright 2020 Harish.K<harish2704@gmail.com>
#  Date created: Sun Mar 15 2020 15:26:21 GMT+0530 (GMT+05:30)
#  Date last modified: Sun Mar 15 2020 15:26:21 GMT+0530 (GMT+05:30)
#  Python Version: 3.x


import glob
import json
import numpy as np
from random import choice
from zipfile import ZipFile
from PIL import Image
from keras.utils import Sequence
from keras import models

from pottan_ocr.utils import normaizeImg
from pottan_ocr import string_converter

F_LINES = 'lines.json'
F_IMG = 'image.png'


def resize_img( img, targetH, max_line_width ):
    origW, origH = img.size
    targetW = int( origW * targetH/origH )
    if( targetW > max_line_width ):
        targetW = max_line_width
    img = img.resize( (targetW, targetH ), Image.BILINEAR )
    return normaizeImg( np.array( img ), 2 )

def compute_pred_size( crnn, img_width ):
    config = crnn.get_config()
    input_shape = list( config['layers'][0]['config']['batch_input_shape'] )
    input_shape[2] = img_width
    config['layers'][0]['config']['batch_input_shape'] = tuple( input_shape )
    temp_model = models.Sequential.from_config( config )
    return temp_model.output_shape[2]

def get_bg_value( img ):
    """predict background value by looking at most repeated value in first row of pixels"""
    values, counts = np.unique( np.squeeze(img)[0], return_counts=True)
    return values[np.argmax( counts) ]

def read_from_zip( zipfname, fname, parser_fn ):
    with ZipFile(zipfname) as zf:
        out = parser_fn(zf.open(fname))
    return out

class CustomDataset(Sequence):

    def __init__(self, dir_name, crnn_model, repeat=10, line_height=24 ):
        self.dir_name = dir_name
        self.line_height = line_height
        self.datafiles = glob.glob( dir_name + '/*.zip' )
        self.orig_item_count = len( self.datafiles )
        self.item_count = self.orig_item_count * repeat
        self.repeat = repeat

        # initialize database stats
        max_line_widths = []
        max_text_lengths = []
        total_lines=0
        for zipfname in self.datafiles:
            lines = read_from_zip( zipfname, F_LINES, json.load )
            max_line_widths.append( max( [  i['line']['w'] for i in lines ] ) )
            max_text_lengths.append( max( [ len(i['text']) for i in lines ] ))
            total_lines = total_lines + len( lines )
        self.max_line_width = max( max_line_widths )
        self.max_text_length = max( max_text_lengths )
        self.total_lines = total_lines
        self.prediction_size = compute_pred_size( crnn_model, self.max_line_width )
        self.pad_range = range(-1,3)


    def __len__(self):
        return self.item_count


    def get_batch( self, batch_index):
        file_index = batch_index % self.orig_item_count
        zipfname = self.datafiles[ file_index ];
        lines = read_from_zip( zipfname, F_LINES, json.load )
        image = read_from_zip( zipfname, F_IMG, Image.open )
        image = image.convert('L')

        pad_top = 0
        pad_bottom = 0
        if( batch_index > self.orig_item_count ):
            pad_top = choice( self.pad_range )
            pad_bottom = choice( self.pad_range )

        img_lines = []
        text_lines = []
        for line in lines:
            box = line['line']
            text = line['text']
            cords = [ box['x'], box['y']-pad_top, box['x'] + box['w'], box['y'] + box['h'] + pad_bottom ]
            img_line = image.crop( cords )

            img_line = resize_img( img_line, self.line_height, self.max_line_width )
            bg_value = get_bg_value( img_line )
            img_line = np.pad( img_line, [(0, 0), (0, self.max_line_width - img_line.shape[1] ), (0,0)], mode='constant', constant_values=bg_value )
            img_lines.append( img_line )
            text_lines.append( text )
        return ( img_lines, text_lines )


    def __getitem__(self, batchIndex):
        images, labels = self.get_batch( batchIndex )
        batchSize = len( labels );
        labels, label_lengths  = string_converter.encodeStrListRaw( labels, self.max_text_length )
        inputs = {
                'the_images': np.array(images),
                'prediction_lengths': np.array( [ self.prediction_size ] * batchSize ),
                'the_labels': np.array( labels ),
                'label_lengths': np.array( label_lengths ),
                }
        outputs = {'ctc': np.zeros([ batchSize ])}  # dummy data for dummy loss function
        return (inputs, outputs)




