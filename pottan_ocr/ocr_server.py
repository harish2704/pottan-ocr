#!/usr/bin/env python
#  ഓം ബ്രഹ്മാർപ്പണം

#
#  File name: pottan_ocr/ocr_server.py
#  Author: Harish.K<harish2704@gmail.com>
#  Copyright 2020 Harish.K<harish2704@gmail.com>
#  Date created: Mon Mar 09 2020 01:29:47 GMT+0530 (GMT+05:30)
#  Date last modified: Mon Mar 09 2020 01:29:47 GMT+0530 (GMT+05:30)
#  Python Version: 3.x




from PIL import Image
import argparse
import numpy as np
from pyquery import PyQuery as pq
from pyquery.text import extract_text

from pottan_ocr.utils import config, readFile, writeFile
from pottan_ocr.dataset import normaizeImg
from pottan_ocr.string_converter import decodeStr
from pottan_ocr import utils
import subprocess
from datetime import datetime
import os

from flask import request, Flask, jsonify
import uuid


TEMP_DIR='ocr_server_temp'


def resize_img( img, targetH ):
    origW, origH = img.size
    targetW = int( origW * targetH/origH )
    img = img.resize( (targetW, targetH ), Image.BILINEAR )
    return normaizeImg( np.array( img ), 2 )


def ocr_images( images ):
    global model
    maxWidth = max([i.shape[1] for i in images ])
    images = [ np.pad( i, [(0, 0), (0, maxWidth - i.shape[1] ), (0,0)], mode='constant', constant_values=1) for i in images ]
    out =  model.predict( np.array(images) )
    out = out.argmax(2)
    textResults = [ decodeStr( i, raw=False ) for i in out ]
    return textResults


class OcrTask:
    """Represents a single OCR request task"""
    def __init__( self ):
        self.reqId = uuid.uuid4()
        self.imageFile = '%s/%s.png' % ( TEMP_DIR, self.reqId )
        self.hocrFile = '%s/%s.hocr' % ( TEMP_DIR, self.reqId )

    def log( self, msg ):
        now = datetime.now()
        print(' %s [%s] %s' % ( now, self.reqId, msg ), flush=True )

    def cleanFiles(self):
        os.remove(self.imageFile)
        os.remove(self.hocrFile)

    def getLineSegs( self ):
        self.log('getLineSegs')
        subprocess.run(['tesseract','-c', 'tessedit_create_hocr=1', self.imageFile, self.hocrFile[:-5]])
        hocr =  readFile( self.hocrFile )
        dom = pq( hocr.encode('utf-8') )
        lineSegs = []
        for el in dom('.ocr_line'):
            if extract_text(el).strip():
                title = el.get('title');
                #  cords – a 4-tuple defining the left, upper, right, and lower pixel coordinate.
                cords = [ int(i) for i in title.split(';')[0].split(' ')[1:] ]
                lineSegs.append( cords )
        return lineSegs

    def getOcrResult( self, lineSegs ):
        global lineHeight
        self.log('getOcrResult')
        img = Image.open( self.imageFile ).convert('L')
        img_lines = []
        for cords in lineSegs:
            img_line = img.crop( cords )
            img_line = resize_img( img_line, lineHeight )
            img_lines.append( img_line )
        ocr_res = ocr_images( img_lines )
        return ocr_res

app = Flask(__name__)

@app.route('/ocr', methods=['POST'])
def do_ocr():
    f = request.files['image']
    ocrTask = OcrTask()
    f.save( ocrTask.imageFile )
    lineSegs = ocrTask.getLineSegs()
    ocrResult = ocrTask.getOcrResult( lineSegs )
    ocrTask.cleanFiles()
    return jsonify( lines=lineSegs, text=ocrResult )


if( __name__ == '__main__' ):
    parser = argparse.ArgumentParser()
    parser.add_argument('--crnn', required=True, help="path to pre trained model ( Keras saved model (.h5 file) )")
    opt = parser.parse_args()
    from keras import models
    model = models.load_model( opt.crnn )
    lineHeight = model.input.shape[1].value
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)
    app.run(host='0.0.0.0', port=5544)
