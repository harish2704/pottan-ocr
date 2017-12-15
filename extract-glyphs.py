#!/usr/bin/env python3

from GlyphExtractor import GlyphExtractor
from utils import readJson
import os
from os import path
import cv2

import sys

def main( fname, ourDir ):
    gd = GlyphExtractor( fname )
    contours = gd.detectContours( rect=(2,2) )
    gd.extractContours( contours, ourDir )

if __name__ == "__main__":
    main( sys.argv[1],  sys.argv[2] )

