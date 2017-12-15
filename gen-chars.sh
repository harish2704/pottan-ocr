#!/usr/bin/env bash

varamozhiConverter='../varamozhi/lamsrc/lamvi_unicode' 

cache=./cache
glypListFile=$cache/glyphs.txt
glypLabelListFile=$cache/glyphs-labels.txt
glypLabelMaping=$cache/glyph_labels.json

# generate all possible combination of glyphs.
# Glyph can have one or more unicode charectors. eg: 'Nta' is a glyph. 'koo' in old lipi is a glyph
./gen-glyph-list.js  > $glypListFile


# Generate human readable labels for each glyph using varamozhi
cat $glypListFile | $varamozhiConverter > $glypLabelListFile

# Create a mapping b/w glyph -> label for future use
./split-glyph-list.js $glypListFile $glypLabelListFile $glypLabelMaping


# for font in AnjaliOldLipi Rachana NotoSerifMalayalam-Regular Meera Kalyani; do
  # convert -font $font -colorspace gray -pointsize 48  pango:@$glypListFile $cache/$font-regular.png
  # convert -font $font -colorspace gray -pointsize 48  pango:"<i>$(cat $glypListFile)</i>" $cache/$font-italic.png
  # convert -font $font -colorspace gray -pointsize 48  pango:"<b>$(cat $glypListFile)</b>" $cache/$font-bold.png
  # convert -font $font -colorspace gray -pointsize 48  pango:"<b><i>$(cat $glypListFile)</i></b>" $cache/$font-bolditalic.png
  # for kind in regular italic bold bolditalic; do
    # ./extract-glyphs.py cache/$font-$kind.png cache/generated/
  # done
# done

# for font in Kalyani; do
  # convert -font $font -colorspace gray -pointsize 48  pango:"<b>$(cat $glypListFile)</b>" $cache/$font-bold.png
  # for kind in bold; do
    # ./extract-glyphs.py cache/$font-$kind.png cache/generated/
  # done
# done


# exit

# rm ./ml-glyphs/text/*
# node gen-train.js


# cd ./ml-glyphs/text

# dirName="../anjali"
# mkdir -p $dirName
# j=0
# for i in $(ls *.txt); do
  # label=${i%.*}
  # # dirname="../img/$label"
  # # mkdir -p $dirName
  # convert -font AnjaliOldLipi -colorspace gray -pointsize 48 -gravity center  pango:@$i $dirName/$label.png
  # (( j++ ))
  # echo $j
  # if [ "$j" = 30 ]; then
      # break
  # fi
# done
