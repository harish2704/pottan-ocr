#!/usr/bin/env bash

cd ./ml-glyphs/img

for i in $(ls); do
  mv ../from-pages/${i}_* ./$i/
done
