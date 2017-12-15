#!/usr/bin/env bash
IFS=$'\n'
files=$(cat ./img-list)

for i in $files; do
  echo Processing $i;
  ./test.py "$i" ./ml-glyphs/from-pages/ ;
done
