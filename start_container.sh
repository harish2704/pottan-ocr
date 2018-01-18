#!/usr/bin/env bash

floyd run  --gpu --env pytorch-0.3 \
  --data harish2704/datasets/pottan-ocr/1:/basedeps \
  --data harish2704/datasets/pottan-ocr/2:/datadeps \
  "bash bootstrap.sh"
