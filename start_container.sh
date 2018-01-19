#!/usr/bin/env bash

floyd run  --gpu --env pytorch-0.3 \
  --data harish2704/datasets/pottan-ocr/1:/basedeps \
  --data harish2704/datasets/pottan-ocr/2:/datadeps \
  --data harish2704/projects/testbed/6/output:/presession \
  --data harish2704/projects/pottan-ocr/2/output:/presession2 \
  "bash bootstrap.sh netCRNN_01-18-21-01-10_1.pth"
