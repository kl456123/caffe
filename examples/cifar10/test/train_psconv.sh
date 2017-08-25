#!/bin/bash

LOG=PSConv/train-`date +%Y-%m-%d-%H-%M-%S`.log
CAFFE=~/caffe/build/tools/caffe

$CAFFE train --solver=./PSConv_cifar10_quick_solver.prototxt 2>&1 | tee $LOG

