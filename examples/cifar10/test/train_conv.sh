#!/bin/bash

LOG=Conv/train-`date +%Y-%m-%d-%H-%M-%S`.log
CAFFE=~/mycaffe/build/tools/caffe

$CAFFE train --solver=./Conv_cifar10_quick_solver.prototxt --gpu 0,1 2>&1 | tee $LOG

