#!/bin/bash

LOG=Conv/train-`date +%Y-%m-%d-%H-%M-%S`.log
CAFFE=~/caffe/build/tools/caffe

$CAFFE train --solver=./Conv_cifar10_quick_solver.prototxt 2>&1 | tee $LOG

