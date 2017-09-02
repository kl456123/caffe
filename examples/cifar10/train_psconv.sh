#!/bin/bash

LOG=examples/cifar10/log/train-`date +%Y-%m-%d-%H-%M-%S`.log
CAFFE=~/mycaffe/build/tools/caffe

$CAFFE train --solver=examples/cifar10/cifar10_quick_solver_lr1.prototxt --gpu 0,1 2>&1 | tee $LOG

