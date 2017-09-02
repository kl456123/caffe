#!/bin/bash

LOG=quickConv/train-`date +%Y-%m-%d-%H-%M-%S`.log
CAFFE=~/mycaffe/build/tools/caffe

$CAFFE train \
  --solver=cifar10_quick_solver.prototxt 2>&1 | tee $LOG

# reduce learning rate by factor of 10 after 8 epochs
$CAFFE train \
  --solver=cifar10_quick_solver_lr1.prototxt \
  --snapshot=quickConv/cifar10_quick_iter_4000.solverstate 2>&1 | tee $LOG


