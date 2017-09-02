#!/usr/bin/env sh
set -e

TOOLS=../../../build/tools
LOG=quickConv/train-`date +%Y-%m-%d-%H-%M-%S`.log

$TOOLS/caffe train \
    --solver=cifar10_full_solver.prototxt \
    --gpu 0,1\
    2>&1 | tee $LOG

## reduce learning rate by factor of 10
$TOOLS/caffe train \
    --solver=cifar10_full_solver_lr1.prototxt \
    --snapshot=quickConv/cifar10_quick_iter_60000.solverstate.h5 \
    --gpu 0,1\
    2>&1 | tee $LOG

# reduce learning rate by factor of 10
$TOOLS/caffe train \
    --solver=cifar10_full_solver_lr2.prototxt \
    --snapshot=quickConv/cifar10_quick_iter_65000.solverstate.h5 \
    --gpu 0,1 \
     2>&1 | tee $LOG
