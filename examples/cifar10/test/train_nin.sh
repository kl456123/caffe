#!/bin/bash

LOG=psfullnin/train-`date +%Y-%m-%d-%H-%M-%S`.log
CAFFE=~/mycaffe/build/tools/caffe

$CAFFE train --solver=./solver.prototxt --gpu 0 2>&1 | tee $LOG


#$CAFFE train --solver=./solver_ps.prototxt --weight--gpu 0,1 2>&1 | tee $LOG

