#!/bin/bash

# Evaluation
sbatch eval.sh 377
sbatch eval.sh 386
sbatch eval.sh 387
#sbatch eval.sh 392
#sbatch eval.sh 393
#sbatch eval.sh 394

# Cam 12 video
sbatch video.sh 377 12
sbatch video.sh 386 12
sbatch video.sh 387 12
#sbatch video.sh 392 12
#sbatch video.sh 393 12
#sbatch video.sh 394 12

# OOD Pose
sbatch test.sh 377 0
sbatch test.sh 377 1
sbatch test.sh 377 2
sbatch test.sh 386 0
sbatch test.sh 386 1
sbatch test.sh 386 2
sbatch test.sh 387 0
sbatch test.sh 387 1
sbatch test.sh 387 2
#sbatch test.sh 392 0
#sbatch test.sh 392 1
#sbatch test.sh 392 2
#sbatch test.sh 393 0
#sbatch test.sh 393 1
#sbatch test.sh 393 2
#sbatch test.sh 394 0
#sbatch test.sh 394 1
#sbatch test.sh 394 2
