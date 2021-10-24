bsub -P gpu -M 16000 -gpu "num=1:j_exclusive=yes" sh train.sh 0.25
bsub -P gpu -M 16000 -gpu "num=1:j_exclusive=yes" sh train.sh 0.5
bsub -P gpu -M 16000 -gpu "num=1:j_exclusive=yes" -m "gpu-009" sh train.sh 1.0
