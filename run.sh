SHOT=5
WAY=5
SPATH="proto-kl-5-5"
GPU=0

CUDA_CACHE_PATH=/st1/hayeon/tmp python train.py --shot $SHOT --train-way $WAY --save-path "./save/"$SPATH --gpu $GPU | tee "./log/"$WAY"w"$SHOT"s_"$SPATH".log"
