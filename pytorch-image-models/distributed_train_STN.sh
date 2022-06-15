#!/bin/bash
NUM_PROC=$1
shift
python3 -m torch.distributed.launch --nproc_per_node=$NUM_PROC --master_port 46667 train_add_STN.py "$@"

