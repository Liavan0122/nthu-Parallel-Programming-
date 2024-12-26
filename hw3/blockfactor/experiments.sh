#!/bin/bash

DATA=/home/pp24/pp24s065/hw3-2/blockfactor/hw3-2-B
IN=/home/pp24/share/hw3-2/testcases/c21.1
OUT=./output/B
NVPROF_METRIC=achieved_occupancy,sm_efficiency,shared_load_throughput,shared_store_throughput,gld_throughput,gst_throughput

# Compile code
make

for i in {16,32,64};
do
    echo -e "\n------------- Blocking Factor: $i -------------------"
    echo -e "\n########## Time ##########"
    srun -p nvidia -N1 -n1 --gres=gpu:1 nvprof $DATA$i $IN $OUT
    echo -e "\n########## Metric ##########"
    srun -p nvidia -N1 -n1 --gres=gpu:1 nvprof -m $NVPROF_METRIC $DATA$i $IN $OUT$i-c21.1.out
done

make clean