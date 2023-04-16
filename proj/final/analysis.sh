#!/bin/bash

# NOTE: For this script to work, you must have the same directory structure:
# ls ~
# ISPASS  
# analyis.sh  
# env  
# gpgpu-sim_distribution  
# rodinia

cd /root/rodinia


# Change config based on GPU
# SM2_GTX480
CONFIG="SM7_QV100"

cd BP
cp ~/gpgpu-sim_distribution/configs/tested-cfgs/$CONFIG/* .
./backprop-rodinia-3.1 65536 > gpu_result.txt
cd ..


cd HS
cp ~/gpgpu-sim_distribution/configs/tested-cfgs/$CONFIG/* .
./hotspot-rodinia-3.1 512 2 2 ./data/temp_512 ./data/power_512 output.out > gpu_result.txt
cd ..

cd LUD
cp ~/gpgpu-sim_distribution/configs/tested-cfgs/$CONFIG/* .
./lud-rodinia-3.1 -s 256 -v > gpu_result.txt
cd ..

cd /root/ISPASS

cd BFS
cp ~/gpgpu-sim_distribution/configs/tested-cfgs/$CONFIG/* .
./ispass-2009-BFS graph65536.txt > gpu_result.txt
cd ..


cd LPS
cp ~/gpgpu-sim_distribution/configs/tested-cfgs/$CONFIG/* .
./ispass-2009-LPS > gpu_result.txt
cd ..


cd NQU
cp ~/gpgpu-sim_distribution/configs/tested-cfgs/$CONFIG/* .
./ispass-2009-NQU > gpu_result.txt
cd ..