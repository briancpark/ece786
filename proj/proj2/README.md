# Project 2: GPGPU-SIM

Run Docker container:

```sh
docker run -w /root -it pli11/gpgpusim:cuda_10_1 /bin/bash
```

If reconnecting to an already running container:

```sh
docker exec -it ContainerID /bin/bash
```

## Task 1

Copy instructions.cc to `/root/gpgpusim_distribution/src/cuda-sim/` directory

```sh
cd /root/gpgpusim_distribution
make -j
source setup_environment 
cd /root/vectorAdd
make -j
cp ~/gpgpu-sim_distribution/configs/tested-cfgs/SM7_QV100/* .
./vectorAdd
```

## Task 2

Run benchmarks as described in the project description. Then run the following commands to generate the plots (it doesn't have to work inside the Docker container once you obtain the data):

```sh
python3 plot.py
```

## Task 3

Run the following:

```sh
cd ~/gpgpu-sim_distribution
make clean
make -j
cd ~/ISPASS/LPS
./ispass-2009-LPS > result.txt
```

## Task 4

Run the same commands, but with task 4 source files copied over.
