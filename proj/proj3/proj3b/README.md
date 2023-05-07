# ECE 786 Project 3B

Here are some useful commands to consider when running the project

```sh
make clean && make && ./quamsimV3 input/input_for_qc7_q0_q2_q3_q4_q5_q6.txt 
```

## First copy over the files from host to docker container

```sh
sudo docker cp proj3b e2a6a1da2cc1:/root
```

## Then run the following commands inside the docker container

```sh
cd /root/proj3b
cp ~/gpgpu-sim_distribution/configs/tested-cfgs/SM7_QV100/* .
make clean && make && ./quamsimV1 input/input_for_qc7_q0_q2_q3_q4_q5_q6.txt 
```
