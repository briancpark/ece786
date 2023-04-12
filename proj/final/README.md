# ECE 786 Final Project

Here are some useful commands to consider when running the project
```sh
make clean && make && ./quamsimV3 input/input_for_qc7_q0_q2_q3_q4_q5_q6.txt 
```



## First copy over the files from host to docker container

```
sudo docker cp rodinia/ e2a6a1da2cc1:/root
```

docker exec -it e2a6a1da2cc1 /bin/bash

## Then run the following commands inside the docker container

```sh
cd /root/rodinia
cp ~/gpgpu-sim_distribution/configs/tested-cfgs/SM2_GTX480/* .
make clean && make && ./quamsimV1 input/input_for_qc7_q0_q2_q3_q4_q5_q6.txt 
```

