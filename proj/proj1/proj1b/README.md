# Proj1B


```sh
docker pull pli11/gpgpusim:cuda_10_1
docker run -w /root -it pli11/gpgpusim:cuda_10_1 /bin/bash

# Within the docker container
cd ~/gpgpu-sim_distribution/
git checkout tags/v4.0.1 -b v401

cd ~/gpgpu-sim_distribution/

make clean
source setup_environment
make

cd

git clone https://github.com/peiyi1/vectorAdd.git
cd ~/vectorAdd/
make clean
make

cd ~/vectorAdd/
cp ~/gpgpu-sim_distribution/configs/tested-cfgs/SM7_QV100/* .
./vectorAdd > output.txt

```