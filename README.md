=Introduction
This is a modified version from the 'nccl examples' page at here:
https://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/index.html#examples.
This runs across various buffer sizes and notes down the transfer times.
Should be useful if you are trying to benchmark your system for multi-gpu
communications, especially using AllReduce primitives given by nccl library.

=Pre-requisites
https://github.com/teju85/dockerfiles#pre-reqs

=Running
```bash
git clone https://github.com/teju85/nccl-bench
git clone https://github.com/teju85/dockerfiles
cd dockerfiles/ubuntu1604
make nccl-bench
cd ../..
./dockerfiles/scripts/launch -runas user nccl-bench:latest /bin/bash
container$ cd /work/nccl-bench
container$ make
```

=TODO
* Support one-device-per-process benchmarking
