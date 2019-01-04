# Introduction
This is a modified version from the 'nccl examples' page at here:
https://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/index.html#examples.
This runs across various buffer sizes and notes down the transfer times.
Should be useful if you are trying to benchmark your system for multi-gpu
communications, especially using AllReduce primitives given by nccl library.

# Pre-requisites
https://github.com/teju85/dockerfiles#pre-reqs

# Setting up benchmark
This should be a one-time thing, in a given machine.
```bash
git clone https://github.com/teju85/dockerfiles
cd dockerfiles/ubuntu1604
make nccl-bench
cd ../..
```
This container should have the git repo also built-inside it!

# Running the bench
```bash
./dockerfiles/scripts/launch -runas user nccl:bench /bin/bash
inside-container$ /opt/nccl-bench/nccl -h
```
