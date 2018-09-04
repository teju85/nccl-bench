=Introduction
This is a modified version from the 'nccl examples' page at here:
https://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/index.html#examples.
This runs across various buffer sizes and notes down the transfer times.
Should be useful if you are trying to benchmark your system for multi-gpu
communications, especially using AllReduce primitives given by nccl library.

=Pre-requisites
https://github.com/teju85/dockerfiles#pre-reqs

=Setting up container
This should be a one-time thing, in a given machine.
```bash
git clone https://github.com/teju85/dockerfiles
cd dockerfiles/ubuntu1604
make nccl-bench
cd ../..
```

=Running the bench
```bash
git clone https://github.com/teju85/nccl-bench
./dockerfiles/scripts/launch -runas user nccl-bench:latest /bin/bash
inside-container$ cd /work/nccl-bench
inside-container$ make
```

=TODO
* Support one-device-per-process benchmarking
