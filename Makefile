# Assumes that one is running on a DGX1-box!
ARCH    ?= 70
DEVICES ?= 2 4 8
SIZES   ?= 256 1024 4096 16384 65536 262144 1048576 4194304 16777216 67108864

default: build
	for nDevices in $(DEVICES); do \
	    for size in $(SIZES); do \
	        ./nccl -n $nDevices -s $size; \
	    done \
	done

build:
	nvcc -arch=$(ARCH) -o nccl main.cu

clean:
	rm -f nccl
