# Assumes that one is running on a DGX1-box!
ARCH    ?= 70

EXE     := nccl
GENCODE := $(foreach a,$(ARCH),-gencode arch=compute_$(a),code=sm_$(a) -arch=sm_$(a))
LIBS    := -lnccl
FLAGS   := $(GENCODE) -std=c++11 -ccbin mpiCC

default:
	@echo "make what? Available targets are:"
	@echo "  . build  - builds the executable"
	@echo "  . clean  - cleans the built files"
	@echo "Variables that customize build behavior are:"
	@echo "  . ARCH   - space-separated list of gpu architectures to"
	@echo "             compile for [$(ARCH)]"

build:
	nvcc $(FLAGS) -o $(EXE) main.cu $(LIBS)

clean:
	rm -f nccl
