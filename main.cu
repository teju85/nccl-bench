// modified from the sample code at:
//  https://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/index.html#examples
#include <stdio.h>
#include <nccl.h>
#include <string>
#include <stdexcept>
#include <string.h>
#include <stdlib.h>
#include <mpi.h>

#define THROW(fmt, ...)                                         \
    do {                                                        \
        std::string msg;                                        \
        char errMsg[2048];                                      \
        sprintf(errMsg, "Exception occured! file=%s line=%d: ", \
                __FILE__, __LINE__);                            \
        msg += errMsg;                                          \
        sprintf(errMsg, fmt, ##__VA_ARGS__);                    \
        msg += errMsg;                                          \
        throw std::runtime_error(msg);                          \
    } while(0)

#define ASSERT(check, fmt, ...)                  \
    do {                                         \
        if(!(check))  THROW(fmt, ##__VA_ARGS__); \
    } while(0)

#define CUDA_CHECK(call)                                \
    do {                                                \
        cudaError_t status = call;                      \
        ASSERT(status == cudaSuccess,                   \
               "FAIL: call='%s'. Reason:%s\n",          \
               #call, cudaGetErrorString(status));      \
    } while(0)

#define NCCL_CHECK(call)                                   \
    do {                                                   \
        ncclResult_t status = call;                        \
        ASSERT(status == ncclSuccess,                      \
               "FAIL: nccl-call='%s'. Reason:%s\n",        \
               #call, ncclGetErrorString(status));         \
    } while(0)

#define MPI_CHECK(call)                                    \
  do {                                                     \
    auto status = call;                                    \
    ASSERT(status == 0, "FAIL: mpi-call='%s'!", #call);    \
  } while (0)

void printHelp() {
    printf("USAGE:\n"
           "./nccl -h \n"
           "  -h             Print this help message and exit.\n\n"
           "mpirun -np <nRanks> ./nccl [-op <op>] [-ne <nElems>]\n"
           "  <nRanks>       Number of ranks to run nccl ops on. It's assumed\n"
           "                 that one is launching one-rank-per-gpu!\n"
           "  -ne <nElems>   Number of elements to be transferred.\n"
           "  -op <op>       Which nccl op to run. Available options are:\n"
           "                   . allReduce\n"
           "                   . allGather\n"
           "                   . broadcast\n"
           "                   . reduce\n"
           "                   . reduceScatter\n"
           "                 Default is allReduce.\n");
}

int ceildiv(int a, int b) {
    return (a + b - 1) / b;
}

int getMyRank() {
    int myRank;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myRank));
    return myRank;
}

int getTotalRanks() {
    int nRanks;
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &nRanks));
    return nRanks;
}

bool amIroot(int rootRank = 0) {
    auto myRank = getMyRank();
    return myRank == rootRank;
}

void setMyGpu() {
    int nDevices;
    CUDA_CHECK(cudaGetDeviceCount(&nDevices));
    int id = getMyRank() % nDevices;
    CUDA_CHECK(cudaSetDevice(id));
}

ncclUniqueId getNcclId() {
    ncclUniqueId id;
    if(amIroot()) {
        ncclGetUniqueId(&id);
    }
    MPI_CHECK(MPI_Bcast((void*)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));
    return id;
}

__global__ void fillKernel(float* buff, int len, float val) {
    const int stride = gridDim.x * blockDim.x;
    const int tid = threadIdx.x + (blockIdx.x * blockDim.x);
    for(int idx=tid;idx<len;idx+=stride) {
        buff[idx] = val;
    }
}

void initializeBuffer(float* buff, int len, float val, cudaStream_t s) {
    const int tpb = 256;
    int nblks = ceildiv(len, 256 * 8);
    fillKernel<<<nblks, tpb, 0, s>>>(buff, len, val);
    CUDA_CHECK(cudaPeekAtLastError());
}

int main(int argc, char** argv) {
    // arg parse and initialize
    MPI_CHECK(MPI_Init(&argc, &argv));
    auto nRanks = getTotalRanks();
    int nElems = 32*1024*1024;
    std::string op("allReduce");
    for(int i=1;i<argc;++i) {
        if(!strcmp("-h", argv[i])) {
            printHelp();
            return 0;
        } else if(!strcmp("-ne", argv[i])) {
            ASSERT(i < argc, "'-ne' requires an argument!");
            ++i;
            nElems = atoi(argv[i]);
        } else if(!strcmp("-op", argv[i])) {
            ASSERT(i < argc, "'-op' requires an argument!");
            ++i;
            op = argv[i];
        } else {
            ASSERT(false, "Incorrect argument '%s'!", argv[i]);
        }
    }
    if(op == "allGather" || op == "reduceScatter")
        ASSERT(nElems % nRanks == 0,
               "For op=allGather|reduceScatter, <nElems> must be divisible by"
               " <nRanks>!");

    // setup
    setMyGpu();
    auto nid = getNcclId();
    ncclComm_t comm;
    NCCL_CHECK(ncclCommInitRank(&comm, nRanks, nid, getMyRank()));
    float *inbuff = nullptr, *outbuff = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&inbuff, sizeof(float)*nElems));
    CUDA_CHECK(cudaMalloc((void**)&outbuff, sizeof(float)*nElems));
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    initializeBuffer(inbuff, nElems, (float)getMyRank(), stream);

    // benchmark
    CUDA_CHECK(cudaEventRecord(start, stream));
    if(op == "allReduce") {
        NCCL_CHECK(ncclAllReduce((const void*)inbuff, (void*)outbuff, nElems,
                                 ncclFloat, ncclSum, comm, stream));
    } else if(op == "allGather") {
        NCCL_CHECK(ncclAllGather((const void*)inbuff, (void*)outbuff,
                                 nElems/nRanks, ncclFloat, comm, stream));
    } else if(op == "broadcast") {
        NCCL_CHECK(ncclBroadcast((const void*)inbuff, (void*)outbuff, nElems,
                                 ncclFloat, 0, comm, stream));
    } else if(op == "reduce") {
        NCCL_CHECK(ncclReduce((const void*)inbuff, (void*)outbuff, nElems,
                              ncclFloat, ncclSum, 0, comm, stream));
    } else if(op == "reduceScatter") {
        NCCL_CHECK(ncclReduceScatter((const void*)inbuff, (void*)outbuff,
                                     nElems/nRanks, ncclFloat, ncclSum, comm,
                                     stream));
    } else {
        ASSERT(false, "Invalid op=%s!", op.c_str());
    }
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    float et;
    CUDA_CHECK(cudaEventElapsedTime(&et, start, stop));
    printf("Result: rank=%d time-ms=%f\n", getMyRank(), et);
    fflush(stdout);

    // cleanup
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(outbuff));
    CUDA_CHECK(cudaFree(inbuff));
    NCCL_CHECK(ncclCommDestroy(comm));
    MPI_CHECK(MPI_Finalize());
    return 0;
}
