// modified from the sample code at:
//  https://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/index.html#examples
#include <stdio.h>
#include <nccl.h>
#include <string>
#include <stdexcept>
#include <string.h>
#include <stdlib.h>

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

#define NCCL_CHECK(cmd)                                   \
    do {                                                  \
        ncclResult_t status = cmd;                        \
        ASSERT(status == ncclSuccess,                     \
               "FAIL: nccl-call='%s'. Reason:%s\n",       \
               #call, ncclGetErrorString(r));             \
    } while(0)

bool isPo2(int in) {
    return (in > 1) && (in & in-1);
}

void printHelp() {
    printf("USAGE:\n");
    printf(" ./nccl [-h] [-n <nDevices>] [-s <buffSize>]\n");
}

int main(int argc, char** argv) {
    int nDevices = 2;
    int size = 32*1024*1024;
    for(int i=1;i<argc;++i) {
        if(!strcmp("-h", argv[i])) {
            printHelp();
            return 0;
        } else if(!strcmp("-n", argv[i])) {
            ASSERT(i < argc, "'-n' requires an argument!");
            ++i;
            nDevices = atoi(argv[i]);
            ASSERT(isPo2(nDevices), "Num devices must be PO2 and more than 1!");
        } else if(!strcmp("-s", argv[i])) {
            ASSERT(i < argc, "'-s' requires an argument!");
            ++i;
            size = atoi(argv[i]);
        } else {
            ASSERT(false, "Incorrect argument '%s'!", argv[i]);
        }
    }
    ncclComm_t* comms = new ncclComm_t[nDevices];
    int* devs = new int[nDevices];
    for(int i=0;i<nDevices;++i) {
        devs[i] = i;
    }
    cudaStream_t* streams = new cudaStream_t[nDevices];
    int** sendbuff = new int*[nDevices];
    int** recvbuff = new int*[nDevices];
    cudaEvent_t* events = new cudaEvent_t[2*nDevices];
    for(int i=0;i<nDevices;++i) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaMalloc(sendbuff+i, size*sizeof(int)));
        CUDA_CHECK(cudaMalloc(recvbuff + i, size*sizeof(int)));
        CUDA_CHECK(cudaMemset(sendbuff[i], 1, size*sizeof(int)));
        CUDA_CHECK(cudaMemset(recvbuff[i], 0, size*sizeof(int)));
        CUDA_CHECK(cudaStreamCreate(streams+i));
        CUDA_CHECK(cudaEventCreate(events+2*i));
        CUDA_CHECK(cudaEventCreate(events+2*i+1));
    }
    NCCL_CHECK(ncclCommInitAll(comms, nDevices, devs));
    for(int i=0;i<nDevices;++i) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaEventRecord(events[2*i], streams[i]));
    }
    NCCL_CHECK(ncclGroupStart());
    for(int i=0;i<nDevices;++i) {
        NCCL_CHECK(ncclAllReduce((const void*)sendbuff[i], (void*)recvbuff[i],
                                 size, ncclInt, ncclSum, comms[i], streams[i]));
    }
    NCCL_CHECK(ncclGroupEnd());
    for(int i=0;i<nDevices;++i) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaEventRecord(events[2*i+1], streams[i]));
    }
    for(int i=0;i<nDevices;++i) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaEventSynchronize(events[2*i+1]));
        CUDA_CHECK(cudaStreamSynchronize(streams[i]));
    }
    for(int i=0;i<nDevices;++i) {
        CUDA_CHECK(cudaSetDevice(i));
        float et;
        CUDA_CHECK(cudaEventElapsedTime(&et, events[2*i], events[2*i+1]));
        printf("Device=%d,nDevices=%d,size=%d,time=%fms\n", i, nDevices, size, et);
    }
    for(int i=0;i<nDevices;++i) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaFree(recvbuff[i]));
        CUDA_CHECK(cudaFree(sendbuff[i]));
        CUDA_CHECK(cudaEventDestroy(events[2*i]));
        CUDA_CHECK(cudaEventDestroy(events[2*i+1]));
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }
    for(int i=0;i<nDevices;++i) {
        NCCL_CHECK(ncclCommDestroy(comms[i]));
    }
    delete [] events;
    delete [] recvbuff;
    delete [] sendbuff;
    delete [] streams;
    delete [] devs;
    delete [] comms;
    printf("Success\n");
    return 0;
}
