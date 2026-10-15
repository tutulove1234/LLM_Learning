// NOT BUILT BY DEFAULT. Cross-machine communication is a library concern, not a
// CUDA Runtime API feature. Compile with an MPI implementation and NCCL.
#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>

#include <iostream>
#include <stdexcept>

#define MPI_OK(x) do { if ((x) != MPI_SUCCESS) throw std::runtime_error("MPI failure"); } while (0)
#define NCCL_OK(x) do { if ((x) != ncclSuccess) throw std::runtime_error(ncclGetErrorString(x)); } while (0)
#define CUDA_OK(x) do { cudaError_t e = (x); if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); } while (0)

__global__ void set_rank(float* p, int rank) { p[threadIdx.x] = float(rank + 1); }

int main(int argc, char** argv) {
  MPI_OK(MPI_Init(&argc, &argv));
  int rank = 0, size = 0, local_rank = 0;
  MPI_OK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  MPI_OK(MPI_Comm_size(MPI_COMM_WORLD, &size));
  // In production derive local_rank from MPI shared-memory communicators or launcher env.
  MPI_Comm local_comm{};
  MPI_OK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                             MPI_INFO_NULL, &local_comm));
  MPI_OK(MPI_Comm_rank(local_comm, &local_rank));
  CUDA_OK(cudaSetDevice(local_rank));

  ncclUniqueId id{};
  if (rank == 0) NCCL_OK(ncclGetUniqueId(&id));
  MPI_OK(MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));
  ncclComm_t comm{};
  NCCL_OK(ncclCommInitRank(&comm, size, id, rank));
  cudaStream_t stream{};
  CUDA_OK(cudaStreamCreate(&stream));
  float* value = nullptr;
  CUDA_OK(cudaMalloc(&value, sizeof(float)));
  set_rank<<<1, 1, 0, stream>>>(value, rank);
  // Every rank receives the sum. NCCL chooses NVLink/PCIe/network transports.
  NCCL_OK(ncclAllReduce(value, value, 1, ncclFloat, ncclSum, comm, stream));
  float host = 0;
  CUDA_OK(cudaMemcpyAsync(&host, value, sizeof(host), cudaMemcpyDeviceToHost, stream));
  CUDA_OK(cudaStreamSynchronize(stream));
  std::cout << "rank " << rank << " all-reduce=" << host << '\n';
  CUDA_OK(cudaFree(value));
  CUDA_OK(cudaStreamDestroy(stream));
  NCCL_OK(ncclCommDestroy(comm));
  MPI_OK(MPI_Comm_free(&local_comm));
  MPI_OK(MPI_Finalize());
}
