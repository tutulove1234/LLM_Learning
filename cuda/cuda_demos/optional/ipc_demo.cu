// Linux-only, same-host, two-process CUDA IPC example.
// fork() happens before either process initializes CUDA; never fork after CUDA setup.
#include "cuda_demo.hpp"

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <cstring>

__global__ void ipc_fill(int* values, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) values[i] = i * 3;
}

template <typename T>
void write_exact(int fd, const T& object) {
  const char* p = reinterpret_cast<const char*>(&object);
  std::size_t left = sizeof(T);
  while (left) {
    ssize_t n = write(fd, p, left);
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) throw std::runtime_error("pipe write failed");
    p += n;
    left -= static_cast<std::size_t>(n);
  }
}

template <typename T>
void read_exact(int fd, T& object) {
  char* p = reinterpret_cast<char*>(&object);
  std::size_t left = sizeof(T);
  while (left) {
    ssize_t n = read(fd, p, left);
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) throw std::runtime_error("pipe read failed");
    p += n;
    left -= static_cast<std::size_t>(n);
  }
}

struct IpcMessage {
  int device;
  cudaIpcMemHandle_t memory;
  cudaIpcEventHandle_t event;
};

int child_process(int read_fd) {
  IpcMessage message{};
  read_exact(read_fd, message);
  close(read_fd);
  CUDA_CHECK(cudaSetDevice(message.device));
  int* remote = nullptr;
  cudaEvent_t remote_event{};
  CUDA_CHECK(cudaIpcOpenMemHandle(reinterpret_cast<void**>(&remote), message.memory,
                                  cudaIpcMemLazyEnablePeerAccess));
  CUDA_CHECK(cudaIpcOpenEventHandle(&remote_event, message.event));
  CUDA_CHECK(cudaEventSynchronize(remote_event));
  std::array<int, 256> result{};
  CUDA_CHECK(cudaMemcpy(result.data(), remote, sizeof(result), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaEventDestroy(remote_event));
  CUDA_CHECK(cudaIpcCloseMemHandle(remote));
  if (result[17] != 51 || result[255] != 765)
    throw std::runtime_error("IPC data verification failed");
  return 0;
}

int parent_process(int write_fd, pid_t child) {
  int device = 0;
  CUDA_CHECK(cudaSetDevice(device));
  int* memory = nullptr;
  CUDA_CHECK(cudaMalloc(&memory, 256 * sizeof(int)));
  cudaEvent_t event{};
  CUDA_CHECK(cudaEventCreateWithFlags(
      &event, cudaEventInterprocess | cudaEventDisableTiming));
  IpcMessage message{};
  message.device = device;
  CUDA_CHECK(cudaIpcGetMemHandle(&message.memory, memory));
  CUDA_CHECK(cudaIpcGetEventHandle(&message.event, event));
  ipc_fill<<<1, 256>>>(memory, 256);
  CUDA_KERNEL_CHECK();
  CUDA_CHECK(cudaEventRecord(event));
  write_exact(write_fd, message);
  close(write_fd);
  int status = 0;
  if (waitpid(child, &status, 0) < 0) throw std::runtime_error("waitpid failed");
  CUDA_CHECK(cudaEventDestroy(event));
  CUDA_CHECK(cudaFree(memory));
  if (!WIFEXITED(status) || WEXITSTATUS(status) != 0)
    throw std::runtime_error("IPC child failed");
  return 0;
}

int main() {
  int fds[2];
  if (pipe(fds) != 0) {
    std::cerr << "pipe: " << std::strerror(errno) << '\n';
    return 1;
  }
  pid_t pid = fork();
  if (pid < 0) {
    std::cerr << "fork: " << std::strerror(errno) << '\n';
    return 1;
  }
  try {
    if (pid == 0) {
      close(fds[1]);
      return child_process(fds[0]);
    }
    close(fds[0]);
    int result = parent_process(fds[1], pid);
    std::cout << "PASS: cudaIpcMemHandle + cudaIpcEventHandle across two processes\n";
    return result;
  } catch (const std::exception& e) {
    std::cerr << (pid == 0 ? "child" : "parent") << ": " << e.what() << '\n';
    return 1;
  }
}

