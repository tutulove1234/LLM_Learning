cmake_minimum_required(VERSION 3.24)
project(cuda_runtime_api_tour LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# A widely supported default. Override for your GPU, e.g. -DCMAKE_CUDA_ARCHITECTURES=native
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

find_package(CUDAToolkit 12.0 REQUIRED)

add_executable(cuda_api_tour
  src/main.cu
  src/device_info.cu
  src/memory.cu
  src/execution.cu
  src/graphs.cu
  src/mempool.cu
  src/multigpu.cu)
target_include_directories(cuda_api_tour PRIVATE include)
target_link_libraries(cuda_api_tour PRIVATE CUDA::cudart)
set_target_properties(cuda_api_tour PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(cuda_api_tour PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;-lineinfo>)

if(UNIX AND NOT APPLE)
  add_executable(cuda_ipc_demo optional/ipc_demo.cu)
  target_include_directories(cuda_ipc_demo PRIVATE include)
  target_link_libraries(cuda_ipc_demo PRIVATE CUDA::cudart)
  set_target_properties(cuda_ipc_demo PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

option(BUILD_EXTERNAL_INTEROP "Compile the Linux external-memory/semaphore skeleton" OFF)
if(BUILD_EXTERNAL_INTEROP AND UNIX AND NOT APPLE)
  add_library(cuda_external_interop OBJECT optional/external_interop_skeleton.cu)
  target_include_directories(cuda_external_interop PRIVATE include)
endif()

option(BUILD_NCCL_MPI "Build the cross-host NCCL + MPI example" OFF)
if(BUILD_NCCL_MPI)
  find_package(MPI REQUIRED COMPONENTS CXX)
  find_library(NCCL_LIBRARY nccl REQUIRED)
  find_path(NCCL_INCLUDE_DIR nccl.h REQUIRED)
  add_executable(cuda_nccl_mpi optional/nccl_mpi_cross_host.cu)
  target_include_directories(cuda_nccl_mpi PRIVATE ${NCCL_INCLUDE_DIR})
  target_link_libraries(cuda_nccl_mpi PRIVATE CUDA::cudart MPI::MPI_CXX ${NCCL_LIBRARY})
endif()

include(CTest)
if(BUILD_TESTING)
  add_test(NAME cuda_api_tour_help COMMAND cuda_api_tour --help)
  add_test(NAME cuda_api_tour_all COMMAND cuda_api_tour all)
  set_tests_properties(cuda_api_tour_all PROPERTIES SKIP_RETURN_CODE 77)
endif()
