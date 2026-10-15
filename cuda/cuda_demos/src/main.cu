#include "cuda_demo.hpp"

#include <algorithm>
#include <iomanip>

int main(int argc, char** argv) {
  const std::vector<Demo> demos = {
      {"info", "device/version/error/limit/cache queries", print_device_info},
      {"memory", "linear, pinned, registered, managed, pitched, array, texture/surface", demo_memory},
      {"execution", "streams, events, callbacks, occupancy, launch APIs", demo_execution_and_streams},
      {"graphs", "explicit graph construction, update and stream capture", demo_graphs},
      {"mempool", "stream-ordered allocation and explicit memory pools", demo_mempool},
      {"multigpu", "peer-access discovery and peer copies (skips with <2 GPUs)", demo_multigpu},
  };

  auto usage = [&] {
    std::cout << "CUDA Runtime API tour\n\nUsage: cuda_api_tour <command>\n\nCommands:\n";
    for (const auto& demo : demos)
      std::cout << "  " << std::left << std::setw(12) << demo.name << demo.description << '\n';
    std::cout << "  all         run every applicable demo\n"
                 "  --help      show this help (works without a GPU)\n";
  };

  if (argc != 2 || std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h") {
    usage();
    return argc == 2 ? 0 : 2;
  }

  const std::string command = argv[1];
  try {
    if (!has_cuda_device()) {
      std::cerr << "SKIP: no CUDA-capable device/driver is available. --help remains usable.\n";
      return 77;
    }
    bool found = false;
    for (const auto& demo : demos) {
      if (command == demo.name || command == "all") {
        found = true;
        std::cout << "\n== " << demo.name << ": " << demo.description << " ==\n";
        demo.run();
      }
    }
    if (!found) {
      usage();
      return 2;
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "\nPASS\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << '\n';
    return 1;
  }
}

