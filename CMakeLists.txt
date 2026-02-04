cmake_minimum_required(VERSION 3.18)
project(Cuda_MCTS_Duelist LANGUAGES CXX CUDA)

# Fetching Full Boost (we're interested in 'asio' for now)
include(FetchContent)
FetchContent_Declare(
  boost
  URL https://github.com/boostorg/boost/releases/download/boost-1.84.0/boost-1.84.0.tar.gz
)

FetchContent_MakeAvailable(boost)

# Five Generations of NVIDIA Support (known as Fat Binary)
# Targets Architecture:
# Maxwell (5.2), Pascal (6.1), Turing (7.5), Ampere (8.6), Ada (8.9), Blackwell (10.0) and PTX (100-virtual, for future generations)
set(CMAKE_CUDA_ARCHITECTURES 52 61 75 86 89 100 100-virtual)

# 3. Project Files
set(SOURCES
    src/main.cpp
    src/kernel.cu
)

add_executable(Cuda_MCTS_Duelist ${SOURCES})

# 4. Compiler Settings
set_target_properties(Cuda_MCTS_Duelist PROPERTIES 
    CXX_STANDARD 17
    CUDA_STANDARD 17
    CUDA_SEPARABLE_COMPILATION ON
)

# Definitions
target_compile_definitions(Cuda_MCTS_Duelist PRIVATE USE_CUDA)

# Linking Boost and Threads
find_package(Threads REQUIRED)
target_link_libraries(Cuda_MCTS_Duelist PRIVATE boost Threads::Threads)

# Optimization (Optional but recommended for MCTS, with O3 offering the highest performance)
target_compile_options(Cuda_MCTS_Duelist PRIVATE 
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)