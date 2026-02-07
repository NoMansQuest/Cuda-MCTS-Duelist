cmake_minimum_required(VERSION 3.18)
project(Cuda_MCTS_Duelist LANGUAGES CXX CUDA)

# Fetching Full Boost (we're interested in 'asio' for network communication purposes)
set(BOOST_ENABLE_CMAKE ON CACHE BOOL "" FORCE)
set(BOOST_INCLUDE_LIBRARIES asio system filesystem program_options) # Only build/include what you need

include(FetchContent)
FetchContent_Declare(
  Boost
  GIT_REPOSITORY https://github.com/boostorg/boost.git
  GIT_PROGRESS TRUE
  GIT_TAG boost-1.84.0
)

FetchContent_MakeAvailable(Boost)

# Five Generations of NVIDIA Support (known as Fat Binary)
# Targets Architecture:
# Maxwell (5.2), Pascal (6.1), Turing (7.5), Ampere (8.6), Ada (8.9), Blackwell (10.0) and PTX (100-virtual, for future generations)
set(CMAKE_CUDA_ARCHITECTURES 52 61 75 86 89 100 100-virtual)

# Project Files
set(SOURCES
src/main.cpp
src/kernel.cu
)

add_executable(Cuda_MCTS_Duelist ${SOURCES})

# Include boost folders
target_include_directories(Cuda_MCTS_Duelist PRIVATE ${Boost_INCLUDE_DIRS})

# Compiler Settings
set_target_properties(Cuda_MCTS_Duelist PROPERTIES 
    CXX_STANDARD 17
    CUDA_STANDARD 17
    CUDA_SEPARABLE_COMPILATION ON
    CMAKE_CUDA_STANDARD_REQUIRED ON
)

# Definitions
target_compile_definitions(Cuda_MCTS_Duelist PRIVATE USE_CUDA)

# Linking Boost and Threads
find_package(Threads REQUIRED)

target_link_libraries(Cuda_MCTS_Duelist PRIVATE
    Boost::asio       # This provides the headers AND necessary definitions
    Boost::system     # Asio often requires Boost.System    
    Boost::filesystem
    Boost::program_options
    Threads::Threads
)

# Optimization (Optional but recommended for MCTS, with O3 offering the highest performance)
if(MSVC)
    target_compile_options(Cuda_MCTS_Duelist PRIVATE
        # 1. Flags for .cpp files (main.cpp)
        $<$<COMPILE_LANGUAGE:CXX>:/W4 /EHsc>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Debug>>:/W4 /Zi /Ob0 /RTC1>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:/O2>

        # 2. Flags for .cu files (kernel.cu)
        # Note: These are wrapped in -Xcompiler so NVCC passes them to MSVC correctly (MSVC could be, as always, be problematic)
        # Note: '--expt-relaxed-constexpr' is used to allow constexpr calls in CUDA
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -Xcompiler=/W4,-EHsc>
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-Xcompiler=/W4,-Zi,-Ob0,-RTC1>
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-Xcompiler=/O2>
    )
else()
    # Fallback for Linux/GCC
    # Note: '--expt-relaxed-constexpr' is used to allow constexpr calls in CUDA
    target_compile_options(Cuda_MCTS_Duelist PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -Xcompiler=-Wall,-Wextra>
    )
endif()