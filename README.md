# CUDA MCTS Duelist – Connect 4 AI on GPU

**GPU-accelerated Monte Carlo Tree Search (MCTS) player for Connect 4**, implemented in CUDA C++.  
Two AI instances (or human + AI) can duel each other via TCP/IP socket communication — perfect for distributed or remote matches.

<p align="center">
  <img src="https://via.placeholder.com/800x400?text=Connect+4+GPU+MCTS+duel" alt="Connect 4 GPU MCTS visualization" width="800"/>
  <!-- Replace with real screenshot / gif of gameplay or stats later -->
</p>

## Features

- Pure CUDA kernels for massively parallel MCTS simulations (100k–1M+ rollouts/sec on modern GPUs)
- CPU reference MCTS implementation (for validation, comparison & debugging)
- TCP/IP networking layer — run AI vs AI, human vs AI, or AI vs AI across machines
- Efficient GPU-friendly board representation and move generation
- CMake-based cross-platform build (Linux/Windows + NVIDIA GPU)
- MIT licensed — free to use, modify, learn from

## Why Connect 4 + GPU MCTS?

Connect 4 has a much larger state space and deeper trees than Tic-Tac-Toe → GPU parallelism shines here.  
This project demonstrates:
- High-performance CUDA kernel design
- MCTS algorithm on GPU (tree traversal, expansion, simulation, backpropagation)
- Inter-process / networked AI competition setup

## Project Structure
.
├── CMakeLists.txt
├── README.md
├── LICENSE
├── .gitignore
├── src/
│   ├── main.cpp            # Entry point, argument parsing, mode selection
│   ├── chat_session.cpp    # CPU MCTS reference
│   ├── client_mode.cpp     # Core CUDA MCTS kernels + host code
│   ├── server_mode.cpp     # Board logic, move generation, win-check on GPU
│   ├── comm_result.cpp     # TCP server/client implementation
│   ├── game_engine.cpp     # Shared structs, enums (Player, GameState, Move…)
│   ├── game_helpers.cpp    # Shared structs, enums (Player, GameState, Move…)
│   ├── server_mode.cpp     # Shared structs, enums (Player, GameState, Move…)
│   ├── kernel.cu           # Shared structs, enums (Player, GameState, Move…)
│   └── session_message.h   # RNG, reduction helpers, etc.
└── (optional) tests/       # Add later

## Prerequisites

- CMake ≥ 3.18
- CUDA Toolkit 11.8 or 12.x
- C++17 compatible compiler (g++/MSVC)
- NVIDIA GPU with compute capability ≥ 6.1 (Pascal+) recommended

## Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release --parallel
```

On Windows with Visual Studio, open the generated solution and build Release|x64.

## Usage Example

```bash
# AI vs AI duel on GPU – most impressive mode
./cuda_mcts_duelist --mode gpu-ai-vs-ai --simulations 200000 --host localhost --port 5555

# Start AI server (GPU thinker)
./cuda_mcts_duelist --mode gpu-server --port 5555 --simulations 100000

# Human client connects and plays against it
./cuda_mcts_duelist --mode human-client --host 127.0.0.1 --port 5555

# CPU-only baseline duel (for comparison)
./cuda_mcts_duelist --mode cpu-ai-vs-ai --simulations 50000
```

Run with **--help** for full options.

## License

MIT License: see LICENSE
Made with :heart: and a lot of CUDA kernels.

![C++](https://img.shields.io/badge/language-C++-blue)
![CUDA](https://img.shields.io/badge/CUDA-12.x-green)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

