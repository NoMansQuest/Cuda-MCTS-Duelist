# CUDA MCTS Duelist – Connect 4 AI on GPU

**GPU-accelerated Monte Carlo Tree Search (MCTS) player for Connect 4**, implemented in CUDA C++.  
Two AI instances (or human + AI) can duel each other via TCP/IP socket communication — perfect for distributed or remote matches.

## Features

- Pure CUDA kernels for massively parallel MCTS simulations (100k–1M+ rollouts/sec on modern GPUs)
- CPU reference MCTS implementation (for validation, comparison & debugging)
- TCP/IP networking layer used to communicate between the two players (one server, one client)
- Efficient GPU-friendly board representation and move generation
- CMake-based cross-platform build (Linux/Windows + NVIDIA GPU)
- MIT licensed - free to use, modify, learn from

## Why Connect 4 + GPU MCTS?

Connect 4 has a much larger state space and deeper trees than Tic-Tac-Toe, GPU parallelism shines here.  
This project demonstrates:
- High-performance CUDA kernel design
- MCTS algorithm on GPU (tree traversal, expansion, simulation, back-propagation)
- Inter-process / networked AI competition setup (BOOST::asio)

## Project Structure

```
.
├── CMakeLists.txt
├── README.md
├── LICENSE
├── .gitignore
├── src/
│   ├── main.cpp            # Entry point, argument parsing, mode selection
│   ├── chat_session.cpp    # Networking layer.
│   ├── client_mode.cpp     # Client-mode code
│   ├── server_mode.cpp     # Server-mode code
│   ├── comm_result.cpp     # Communication result enum
│   ├── game_engine.cpp     # Game engine, playing the game and invoking CUDA procedures
│   ├── game_helpers.cpp    # Shared code, helper
│   ├── kernel.cu           # CUDA kernel and infrastructure code
│   └── session_message.h   # Object exchanged between server and client (serializable)
└── (optional) tests/       # To be added later

```

## Prerequisites

- CMake ≥ 3.18
- CUDA Toolkit 11.8 or 12.x
- C++17 compatible compiler (g++/MSVC)
- NVIDIA GPU with compute capability ≥ 6.1 (Pascal+) recommended

## Build

Once the repository is cloned to a local folder, the following commands could be used to configure and build the solution (the following example is for 'Debug' build):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CUDA_FLAGS="--allow-unsupported-compiler"
cmake --build build --config Debug --parallel
```
This would build the project under 'build' folder, where you could then locate the executable.


## Usage Example

Once the solution is compiled, you are ready to run the application. Assuming your terminal is at the root folder of the project, the following command will launch the application as server:

```bash
.\build\Debug\Cuda_MCTS_Duelist.exe --server --port 60000
```

Once the server is running, the client could then be launched. This client and server will play with each other in turn until either one of the parties win or a tie is reached. To launch the client:

```bash
.\build\Debug\Cuda_MCTS_Duelist.exe --client --ip 127.0.0.1 --port 60000
```

To access the help menu, run the application with the **--help** argument.

Once launched, the game will be played in turn and an output similar to the following would be produced:

```
PS C:\Users\nasse\Source\Repos\Cuda-MCTS-Duelist> .\build\Debug\Cuda_MCTS_Duelist.exe --client --ip 127.0.0.1 --port 60000
CUDA MCTS Duelist starting...
Starting as client, connecting to 127.0.0.1:60000...
Connected to 127.0.0.1:60000
Opponent played row 5, column 6
 Played row 5, column 5
Opponent played row 4, column 6
 Played row 4, column 5
Opponent played row 3, column 6
 Played row 2, column 6
Opponent played row 5, column 1
 Played row 3, column 5
Opponent played row 2, column 5
 Played row 5, column 3
Opponent played row 4, column 1
 Played row 5, column 2
Opponent played row 5, column 4
 Played row 4, column 4 - We have won!
Final state of the game-board:

      1 2 3 4 5 6 7
     ——————————————
  1 | - - - - - - -
  2 | - - - - - - -
  3 | - - - - - X O
  4 | - - - - - O X
  5 | - X - - O O X
  6 | - X O O X O X

```
As illustrated above, player 'O' (the client) has won given the four diagonal discs placed from Row-3:Column-7 to Row-6:Column-4.

## License

MIT License: see LICENSE
Made with :heart: and a lot of CUDA kernels.

![C++](https://img.shields.io/badge/language-C++-blue)
![CUDA](https://img.shields.io/badge/CUDA-12.x-green)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

