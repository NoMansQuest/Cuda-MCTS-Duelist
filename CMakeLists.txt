cmake_minimum_required(VERSION 3.15)

# 1. Name your project
project(CudaMctsDuelist LANGUAGES CXX)

# 2. Require C++20 for Coroutines support
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 3. Use FetchContent to download Asio automatically
include(FetchContent)

FetchContent_Declare(
  asio
  GIT_REPOSITORY https://github.com/chriskohlhoff/asio.git
  GIT_TAG        asio-1-30-2  # The stable version tag
)

# This makes the asio headers available and creates an 'asio' target
FetchContent_MakeAvailable(asio)

# 4. Define your executables (Server and Client)
add_executable(server server.cpp)
add_executable(client client.cpp)

# 5. Link Asio and set definitions
# We use INTERFACE because Asio is header-only
target_link_libraries(server PRIVATE asio)
target_compile_definitions(server PRIVATE ASIO_STANDALONE)

target_link_libraries(client PRIVATE asio)
target_compile_definitions(client PRIVATE ASIO_STANDALONE)

# 6. Windows-specific: Link required networking libraries (Winsock)
if(WIN32)
    target_link_libraries(server PRIVATE ws2_32 mswsock)
    target_link_libraries(client PRIVATE ws2_32 mswsock)
endif()