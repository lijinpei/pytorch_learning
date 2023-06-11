cmake_minimum_required(VERSION 3.26)
project(pytorch_learning)

include(FetchContent)
FetchContent_Declare(
	googletest
	URL https://codeload.github.com/google/googletest/zip/refs/tags/v1.13.0
	)
FetchContent_MakeAvailable(googletest)

enable_testing()


set(CMAKE_CXX_STANDARD 17)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

include(GoogleTest)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(hello_world)
add_subdirectory(conv_2d)
