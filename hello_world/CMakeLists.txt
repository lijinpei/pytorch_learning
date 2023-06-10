cmake_minimum_required(VERSION 3.26)
project(hello_world)

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

add_executable(hello_world hello_world.cpp)
target_link_libraries(hello_world "${TORCH_LIBRARIES}" GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(hello_world)
