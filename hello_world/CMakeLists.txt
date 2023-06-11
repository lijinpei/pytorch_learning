add_executable(hello_world hello_world.cpp)
target_link_libraries(hello_world "${TORCH_LIBRARIES}" GTest::gtest_main)

gtest_discover_tests(hello_world)
