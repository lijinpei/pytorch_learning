add_executable(conv_2d conv_2d.cpp)
target_link_libraries(conv_2d "${TORCH_LIBRARIES}" GTest::gtest_main)

gtest_discover_tests(conv_2d)
