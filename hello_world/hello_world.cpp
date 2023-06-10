#include <iostream>
#include <utility>

#include "gtest/gtest.h"
#include "torch/torch.h"

struct Shape2D {
  ssize_t x;
  ssize_t y;
};

class TensorAddTest : public testing::TestWithParam<Shape2D> {};

torch::Tensor tensor_add(torch::Tensor op1, torch::Tensor op2) {
  assert(op1.sizes() == op2.sizes());
  auto result = torch::empty_like(op1);
  auto op1_acc = op1.accessor<float, 2>();
  auto op2_acc = op2.accessor<float, 2>();
  auto result_acc = result.accessor<float, 2>();
  auto size_x = op1.size(0);
  auto size_y = op1.size(1);
  for (std::size_t x = 0; x < size_x; ++x) {
    for (std::size_t y = 0; y < size_y; ++y) {
      result_acc[x][y] = op1_acc[x][y] + op2_acc[x][y];
    }
  }
  return result;
}

TEST_P(TensorAddTest, Case1) {
  auto params = GetParam();
  auto size_x = params.x;
  auto size_y = params.y;
  auto tensor1 = torch::randn({size_x, size_y});
  auto tensor2 = torch::randn({size_x, size_y});
  auto tensor3 = tensor1 + tensor2;
  auto result = tensor_add(tensor1, tensor2);
  bool match = tensor3.equal(result);
  EXPECT_TRUE(match);
}

INSTANTIATE_TEST_SUITE_P(SomeShapes, TensorAddTest,
                         testing::Values(Shape2D{128, 128}, Shape2D{512, 512},
                                         Shape2D{128, 512}));
