#include <limits>

#include "c10/util/variant.h"
#include "gtest/gtest.h"
#include "torch/expanding_array.h"
#include "torch/nn/functional/conv.h"
#include "torch/torch.h"

namespace nn_func = torch::nn::functional;

struct PaddingModeVisitor {
  long index;
  long filter_size;
  long operator()(const torch::ExpandingArray<2>& pm) { return (*pm)[index]; }
  long operator()(const torch::enumtype::kValid&) { return 0; }
  long operator()(const torch::enumtype::kSame&) { return filter_size - 1; }
};

torch::Tensor conv_2d(torch::Tensor activation, torch::Tensor weight,
                      nn_func::Conv2dFuncOptions options) {
  auto weight_sizes = weight.sizes();
  assert(weight_sizes.size() == 4);
  auto activation_sizes = activation.sizes();
  assert(activation_sizes.size() == 4);
  auto N = activation_sizes[0];
  auto Cin = activation_sizes[1];
  auto AH = activation_sizes[2];
  auto AW = activation_sizes[3];
  auto Cout = weight_sizes[0];
  assert(Cin == weight_sizes[1]);
  auto FH = weight_sizes[2];
  auto FW = weight_sizes[3];
  auto stride = options.stride();
  auto padding = options.padding();
  auto dilation = options.dilation();
  auto group = options.groups();
  assert(std::all_of(stride->begin(), stride->end(),
                     [](int64_t v) { return v == 1; }));
  assert(std::all_of(dilation->begin(), dilation->end(),
                     [](int64_t v) { return v == 1; }));
  assert(group == 1);
  auto h_padding_size = c10::visit(PaddingModeVisitor{0, FH}, padding);
  auto w_padding_size = c10::visit(PaddingModeVisitor{1, FW}, padding);
  auto OH = AH - FH + 1 + h_padding_size;
  auto OW = AW - FW + 1 + w_padding_size;
  auto pad_h = (h_padding_size + 1) / 2;
  auto pad_w = (w_padding_size + 1) / 2;
  auto output = torch::empty({N, Cout, OH, OW},
                             torch::TensorOptions().dtype(torch::kInt));
  auto act_acc = activation.accessor<int, 4>();
  auto weight_acc = weight.accessor<int, 4>();
  auto output_acc = output.accessor<int, 4>();
  auto get_actvation = [&](int n, int ci, int h, int w) -> int {
    if (h >= 0 && h < AH && w >= 0 && w < AW) {
      return act_acc[n][ci][h][w];
    }
    return 0;
  };
  for (int n = 0; n < N; ++n) {
    for (int co = 0; co < Cout; ++co) {
      for (int oh = 0; oh < OH; ++oh) {
        for (int ow = 0; ow < OW; ++ow) {
          int sum = 0;
          for (int ci = 0; ci < Cin; ++ci) {
            for (int fh = 0; fh < FH; ++fh) {
              for (int fw = 0; fw < FW; ++fw) {
                sum += weight_acc[co][ci][fh][fw] *
                       get_actvation(n, ci, oh + fh - pad_h, ow + fw - pad_w);
              }
            }
          }
          output_acc[n][co][oh][ow] = sum;
        }
      }
    }
  }
  return output;
}

TEST(TensorConv2D, Basic) {
  constexpr int N = 10;
  constexpr int Cin = 5;
  constexpr int H = 16;
  constexpr int W = 17;
  constexpr int Cout = 7;
  constexpr int FH = 4;
  constexpr int FW = 3;
  constexpr int int_max = std::numeric_limits<int>::max();
  constexpr int int_min = std::numeric_limits<int>::min();
  auto input = torch::randint(int_min, int_max, {N, Cin, H, W}, at::kInt);
  auto weight = torch::randint(int_min, int_max, {Cout, Cin, FH, FW}, at::kInt);
  auto conv_options =
      nn_func::Conv2dFuncOptions().padding(torch::enumtype::kValid());
  auto output1 = nn_func::conv2d(input, weight);
  auto output = conv_2d(input, weight, {});
  EXPECT_EQ(output.sizes(), output1.sizes());
  bool allclose = torch::allclose(output, output1);
  EXPECT_TRUE(allclose);
  // std::cout << torch::isclose(output, output1) << std::endl;
  // std::cout << output1 << std::endl;
  // std::cout << output << std::endl;
}
