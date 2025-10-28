#include <axon/compiler.hpp>

void
compile(axon::Compiler& compiler)
{
  const auto u = axon::input();
  const auto v = axon::input();
  const auto uv = axon::Matrix<axon::Value, 2, 1>{ u, v };

  const auto u_f = axon::fourier_embed<6>(u);
  const auto v_f = axon::fourier_embed<6>(v);
  const auto input = axon::concat(axon::concat(uv, u_f), v_f);
  const auto wIn = axon::param<16, 26>();
  const auto x0 = relu(matmul(wIn, input));
  const auto x1 = relu(linear(x0));
  const auto x2 = relu(linear(x1));
  const auto x3 = relu(linear(x2));
  const auto rgb = matmul(axon::param<3, 16>(), x3) + axon::param<3, 1>();

  compiler.buildEvalModule({ rgb[0], rgb[1], rgb[2] });

  const auto target = axon::input<3, 1>();

  const auto loss = axon::mse(target, rgb);

  compiler.buildGradModule(loss);
}
