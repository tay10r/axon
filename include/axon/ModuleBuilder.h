#pragma once

#include <axon/Matrix.h>

#include <memory>

#include <stdint.h>

namespace axon {

class Module;
class Value;

class ModuleBuilder
{
public:
  [[nodiscard]] static auto create() -> std::unique_ptr<ModuleBuilder>;

  virtual ~ModuleBuilder();

  /**
   * @brief Call this function for creating modules that can be used for shader code generation.
   * */
  [[nodiscard]] virtual auto build() -> std::unique_ptr<Module> = 0;

  /**
   * @brief Call this function for creating modules that can be used for training the network.
   * */
  [[nodiscard]] virtual auto buildWithGrad(Value loss) -> std::unique_ptr<Module> = 0;

  [[nodiscard]] virtual auto input() -> Value = 0;

  [[nodiscard]] virtual auto param() -> Value = 0;

  [[nodiscard]] virtual auto constant(float value) -> Value = 0;

  [[nodiscard]] virtual auto negate(Value operand) -> Value = 0;

  [[nodiscard]] virtual auto exp(Value operand) -> Value = 0;

  [[nodiscard]] virtual auto relu(Value operand) -> Value = 0;

  [[nodiscard]] virtual auto sigmoid(Value operand) -> Value = 0;

  [[nodiscard]] virtual auto heaviside(Value operand) -> Value = 0;

  [[nodiscard]] virtual auto sin(Value operand) -> Value = 0;

  [[nodiscard]] virtual auto cos(Value operand) -> Value = 0;

  [[nodiscard]] virtual auto add(Value left, Value right) -> Value = 0;

  [[nodiscard]] virtual auto sub(Value left, Value right) -> Value = 0;

  [[nodiscard]] virtual auto mul(Value left, Value right) -> Value = 0;
};

template<uint32_t R, uint32_t C>
[[nodiscard]] auto
input(ModuleBuilder& builder) -> Matrix<Value, R, C>
{
  Matrix<Value, R, C> result;

  for (uint32_t i = 0; i < (R * C); i++) {
    result.data[i] = Value(builder.input());
  }

  return result;
}

template<uint32_t R, uint32_t C>
[[nodiscard]] auto
param(ModuleBuilder& builder) -> Matrix<Value, R, C>
{
  Matrix<Value, R, C> result;

  for (uint32_t i = 0; i < (R * C); i++) {
    result.data[i] = Value(builder.param());
  }

  return result;
}

template<uint32_t R, uint32_t C>
[[nodiscard]] auto
negate(ModuleBuilder& builder, const Matrix<Value, R, C>& x) -> Matrix<Value, R, C>
{
  Matrix<Value, R, C> result;

  for (uint32_t i = 0; i < (R * C); i++) {
    result[i] = builder.negate(x[i]);
  }

  return result;
}

template<uint32_t R, uint32_t C>
[[nodiscard]] auto
exp(ModuleBuilder& builder, const Matrix<Value, R, C>& x) -> Matrix<Value, R, C>
{
  Matrix<Value, R, C> result;

  for (uint32_t i = 0; i < (R * C); i++) {
    result[i] = builder.exp(x[i]);
  }

  return result;
}

template<uint32_t R, uint32_t C>
[[nodiscard]] auto
heaviside(ModuleBuilder& builder, const Matrix<Value, R, C>& x) -> Matrix<Value, R, C>
{
  Matrix<Value, R, C> result;

  for (uint32_t i = 0; i < (R * C); i++) {
    result[i] = builder.heaviside(x[i]);
  }

  return result;
}

template<uint32_t R, uint32_t C>
[[nodiscard]] auto
add(ModuleBuilder& builder, const Matrix<Value, R, C>& a, const Matrix<Value, R, C>& b) -> Matrix<Value, R, C>
{
  Matrix<Value, R, C> result;

  for (uint32_t i = 0; i < (R * C); i++) {
    result[i] = builder.add(a[i], b[i]);
  }

  return result;
}

template<uint32_t R, uint32_t C>
[[nodiscard]] auto
sub(ModuleBuilder& builder, const Matrix<Value, R, C>& a, const Matrix<Value, R, C>& b) -> Matrix<Value, R, C>
{
  Matrix<Value, R, C> result;

  for (uint32_t i = 0; i < (R * C); i++) {
    result[i] = builder.sub(a[i], b[i]);
  }

  return result;
}

template<uint32_t Dim>
[[nodiscard]] auto
dot(ModuleBuilder& builder, const Matrix<Value, Dim, 1>& a, const Matrix<Value, Dim, 1>& b) -> Value
{
  auto x = builder.constant(0.0F);

  for (uint32_t i = 0; i < Dim; i++) {

    const auto prod = builder.mul(a[i], b[i]);

    x = builder.add(prod, x);
  }

  return x;
}

template<uint32_t R, uint32_t M, uint32_t C>
[[nodiscard]] auto
matmul(ModuleBuilder& builder, const Matrix<Value, R, M>& a, const Matrix<Value, M, C>& b) -> Matrix<Value, R, C>
{
  Matrix<Value, R, C> result;

  for (uint32_t i = 0; i < R; ++i) {

    for (uint32_t j = 0; j < C; ++j) {

      auto sum = builder.constant(0.0F);

      for (uint32_t k = 0; k < M; ++k) {

        const auto prod = builder.mul(a(i, k), b(k, j));

        sum = builder.add(sum, prod);
      }

      result(i, j) = sum;
    }
  }

  return result;
}

template<uint32_t R>
[[nodiscard]] auto
linear(ModuleBuilder& builder, const Matrix<Value, R, 1>& x, const bool bias = true) -> Matrix<Value, R, 1>
{
  const auto w = param<R, R>(builder);
  const auto x0 = matmul(builder, w, x);
  if (bias) {
    const auto b = param<R, 1>(builder);
    return add(builder, x0, b);
  } else {
    return x0;
  }
}

template<uint32_t R>
[[nodiscard]] auto
residual(ModuleBuilder& builder, const Matrix<Value, R, 1>& x) -> Matrix<Value, R, 1>
{
  const auto w = param<R, R>(builder);
  const auto b = param<R, 1>(builder);
  const auto y = add(builder, matmul(builder, w, x), b);
  return add(builder, relu(builder, x), y);
}

template<uint32_t Bands>
[[nodiscard]] auto
fourier_embed(ModuleBuilder& builder, Value value) -> Matrix<Value, Bands * 2, 1>
{
  Matrix<Value, Bands * 2, 1> result;

  float e = 1.0F;

  for (uint32_t i = 0; i < Bands; i++) {
    constexpr auto pi = 3.14159265359F;
    const auto f = builder.constant(2.0F * pi * e);
    const auto x = builder.mul(f, value);
    result[i * 2 + 0] = builder.sin(x);
    result[i * 2 + 1] = builder.cos(x);
    e *= 2.0F;
  }

  return result;
}

template<uint32_t R1, uint32_t R2>
[[nodiscard]] auto
concat(const Matrix<Value, R1, 1>& a, const Matrix<Value, R2, 1>& b) -> Matrix<Value, R1 + R2, 1>
{
  Matrix<Value, R1 + R2, 1> result;

  for (uint32_t i = 0; i < R1; i++) {
    result[i] = a[i];
  }

  for (uint32_t i = 0; i < R2; i++) {
    result[i + R1] = b[i];
  }

  return result;
}

//======================//
// Activation Functions //
//======================//

template<uint32_t R, uint32_t C>
[[nodiscard]] auto
relu(ModuleBuilder& builder, const Matrix<Value, R, C>& x) -> Matrix<Value, R, C>
{
  Matrix<Value, R, C> result;

  for (uint32_t i = 0; i < (R * C); i++) {
    result[i] = builder.relu(x[i]);
  }

  return result;
}

template<uint32_t R, uint32_t C>
[[nodiscard]] auto
sigmoid(ModuleBuilder& builder, const Matrix<Value, R, C>& x) -> Matrix<Value, R, C>
{
  Matrix<Value, R, C> result;

  for (uint32_t i = 0; i < (R * C); i++) {
    result[i] = builder.sigmoid(x[i]);
  }

  return result;
}

//================//
// Loss Functions //
//================//

template<uint32_t R, uint32_t C>
[[nodiscard]] auto
mse(ModuleBuilder& builder, const Matrix<Value, R, C>& a, const Matrix<Value, R, C>& b) -> Value
{
  const auto delta = sub(builder, a, b);

  auto sum = builder.constant(0.0F);

  for (uint32_t i = 0; i < (R * C); i++) {

    const auto product = builder.mul(delta[i], delta[i]);

    sum = builder.add(sum, product);
  }

  const auto scale = 1.0F / static_cast<float>(R * C);

  return builder.mul(sum, builder.constant(scale));
}

auto
mse(ModuleBuilder& builder, const Value a, const Value b) -> Value;

} // namespace axon
