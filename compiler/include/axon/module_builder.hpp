#pragma once

#include <axon/matrix.hpp>

#include <memory>
#include <string_view>
#include <vector>

#include <stdint.h>

namespace axon {

class Module;
class Value;

class ModuleBuilder
{
  friend Value;

public:
  [[nodiscard]] static auto current() -> ModuleBuilder*;

  static void setCurrent(ModuleBuilder* builder);

  [[nodiscard]] static auto create(bool setCurrent = true) -> std::unique_ptr<ModuleBuilder>;

  virtual ~ModuleBuilder();

  /**
   * @brief Call this function for creating modules that can be used for inference.
   * */
  [[nodiscard]] virtual auto build(const std::vector<Value>& outputs) -> std::unique_ptr<Module> = 0;

  /**
   * @brief Call this function for creating modules that can be used for training the network.
   * */
  [[nodiscard]] virtual auto buildWithGrad(Value loss) -> std::unique_ptr<Module> = 0;

protected:
  [[nodiscard]] virtual auto constant(float value) -> Value = 0;

  [[nodiscard]] virtual auto param(const std::string_view& name) -> Value = 0;

  [[nodiscard]] virtual auto input() -> Value = 0;

  [[nodiscard]] virtual auto add(Value left, Value right) -> Value = 0;

  [[nodiscard]] virtual auto sub(Value left, Value right) -> Value = 0;

  [[nodiscard]] virtual auto mul(Value left, Value right) -> Value = 0;

  [[nodiscard]] virtual auto relu(Value operand) -> Value = 0;

  [[nodiscard]] virtual auto negate(Value operand) -> Value = 0;

  [[nodiscard]] virtual auto exp(Value operand) -> Value = 0;

  [[nodiscard]] virtual auto sigmoid(Value operand) -> Value = 0;

  [[nodiscard]] virtual auto heaviside(Value operand) -> Value = 0;

  [[nodiscard]] virtual auto sin(Value operand) -> Value = 0;

  [[nodiscard]] virtual auto cos(Value operand) -> Value = 0;
};

[[nodiscard]] inline auto
constant(const float value) -> Value
{
  return Value::constant(value);
}

[[nodiscard]] inline auto
input() -> Value
{
  return Value::input();
}

template<uint32_t R, uint32_t C>
[[nodiscard]] auto
input() -> Matrix<Value, R, C>
{
  Matrix<Value, R, C> result;

  for (uint32_t i = 0; i < (R * C); i++) {
    result.data[i] = Value::input();
  }

  return result;
}

[[nodiscard]] inline auto
param(const std::string_view& name = "") -> Value
{
  return Value::param(name);
}

template<uint32_t R, uint32_t C>
[[nodiscard]] auto
param(const std::string_view& name = "") -> Matrix<Value, R, C>
{
  Matrix<Value, R, C> result;

  for (uint32_t i = 0; i < (R * C); i++) {
    result.data[i] = Value::param(name, i);
  }

  return result;
}

[[nodiscard]] inline auto
sin(const Value& value) -> Value
{
  return value.sin();
}

template<uint32_t R, uint32_t C>
[[nodiscard]] auto
sin(const Matrix<Value, R, C>& x) -> Matrix<Value, R, C>
{
  Matrix<Value, R, C> result;

  for (uint32_t i = 0; i < (R * C); i++) {
    result[i] = sin(x[i]);
  }

  return result;
}

[[nodiscard]] inline auto
cos(const Value& value) -> Value
{
  return value.cos();
}

template<uint32_t R, uint32_t C>
[[nodiscard]] auto
cos(const Matrix<Value, R, C>& x) -> Matrix<Value, R, C>
{
  Matrix<Value, R, C> result;

  for (uint32_t i = 0; i < (R * C); i++) {
    result[i] = cos(x[i]);
  }

  return result;
}

[[nodiscard]] inline auto
exp(const Value& value) -> Value
{
  return value.exp();
}

template<uint32_t R, uint32_t C>
[[nodiscard]] auto
exp(const Matrix<Value, R, C>& x) -> Matrix<Value, R, C>
{
  Matrix<Value, R, C> result;

  for (uint32_t i = 0; i < (R * C); i++) {
    result[i] = exp(x[i]);
  }

  return result;
}

[[nodiscard]] inline auto
heaviside(const Value& value) -> Value
{
  return value.heaviside();
}

template<uint32_t R, uint32_t C>
[[nodiscard]] auto
heaviside(const Matrix<Value, R, C>& x) -> Matrix<Value, R, C>
{
  Matrix<Value, R, C> result;

  for (uint32_t i = 0; i < (R * C); i++) {
    result[i] = heaviside(x[i]);
  }

  return result;
}

template<uint32_t Dim>
[[nodiscard]] auto
dot(const Matrix<Value, Dim, 1>& a, const Matrix<Value, Dim, 1>& b) -> Value
{
  auto x = constant(0.0F);

  for (uint32_t i = 0; i < Dim; i++) {
    x = (a[i] * b[i]) + x;
  }

  return x;
}

template<uint32_t R, uint32_t M, uint32_t C>
[[nodiscard]] auto
matmul(const Matrix<Value, R, M>& a, const Matrix<Value, M, C>& b) -> Matrix<Value, R, C>
{
  Matrix<Value, R, C> result;

  for (uint32_t i = 0; i < R; ++i) {

    for (uint32_t j = 0; j < C; ++j) {

      auto sum = constant(0.0F);

      for (uint32_t k = 0; k < M; ++k) {

        const auto prod = a(i, k) * b(k, j);

        sum = sum + prod;
      }

      result(i, j) = sum;
    }
  }

  return result;
}

template<uint32_t R>
[[nodiscard]] auto
linear(const Matrix<Value, R, 1>& x, const bool bias = true) -> Matrix<Value, R, 1>
{
  const auto x0 = matmul(param<R, R>(), x);
  if (bias) {
    return x0 + param<R, 1>();
  } else {
    return x0;
  }
}

template<uint32_t I, uint32_t O>
[[nodiscard]] auto
linear(const Matrix<Value, I, 1>& x, const bool bias = true) -> Matrix<Value, O, 1>
{
  const auto x0 = matmul(param<O, I>(), x);
  if (bias) {
    return x0 + param<O, 1>();
  } else {
    return x0;
  }
}

template<uint32_t R>
[[nodiscard]] auto
residual(const Matrix<Value, R, 1>& x) -> Matrix<Value, R, 1>
{
  const auto w = param<R, R>();
  const auto b = param<R, 1>();
  const auto y = matmul(w, x) + b;
  return relu(x) + y;
}

template<uint32_t Bands>
[[nodiscard]] auto
fourier_embed(Value value) -> Matrix<Value, Bands * 2, 1>
{
  Matrix<Value, Bands * 2, 1> result;

  float e = 1.0F;

  for (uint32_t i = 0; i < Bands; i++) {
    constexpr auto pi = 3.14159265359F;
    const auto f = constant(2.0F * pi * e);
    const auto x = f * value;
    result[i * 2 + 0] = sin(x);
    result[i * 2 + 1] = cos(x);
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

[[nodiscard]] inline auto
relu(const Value& value) -> Value
{
  return value.relu();
}

template<uint32_t R, uint32_t C>
[[nodiscard]] auto
relu(const Matrix<Value, R, C>& x) -> Matrix<Value, R, C>
{
  Matrix<Value, R, C> result;

  for (uint32_t i = 0; i < (R * C); i++) {
    result[i] = x[i].relu();
  }

  return result;
}

[[nodiscard]] inline auto
sigmoid(const Value& value) -> Value
{
  return value.sigmoid();
}

template<uint32_t R, uint32_t C>
[[nodiscard]] auto
sigmoid(const Matrix<Value, R, C>& x) -> Matrix<Value, R, C>
{
  Matrix<Value, R, C> result;

  for (uint32_t i = 0; i < (R * C); i++) {
    result[i] = sigmoid(x[i]);
  }

  return result;
}

//================//
// Loss Functions //
//================//

auto
mse(const Value a, const Value b) -> Value;

template<uint32_t R, uint32_t C>
[[nodiscard]] auto
mse(const Matrix<Value, R, C>& a, const Matrix<Value, R, C>& b) -> Value
{
  const auto delta = a - b;

  auto sum = constant(0.0F);

  for (uint32_t i = 0; i < (R * C); i++) {

    const auto product = delta[i] * delta[i];

    sum = sum + product;
  }

  const auto scale = 1.0F / static_cast<float>(R * C);

  return sum * constant(scale);
}

} // namespace axon
