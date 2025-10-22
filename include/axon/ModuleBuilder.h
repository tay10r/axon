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

//================//
// Loss Functions //
//================//

template<uint32_t R, uint32_t C>
[[nodiscard]] auto
mse(ModuleBuilder& builder, const Matrix<Value, R, C>& a, const Matrix<Value, R, C>& b) -> Value
{
  Matrix<Value, R, C> result;

  const auto delta = sub(builder, a, b);

  auto sum = builder.constant(0.0F);

  for (uint32_t i = 0; i < (R * C); i++) {

    const auto product = builder.mul(delta[i], delta[i]);

    sum = builder.add(sum, product);
  }

  const auto scale = 1.0F / static_cast<float>(R * C);

  return builder.mul(sum, builder.constant(scale));
}

} // namespace axon
