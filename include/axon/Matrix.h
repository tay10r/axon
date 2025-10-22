#pragma once

#include <axon/ModuleBuilder.h>
#include <axon/Value.h>

#include <stdint.h>

namespace axon {

template<typename T, uint32_t Rows, uint32_t Cols>
struct GenericMatrix final
{
  T data[Rows * Cols];

  [[nodiscard]] auto operator()(const uint32_t row, const uint32_t col) -> T& { return data[row * Cols + col]; }

  [[nodiscard]] auto operator()(const uint32_t row, const uint32_t col) const -> const T&
  {
    return data[row * Cols + col];
  }
};

template<uint32_t Dim>
[[nodiscard]] auto
dot(ModuleBuilder& builder, GenericMatrix<Value, Dim, 1>& a, GenericMatrix<Value, Dim, 1>& b) -> Value
{
  auto x = builder.constant(0.0F);

  for (uint32_t i = 0; i < Dim; i++) {

    const auto prod = builder.mul(a(i, 0), b(i, 0));

    x = builder.add(prod, x);
  }

  return x;
}

} // namespace axon
