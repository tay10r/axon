#pragma once

#include <axon/Value.h>

#include <stdint.h>

namespace axon {

template<typename T, uint32_t Rows, uint32_t Cols>
struct Matrix final
{
  T data[Rows * Cols];

  [[nodiscard]] auto operator()(const uint32_t row, const uint32_t col) -> T& { return data[row * Cols + col]; }

  [[nodiscard]] auto operator()(const uint32_t row, const uint32_t col) const -> const T&
  {
    return data[row * Cols + col];
  }

  [[nodiscard]] auto operator[](const uint32_t index) -> T& { return data[index]; }

  [[nodiscard]] auto operator[](const uint32_t index) const -> const T& { return data[index]; }
};

} // namespace axon
