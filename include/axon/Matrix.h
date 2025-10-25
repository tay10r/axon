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

  [[nodiscard]] auto operator+(const Matrix<T, Rows, Cols>& other) const -> Matrix<T, Rows, Cols>
  {
    Matrix<T, Rows, Cols> result;

    for (uint32_t i = 0; i < (Rows * Cols); i++) {
      result[i] = data[i] + other[i];
    }

    return result;
  }

  [[nodiscard]] auto operator-(const Matrix<T, Rows, Cols>& other) const -> Matrix<T, Rows, Cols>
  {
    Matrix<T, Rows, Cols> result;

    for (uint32_t i = 0; i < (Rows * Cols); i++) {
      result[i] = data[i] - other[i];
    }

    return result;
  }

  template<uint32_t R, uint32_t C>
  [[nodiscard]] auto operator-() const -> Matrix<Value, R, C>
  {
    Matrix<Value, R, C> result;

    for (uint32_t i = 0; i < (R * C); i++) {
      result[i] = -data[i];
    }

    return result;
  }
};

} // namespace axon
