#pragma once

#include <math.h>

namespace axon {

template<int N>
struct VFloat final
{
  float data[N];

  [[nodiscard]] static auto fromPtr(const float* ptr) -> VFloat
  {
    VFloat result;
    for (int i = 0; i < N; i++) {
      result.data[i] = ptr[i];
    }
    return result;
  }

  [[nodiscard]] auto average() const -> float
  {
    auto result = 0.0F;
    for (int i = 0; i < N; i++) {
      result += data[i];
    }
    return result * (1.0F / static_cast<float>(N));
  }

  [[nodiscard]] auto operator+(const VFloat& other) const -> VFloat
  {
    VFloat result;
    for (int i = 0; i < N; i++) {
      result.data[i] = data[i] + other.data[i];
    }
    return result;
  }

  [[nodiscard]] auto operator-(const VFloat& other) const -> VFloat
  {
    VFloat result;
    for (int i = 0; i < N; i++) {
      result.data[i] = data[i] - other.data[i];
    }
    return result;
  }

  [[nodiscard]] auto operator-() const -> VFloat
  {
    VFloat result;

    for (int i = 0; i < N; i++) {
      result.data[i] = -data[i];
    }

    return result;
  }

  [[nodiscard]] auto operator*(const VFloat& other) const -> VFloat
  {
    VFloat result;
    for (int i = 0; i < N; i++) {
      result.data[i] = data[i] * other.data[i];
    }
    return result;
  }
};

template<int N>
[[nodiscard]] auto
rcp(const VFloat<N>& x) -> VFloat<N>
{
  VFloat<N> y;
  for (int i = 0; i < N; i++) {
    y.data[i] = 1.0F / x.data[i];
  }
  return y;
}

template<int N>
[[nodiscard]] auto
sqrt(const VFloat<N>& x) -> VFloat<N>
{
  VFloat<N> y;
  for (int i = 0; i < N; i++) {
    y.data[i] = sqrtf(x.data[i]);
  }
  return y;
}

template<int N>
[[nodiscard]] auto
exp(const VFloat<N>& x) -> VFloat<N>
{
  VFloat<N> y;
  for (int i = 0; i < N; i++) {
    y.data[i] = expf(x.data[i]);
  }
  return y;
}

template<int N>
[[nodiscard]] auto
sin(const VFloat<N>& x) -> VFloat<N>
{
  VFloat<N> y;
  for (int i = 0; i < N; i++) {
    y.data[i] = sinf(x.data[i]);
  }
  return y;
}

template<int N>
[[nodiscard]] auto
cos(const VFloat<N>& x) -> VFloat<N>
{
  VFloat<N> y;
  for (int i = 0; i < N; i++) {
    y.data[i] = cosf(x.data[i]);
  }
  return y;
}

template<int N>
[[nodiscard]] auto
relu(const VFloat<N>& x) -> VFloat<N>
{
  VFloat<N> y;
  for (int i = 0; i < N; i++) {
    y.data[i] = fmaxf(x.data[i], 0.0F);
  }
  return y;
}

template<int N>
[[nodiscard]] auto
sigmoid(const VFloat<N>& x) -> VFloat<N>
{
  VFloat<N> y;
  for (int i = 0; i < N; i++) {
    y.data[i] = 1.0F / (1.0F + expf(-x.data[i]));
  }
  return y;
}

template<int N>
[[nodiscard]] auto
heaviside(const VFloat<N>& x) -> VFloat<N>
{
  VFloat<N> y;
  for (int i = 0; i < N; i++) {
    y.data[i] = (x.data[i] > 0.0F) ? 1.0F : 0.0F;
  }
  return y;
}

template<int N>
[[nodiscard]] auto
broadcast(const float v) -> VFloat<N>
{
  VFloat<N> result;

  for (int i = 0; i < N; i++) {
    result.data[i] = v;
  }

  return result;
}

} // namespace axon
