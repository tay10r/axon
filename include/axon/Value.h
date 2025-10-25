#pragma once

#include <stdint.h>

namespace axon {

class ModuleBuilder;

class Value final
{
public:
  [[nodiscard]] static auto input() -> Value;

  [[nodiscard]] static auto param() -> Value;

  [[nodiscard]] static auto constant(float value) -> Value;

  Value();

  explicit Value(uint32_t index);

  [[nodiscard]] auto index() const -> uint32_t;

  [[nodiscard]] auto valid() const -> bool;

  [[nodiscard]] auto operator+(const Value& other) const -> Value;

  [[nodiscard]] auto operator-(const Value& other) const -> Value;

  [[nodiscard]] auto operator*(const Value& other) const -> Value;

  [[nodiscard]] auto operator-() const -> Value;

  [[nodiscard]] auto relu() const -> Value;

  [[nodiscard]] auto exp() const -> Value;

  [[nodiscard]] auto sigmoid() const -> Value;

  [[nodiscard]] auto heaviside() const -> Value;

  [[nodiscard]] auto sin() const -> Value;

  [[nodiscard]] auto cos() const -> Value;

private:
  uint32_t m_index;

  ModuleBuilder* m_builder;
};

} // namespace axon
