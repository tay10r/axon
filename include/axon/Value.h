#pragma once

#include <stdint.h>

namespace axon {

class Value final
{
public:
  Value();

  explicit Value(uint32_t index);

  [[nodiscard]] auto index() const -> uint32_t;

  [[nodiscard]] auto valid() const -> bool;

private:
  uint32_t m_index;
};

} // namespace axon
