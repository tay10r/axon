#include <axon/Value.h>

#include <limits>

namespace axon {

Value::Value()
  : m_index(std::numeric_limits<uint32_t>::max())
{
}

Value::Value(const uint32_t index)
  : m_index(index)
{
}

auto
Value::index() const -> uint32_t
{
  return m_index;
}

auto
Value::valid() const -> bool
{
  return m_index == std::numeric_limits<uint32_t>::max();
}

} // namespace axon
