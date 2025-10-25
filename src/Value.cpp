#include <axon/Value.h>

#include <axon/ModuleBuilder.h>

#include <limits>

namespace axon {

auto
Value::constant(const float value) -> Value
{
  return ModuleBuilder::current()->constant(value);
}

auto
Value::input() -> Value
{
  return ModuleBuilder::current()->input();
}

auto
Value::param() -> Value
{
  return ModuleBuilder::current()->param();
}

Value::Value()
  : m_index(std::numeric_limits<uint32_t>::max())
  , m_builder(nullptr)
{
}

Value::Value(const uint32_t index)
  : m_index(index)
  , m_builder(ModuleBuilder::current())
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

auto
Value::operator+(const Value& other) const -> Value
{
  return m_builder->add(*this, other);
}

auto
Value::operator-(const Value& other) const -> Value
{
  return m_builder->sub(*this, other);
}

auto
Value::operator-() const -> Value
{
  return m_builder->negate(*this);
}

auto
Value::operator*(const Value& other) const -> Value
{
  return m_builder->mul(*this, other);
}

auto
Value::relu() const -> Value
{
  return m_builder->relu(*this);
}

auto
Value::exp() const -> Value
{
  return m_builder->exp(*this);
}

auto
Value::sigmoid() const -> Value
{
  return m_builder->sigmoid(*this);
}

auto
Value::heaviside() const -> Value
{
  return m_builder->heaviside(*this);
}

auto
Value::sin() const -> Value
{
  return m_builder->sin(*this);
}

auto
Value::cos() const -> Value
{
  return m_builder->cos(*this);
}

} // namespace axon
