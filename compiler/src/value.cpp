#include <axon/value.hpp>

#include <axon/module_builder.hpp>

#include <limits>
#include <sstream>

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
Value::param(const std::string_view& name) -> Value
{
  return ModuleBuilder::current()->param(name);
}

auto
Value::param(const std::string_view& name, const size_t index) -> Value
{
  // for arrays, we don't want to emit a name for each element

  const auto filteredName = (index == 0) ? name : std::string_view();

  return ModuleBuilder::current()->param(filteredName);
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
