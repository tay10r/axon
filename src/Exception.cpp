#include <axon/Exception.h>

namespace axon {

Exception::Exception(const std::string& what, const std::source_location& location)
  : std::runtime_error(what)
  , m_location(location)
{
}

auto
Exception::location() const -> const std::source_location&
{
  return m_location;
}

} // namespace axon
