#pragma once

#include <source_location>
#include <stdexcept>

namespace axon {

class Exception final : public std::runtime_error
{
public:
  Exception(const std::string& what, const std::source_location& location = std::source_location::current());

  [[nodiscard]] auto location() const -> const std::source_location&;

private:
  std::source_location m_location;
};

} // namespace axon
