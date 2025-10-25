#pragma once

#include <stddef.h>

namespace axon {

class DeviceBuffer
{
public:
  virtual ~DeviceBuffer();

  virtual void resize(const size_t size) = 0;

  virtual void upload(const void* data, size_t size, size_t offset) = 0;

  virtual void download(void* data, size_t size, size_t offset) = 0;

  [[nodiscard]] virtual auto size() const -> size_t = 0;
};

} // namespace axon
