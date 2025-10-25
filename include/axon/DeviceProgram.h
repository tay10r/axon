#pragma once

#include <memory>

#include <stddef.h>

namespace axon {

class DeviceBuffer;

class DeviceProgram
{
public:
  virtual ~DeviceProgram();

  virtual void bindBuffer(size_t index, std::shared_ptr<DeviceBuffer> buffer) = 0;

  virtual void setUniform(size_t index, unsigned int value) = 0;

  virtual void setUniform(size_t index, float value) = 0;

  virtual void invoke(const size_t n) = 0;
};

} // namespace axon
