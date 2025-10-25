#pragma once

#include <memory>

namespace axon {

class DeviceBuffer;
class DeviceProgram;

class Device
{
public:
  virtual ~Device();

  virtual void makeCurrent() = 0;

  virtual auto createBuffer() -> std::shared_ptr<DeviceBuffer> = 0;

  virtual auto createRowSumProgram() -> std::unique_ptr<DeviceProgram> = 0;

  [[nodiscard]] virtual auto wait(unsigned int timeout = 5000) -> bool = 0;
};

} // namespace axon
