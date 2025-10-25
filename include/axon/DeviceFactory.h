#pragma once

#include <memory>
#include <string>
#include <vector>

#include <stddef.h>

namespace axon {

class Device;

class DeviceFactory
{
public:
  [[nodiscard]] static auto create() -> std::unique_ptr<DeviceFactory>;

  virtual ~DeviceFactory();

  [[nodiscard]] virtual auto options() const -> std::vector<std::string> = 0;

  [[nodiscard]] virtual auto createDevice(size_t deviceIndex) -> std::unique_ptr<Device> = 0;
};

} // namespace axon
