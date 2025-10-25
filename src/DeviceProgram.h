#pragma once

#include <axon/DeviceProgram.h>

namespace axon {

[[nodiscard]] auto
createDeviceProgram(const char* source, const size_t len) -> std::unique_ptr<DeviceProgram>;

} // namespace axon
