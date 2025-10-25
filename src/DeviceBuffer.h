#pragma once

#include <axon/DeviceBuffer.h>

#include <memory>

#include <GLES3/gl31.h>

namespace axon {

[[nodiscard]] auto
createDeviceBuffer() -> std::shared_ptr<DeviceBuffer>;

[[nodiscard]] auto
getId(DeviceBuffer& deviceBuffer) -> GLuint;

} // namespace axon
