#pragma once

#include <axon/Device.h>

#include <EGL/egl.h>
#include <EGL/eglext.h>

#include <memory>

namespace axon {

[[nodiscard]] auto
createDevice(EGLDeviceEXT device) -> std::unique_ptr<Device>;

} // namespace axon
