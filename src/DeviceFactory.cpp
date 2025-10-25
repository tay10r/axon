#include <axon/DeviceFactory.h>

#include <EGL/egl.h>
#include <EGL/eglext.h>

#include "Device.h"

#include <array>

namespace axon {

namespace {

constexpr size_t MaxDevices = 16;

class DeviceFactoryImpl final : public DeviceFactory
{
public:
  DeviceFactoryImpl()
  {
    m_queryDevices = reinterpret_cast<PFNEGLQUERYDEVICESEXTPROC>(eglGetProcAddress("eglQueryDevicesEXT"));

    m_queryDeviceString =
      reinterpret_cast<PFNEGLQUERYDEVICESTRINGEXTPROC>(eglGetProcAddress("eglQueryDeviceStringEXT"));

    if (m_queryDevices) {
      m_queryDevices(MaxDevices, m_devices.data(), &m_numDevices);
    }
  }

  [[nodiscard]] auto options() const -> std::vector<std::string> override
  {
    if (!m_queryDevices || !m_queryDeviceString) {
      return {};
    }

    std::vector<std::string> options;

    for (EGLint i = 0; i < m_numDevices; i++) {

      const auto option = m_queryDeviceString(m_devices[static_cast<size_t>(i)], EGL_VENDOR);
      if (!option) {
        continue;
      }

      options.emplace_back(option);
    }

    return options;
  }

  [[nodiscard]] auto createDevice(const size_t deviceIndex) -> std::unique_ptr<Device> override
  {
    return axon::createDevice(m_devices.at(deviceIndex));
  }

private:
  PFNEGLQUERYDEVICESEXTPROC m_queryDevices{};

  PFNEGLQUERYDEVICESTRINGEXTPROC m_queryDeviceString{};

  std::array<EGLDeviceEXT, MaxDevices> m_devices{};

  EGLint m_numDevices{};
};

} // namespace

auto
DeviceFactory::create() -> std::unique_ptr<DeviceFactory>
{
  return std::make_unique<DeviceFactoryImpl>();
}

DeviceFactory::~DeviceFactory() = default;

} // namespace axon
