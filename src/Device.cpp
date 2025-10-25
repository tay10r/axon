#include "Device.h"

#include <axon/Exception.h>

#include <sstream>

#include "DeviceBuffer.h"
#include "DeviceProgram.h"

#include <cmrc/cmrc.hpp>

CMRC_DECLARE(axon_shaders);

namespace axon {

namespace {

template<typename T>
auto
getProc(const char* name, const std::source_location& location = std::source_location::current()) -> T
{
  const auto result = reinterpret_cast<T>(eglGetProcAddress(name));
  if (!result) {
    std::ostringstream stream;
    stream << "failed to get EGL extension '" << name << "'";
    throw Exception(stream.str(), location);
  }
  return result;
}

class DeviceImpl final : public Device
{
public:
  DeviceImpl(EGLDeviceEXT device)
  {
    const auto getPlatformDisplay = getProc<PFNEGLGETPLATFORMDISPLAYEXTPROC>("eglGetPlatformDisplayEXT");

    m_display = getPlatformDisplay(EGL_PLATFORM_DEVICE_EXT, device, nullptr);
    if (!m_display) {
      throw Exception("failed to get platform display");
    }

    try {
      m_context = setup(m_display);
    } catch (...) {
      eglTerminate(m_display);
      m_display = nullptr;
      throw;
    }

    try {
      makeCurrent();
    } catch (...) {
      eglDestroyContext(m_display, m_context);
      eglTerminate(m_display);
      throw;
    }
  }

  ~DeviceImpl()
  {
    if (m_display != EGL_NO_DISPLAY) {

      if (m_context != EGL_NO_CONTEXT) {
        eglDestroyContext(m_display, m_context);
      }

      eglTerminate(m_display);
    }
  }

  void makeCurrent() override
  {
    if (!eglMakeCurrent(m_display, EGL_NO_SURFACE, EGL_NO_SURFACE, m_context)) {
      throw Exception("failed to make device context current");
    }
  }

  [[nodiscard]] auto createBuffer() -> std::shared_ptr<DeviceBuffer> override { return axon::createDeviceBuffer(); }

  [[nodiscard]] auto createRowSumProgram() -> std::unique_ptr<DeviceProgram> override
  {
    return createProgram("shaders/row_reduce.glsl");
  }

  [[nodiscard]] auto wait(unsigned int timeout) -> bool override
  {
    GLsync fence = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);

    const auto result = glClientWaitSync(fence, GL_SYNC_FLUSH_COMMANDS_BIT, GLuint64(timeout) * 100'000);

    glDeleteSync(fence);

    return (result == GL_CONDITION_SATISFIED) || (result == GL_ALREADY_SIGNALED);
  }

protected:
  [[nodiscard]] auto createProgram(const char* path) -> std::unique_ptr<DeviceProgram>
  {
    const auto fs = cmrc::axon_shaders::get_filesystem();
    const auto file = fs.open(path);
    return createDeviceProgram(file.begin(), file.size());
  }

  static auto setup(EGLDisplay display) -> EGLContext
  {
    if (!eglInitialize(display, nullptr, nullptr)) {
      throw Exception("failed to initialize EGL");
    }

    if (!eglBindAPI(EGL_OPENGL_ES_API)) {
      throw Exception("failed to bind OpenGL");
    }

    const EGLint contextAttribs[] = { EGL_CONTEXT_MAJOR_VERSION, 3, EGL_CONTEXT_MINOR_VERSION, 1, EGL_NONE };

    EGLContext ctx = eglCreateContext(display, EGL_NO_CONFIG_KHR, EGL_NO_CONTEXT, contextAttribs);
    if (ctx == EGL_NO_CONTEXT) {
      throw Exception("failed to create OpenGL context");
    }

    return ctx;
  }

private:
  EGLDisplay m_display{ EGL_NO_DISPLAY };

  EGLContext m_context{ EGL_NO_CONTEXT };
};

} // namespace

[[nodiscard]] auto
createDevice(EGLDeviceEXT device) -> std::unique_ptr<Device>
{
  return std::make_unique<DeviceImpl>(device);
}

Device::~Device() = default;

} // namespace axon
