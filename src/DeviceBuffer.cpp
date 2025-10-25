#include "DeviceBuffer.h"

#include <axon/Exception.h>

#include "OpenGL.h"

#include <string.h>

namespace axon {

namespace {

class DeviceBufferImpl final : public DeviceBuffer
{
public:
  DeviceBufferImpl() { glGenBuffers(1, &m_id); }

  ~DeviceBufferImpl() { glDeleteBuffers(1, &m_id); }

  void resize(const size_t size) override
  {
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_id);

    glBufferData(GL_SHADER_STORAGE_BUFFER, static_cast<GLsizeiptr>(size), nullptr, GL_DYNAMIC_DRAW);
  }

  void upload(const void* data, const size_t size, const size_t offset) override
  {
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_id);

    glBufferSubData(GL_SHADER_STORAGE_BUFFER, static_cast<GLintptr>(offset), static_cast<GLsizeiptr>(size), data);

    checkError();
  }

  void download(void* data, const size_t size, const size_t offset) override
  {
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_id);

    const auto* src = glMapBufferRange(
      GL_SHADER_STORAGE_BUFFER, static_cast<GLintptr>(offset), static_cast<GLsizeiptr>(size), GL_MAP_READ_BIT);

    checkError();

    memcpy(data, src, size);

    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
  }

  [[nodiscard]] auto size() const -> size_t override { return m_size; }

  [[nodiscard]] auto id() const -> GLuint { return m_id; }

private:
  GLuint m_id{};

  size_t m_size{};
};

} // namespace

auto
createDeviceBuffer() -> std::shared_ptr<DeviceBuffer>
{
  return std::make_shared<DeviceBufferImpl>();
}

DeviceBuffer::~DeviceBuffer() = default;

auto
getId(DeviceBuffer& deviceBuffer) -> GLuint
{
  return static_cast<DeviceBufferImpl&>(deviceBuffer).id();
}

} // namespace axon
