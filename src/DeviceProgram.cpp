#include "DeviceProgram.h"

#include "DeviceBuffer.h"

#include "OpenGL.h"

#include <axon/Exception.h>

#include <map>

#include <string.h>

namespace axon {

namespace {

class DeviceProgramImpl final : public DeviceProgram
{
public:
  DeviceProgramImpl(const char* source, const size_t len)
    : m_id(glCreateProgram())
  {
    const auto shader = glCreateShader(GL_COMPUTE_SHADER);

    const auto sourceLen = static_cast<GLint>(len);

    glShaderSource(shader, 1, &source, &sourceLen);

    glCompileShader(shader);

    GLint status = 0;

    glGetShaderiv(shader, GL_COMPILE_STATUS, &status);

    if (status != GL_TRUE) {

      GLint logLength = 0;
      glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &logLength);

      std::string buf;
      buf.resize(static_cast<size_t>(logLength));

      GLsizei readSize{};
      glGetShaderInfoLog(shader, static_cast<GLsizei>(logLength), &readSize, buf.data());
      buf.resize(static_cast<size_t>(readSize));

      glDeleteProgram(m_id);
      glDeleteShader(shader);
      throw Exception(buf);
    }

    glAttachShader(m_id, shader);

    glLinkProgram(m_id);

    glDetachShader(m_id, shader);

    glDeleteShader(shader);

    GLint linkStatus = 0;

    glGetProgramiv(m_id, GL_LINK_STATUS, &linkStatus);

    if (linkStatus != GL_TRUE) {
      GLint logLength = 0;
      glGetProgramiv(m_id, GL_INFO_LOG_LENGTH, &logLength);

      std::string buf;
      buf.resize(static_cast<size_t>(logLength));

      GLsizei readSize{};
      glGetProgramInfoLog(m_id, static_cast<GLsizei>(logLength), &readSize, buf.data());

      buf.resize(static_cast<size_t>(readSize));

      glDeleteProgram(m_id);
      throw Exception(buf);
    }
  }

  ~DeviceProgramImpl() { glDeleteProgram(m_id); }

  void bindBuffer(const size_t index, std::shared_ptr<DeviceBuffer> buffer) override
  {
    set(m_buffers, index, std::move(buffer));
  }

  void setUniform(const size_t index, const unsigned int value) override { set(m_uintUniforms, index, value); }

  void setUniform(const size_t index, const float value) override { set(m_floatUniforms, index, value); }

  void invoke(const size_t n) override
  {
    glUseProgram(m_id);

    for (auto& v : m_uintUniforms) {
      glUniform1ui(static_cast<GLint>(v.first), v.second);
    }

    for (auto& v : m_floatUniforms) {
      glUniform1f(static_cast<GLint>(v.first), v.second);
    }

    for (auto& v : m_buffers) {

      const auto bufferId = getId(*v.second);

      glBindBuffer(GL_SHADER_STORAGE_BUFFER, bufferId);

      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, static_cast<GLuint>(v.first), bufferId);
    }

    const auto localSize = 256u;

    const auto xGroups = (n + (localSize - 1u)) / localSize;

    glDispatchCompute(static_cast<GLuint>(xGroups), 1, 1);

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    checkError();
  }

protected:
  template<typename T, typename V>
  void set(std::map<size_t, T>& m, size_t index, V value)
  {
    auto it = m.find(index);
    if (it == m.end()) {
      m.emplace(index, std::move(value));
    } else {
      it->second = std::move(value);
    }
  }

private:
  std::map<size_t, std::shared_ptr<DeviceBuffer>> m_buffers;

  std::map<size_t, unsigned int> m_uintUniforms;

  std::map<size_t, float> m_floatUniforms;

  GLuint m_id{};
};

} // namespace

[[nodiscard]] auto
createDeviceProgram(const char* source, const size_t len) -> std::unique_ptr<DeviceProgram>
{
  return std::make_unique<DeviceProgramImpl>(source, len);
}

DeviceProgram::~DeviceProgram() = default;

} // namespace axon
