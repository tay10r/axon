#include "OpenGL.h"

#include <axon/Exception.h>

namespace axon {

void
checkError(const std::source_location& location)
{
  const auto err = glGetError();
  switch (err) {
    case GL_NO_ERROR:
      break;
    case GL_INVALID_ENUM:
      throw Exception("invalid OpenGL enum", location);
    case GL_INVALID_OPERATION:
      throw Exception("invalid OpenGL operation", location);
    case GL_INVALID_VALUE:
      throw Exception("invalid OpenGL value", location);
    case GL_INVALID_INDEX:
      throw Exception("invalid OpenGL index", location);
    case GL_INVALID_FRAMEBUFFER_OPERATION:
      throw Exception("invalid OpenGL framebuffer operation", location);
    case GL_OUT_OF_MEMORY:
      throw Exception("OpenGL ran out of memory");
    default:
      throw Exception("unknown OpenGL error", location);
  }
}

} // namespace axon
