#pragma once

/* OpenGL ES 3.1 */
#include <GLES3/gl31.h>

#include <source_location>

namespace axon {

void
checkError(const std::source_location& location = std::source_location::current());

} // namespace axon
