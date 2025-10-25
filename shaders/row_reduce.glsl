#version 310 es

precision highp float;

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(std430, binding = 0) readonly buffer Input {
    float inputData[];
};

layout(std430, binding = 1) writeonly buffer Output {
    float outputData[];
};

layout(location = 0) uniform uint rows;

layout(location = 1) uniform uint cols;

layout(location = 2) uniform float scale;

void
main()
{
  uint col = gl_GlobalInvocationID.x;

  if (col >= cols) {
    return;
  }

  float sum = 0.0;

  for (uint r = 0u; r < rows; ++r) {
    sum += inputData[r * cols + col];
  }

  outputData[col] = sum * scale;
}
