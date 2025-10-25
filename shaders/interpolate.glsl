#version 310 es

precision highp float;

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(std430, binding = 0) readonly buffer Alpha {
  float alpha[];
};

layout(std430, binding = 0) readonly buffer Beta {
  float beta[];
};

layout(std430, binding = 1) writeonly buffer Output {
    float gamma[];
};

layout(location = 0) uniform uint size;

layout(location = 1) uniform float k;

void
main()
{
  uint i = gl_GlobalInvocationID.x;

  if (i >= size) {
    return;
  }

  gamma[i] = alpha[i] * k + beta[i] * (1.0 - k);
}
