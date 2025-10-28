#include <stdio.h>
#include <stdlib.h>

#include "image_encoder.h"

#include "deps/stb_image.h"
#include "deps/stb_image_write.h"

#include "random.h"

inline static uint8_t
unpack_channel(const float x)
{
  int xi = (int)(x * 255.0F);
  xi = (xi < 0) ? 0 : xi;
  xi = (xi > 255) ? 255 : xi;
  return (uint8_t)xi;
}

static void
predict(const float* parameters, const int epoch)
{
  const int w = 256;
  const int h = 256;
  uint8_t* pixels = malloc(w * h * 3);
  if (!pixels) {
    return;
  }

  for (int i = 0; i < (w * h); i++) {
    const int x = i % w;
    const int y = i / w;

    const float u = ((float)x) / ((float)w);
    const float v = ((float)y) / ((float)h);

    const float input[2] = { u, v };
    float output[3] = { 0, 0, 0 };

    axon_eval(parameters, input, output);

    uint8_t* dst = &pixels[i * 3];

    dst[0] = unpack_channel(output[0]);
    dst[1] = unpack_channel(output[1]);
    dst[2] = unpack_channel(output[2]);
  }

  char filename[256];
  snprintf(filename, sizeof(filename), "epoch_%04d.png", epoch);
  stbi_write_png(filename, w, h, 3, pixels, w * 3);
  free(pixels);
}

int
main()
{
  const int epochs = 20;
  const int train_samples = 1024 * 1024;
  const float lr = 0.01F;
  const float momentum = 0.9F;
  int w = 0;
  int h = 0;
  stbi_uc* pixels = stbi_load("sample.png", &w, &h, NULL, 3);
  if (!pixels) {
    fprintf(stderr, "failed to load image\n");
    return EXIT_FAILURE;
  }

  axon_rng_z rng;
  axon_rng_init(&rng, 0);

  axon_opt_z opt;
  axon_opt_init(&opt, 0);

  float parameters[AXON_PARAMETERS];
  axon_rng_float_array(&rng, parameters, AXON_PARAMETERS, 0.2F, -0.1F);

  float input[AXON_GRAD_INPUTS];

  for (int epoch = 0; epoch < epochs; epoch++) {

    for (int i = 0; i < train_samples; i++) {

      //const uint32_t x = axon_rng_range(&rng, 0, (uint32_t)w - 1);
      //const uint32_t y = axon_rng_range(&rng, 0, (uint32_t)h - 1);
      const int x = randint(0, w - 1);
      const int y = randint(0, h - 1);

      const float u = ((float)x) / ((float)w);
      const float v = ((float)y) / ((float)h);

      const uint32_t j = (y * ((uint32_t)w) + x) * 3;

      const stbi_uc* rgb = pixels + j;

      input[0] = u; // input
      input[1] = v;
      input[2] = ((float)rgb[0]) / 255.0F;
      input[3] = ((float)rgb[1]) / 255.0F;
      input[4] = ((float)rgb[2]) / 255.0F;

      axon_grad(parameters, input, opt.gradient);

      axon_opt_step(&opt, lr, momentum, parameters);
    }

    predict(parameters, epoch);

    printf("epoch[%d]\n", epoch);
  }

  stbi_image_free(pixels);

  return EXIT_SUCCESS;
}
