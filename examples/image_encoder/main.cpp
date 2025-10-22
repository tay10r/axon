#include <axon/Dataset.h>
#include <axon/ExprPrinter.h>
#include <axon/Interpreter.h>
#include <axon/Module.h>
#include <axon/ModuleBuilder.h>
#include <axon/Optimizer.h>
#include <axon/Value.h>

#include "deps/stb_image.h"
#include "deps/stb_image_write.h"

#include <algorithm>
#include <vector>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

namespace {

[[nodiscard]] auto
generateDataset(const char* filename) -> bool
{
  int w = 0;
  int h = 0;
  auto* img = stbi_load("sample.png", &w, &h, nullptr, 3);
  if (!img) {
    fprintf(stderr, "failed to load '%s'\n", filename);
    return false;
  }

  const auto numSamples = static_cast<size_t>(w * h);

  std::vector<float> samples(numSamples * 5, 0.0F);

  for (size_t i = 0; i < numSamples; i++) {

    const auto x = static_cast<int>(i) % w;
    const auto y = static_cast<int>(i) / w;

    const auto u = static_cast<float>(x) / static_cast<float>(w);
    const auto v = static_cast<float>(y) / static_cast<float>(h);

    auto* pixel = &img[i * 3];

    const auto r = static_cast<float>(pixel[0]) * (1.0F / 255.0F);
    const auto g = static_cast<float>(pixel[1]) * (1.0F / 255.0F);
    const auto b = static_cast<float>(pixel[2]) * (1.0F / 255.0F);

    auto* sample = samples.data() + i * 5;
    sample[0] = u; // input
    sample[1] = v;
    sample[2] = r; // target
    sample[3] = g;
    sample[4] = b;
  }

  stbi_image_free(img);

  return axon::Dataset::save(filename, samples.data(), numSamples, 5);
}

void
generateTestImage(axon::Interpreter& interp, axon::Value r, axon::Value g, axon::Value b)
{
  constexpr int w = 256;
  constexpr int h = 256;

  std::vector<uint8_t> img(w * h * 3);

  for (int i = 0; i < (w * h); i++) {

    const auto x = i % w;
    const auto y = i / w;

    const auto u = static_cast<float>(x) / static_cast<float>(w);
    const auto v = static_cast<float>(y) / static_cast<float>(h);

    const float input[2]{ u, v };

    interp.exec(input);

    const auto rOut = std::clamp(interp.getValue(r) * 255.0F, 0.0F, 255.0F);
    const auto gOut = std::clamp(interp.getValue(g) * 255.0F, 0.0F, 255.0F);
    const auto bOut = std::clamp(interp.getValue(b) * 255.0F, 0.0F, 255.0F);

    auto* dst = &img[i * 3];
    dst[0] = static_cast<uint8_t>(rOut);
    dst[1] = static_cast<uint8_t>(gOut);
    dst[2] = static_cast<uint8_t>(bOut);
  }

  stbi_write_png("result.png", w, h, 3, img.data(), w * 3);
}

} // namespace

auto
main() -> int
{
  if (!generateDataset("data.axd")) {
    fprintf(stderr, "failed to generate dataset\n");
    return EXIT_FAILURE;
  }

  auto data = axon::Dataset::create();

  if (!data->load("data.axd")) {
    fprintf(stderr, "failed to load dataset\n");
    return EXIT_FAILURE;
  }

  auto builder = axon::ModuleBuilder::create();
  const auto input_uv = axon::input<2, 1>(*builder);

  const auto w0 = axon::param<3, 2>(*builder);
  const auto w1 = axon::param<3, 3>(*builder);
  const auto w2 = axon::param<3, 3>(*builder);
  const auto w3 = axon::param<3, 3>(*builder);
  const auto x0 = matmul(*builder, w0, input_uv);
  const auto x1 = matmul(*builder, w1, x0);
  const auto x2 = matmul(*builder, w2, x1);
  const auto rgbOut = matmul(*builder, w3, x2);

  auto evalModule = builder->build();

  // train the network

  const auto target = axon::input<3, 1>(*builder);

  const auto loss = axon::mse(*builder, target, rgbOut);

  auto gradModule = builder->buildWithGrad(loss);

  auto optim = axon::Optimizer::create();
  if (!optim->prepare(*gradModule, *data)) {
    fprintf(stderr, "failed to prepare optimizer\n");
    return EXIT_FAILURE;
  }

  const auto epochs = 100;

  for (int i = 0; i < epochs; i++) {

    const int samples = 100'000;

    optim->zeroGrad();

    float lossSum{ 0.0F };

    for (int j = 0; j < samples; j++) {

      lossSum += optim->exec(loss);
    }

    printf("epoch[%d]: %f\n", i, (lossSum / static_cast<float>(samples)));

    optim->step(/*lr=*/0.01F / static_cast<float>(samples));
  }

  // load the eval network and test it

  std::vector<float> parameters(evalModule->numParameters(), 0.0F);

  optim->readParameters(parameters.data());

  auto interp = axon::Interpreter::create(*evalModule, parameters.data());

  generateTestImage(*interp, rgbOut[0], rgbOut[1], rgbOut[2]);

  return EXIT_SUCCESS;
}