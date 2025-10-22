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
  auto* img = stbi_load(CMAKE_CURRENT_SOURCE_DIR "/examples/image_encode/sample.png", &w, &h, nullptr, 3);
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
  auto in_u = builder->input();
  auto in_v = builder->input();

  auto w0 = builder->param();
  auto w1 = builder->param();
  auto w2 = builder->param();
  auto w3 = builder->param();
  auto w4 = builder->param();
  auto w5 = builder->param();

  auto rOut = builder->mul(builder->mul(in_u, w0), builder->mul(in_v, w1));
  auto gOut = builder->mul(builder->mul(in_u, w2), builder->mul(in_v, w3));
  auto bOut = builder->mul(builder->mul(in_u, w4), builder->mul(in_v, w5));

  auto evalModule = builder->build();

  // train the network

  auto rTarget = builder->input();
  auto gTarget = builder->input();
  auto bTarget = builder->input();

  auto rDelta = builder->sub(rTarget, rOut);
  auto gDelta = builder->sub(gTarget, gOut);
  auto bDelta = builder->sub(bTarget, bOut);

  auto rDelta2 = builder->mul(rDelta, rDelta);
  auto gDelta2 = builder->mul(gDelta, gDelta);
  auto bDelta2 = builder->mul(bDelta, bDelta);

  auto loss = builder->add(builder->add(rDelta2, gDelta2), bDelta2);

  auto gradModule = builder->buildWithGrad(loss);

  auto optim = axon::Optimizer::create();
  if (!optim->prepare(*gradModule, *data)) {
    fprintf(stderr, "failed to prepare optimizer\n");
    return EXIT_FAILURE;
  }

  const auto epochs = 100;

  for (int i = 0; i < epochs; i++) {

    const int samples = 1000;

    optim->zeroGrad();

    float lossSum{ 0.0F };

    for (int j = 0; j < samples; j++) {

      lossSum += optim->exec(loss);
    }

    printf("epoch[%d]: %f\n", i, (lossSum / static_cast<float>(samples)));

    optim->step(/*lr=*/0.001F);
  }

  // load the eval network and test it

  std::vector<float> parameters(evalModule->numParameters(), 0.0F);

  optim->readParameters(parameters.data());

  auto interp = axon::Interpreter::create(*evalModule, parameters.data());

  generateTestImage(*interp, rOut, gOut, bOut);

  return EXIT_SUCCESS;
}
