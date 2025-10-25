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
#include <iomanip>
#include <sstream>
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
generateTestImage(axon::Interpreter& interp, axon::Value r, axon::Value g, axon::Value b, const std::string& filename)
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

  stbi_write_png(filename.c_str(), w, h, 3, img.data(), w * 3);
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

  const auto u = axon::input();
  const auto v = axon::input();
  const auto uv = axon::Matrix<axon::Value, 2, 1>{ u, v };
  const auto u_f = axon::fourier_embed<6>(u);
  const auto v_f = axon::fourier_embed<6>(v);
  const auto input = axon::concat(axon::concat(uv, u_f), v_f);

  const auto wIn = axon::param<16, 26>();

  const auto x0 = relu(matmul(wIn, input));
  const auto x1 = relu(linear(x0));
  const auto x2 = relu(linear(x1));
  const auto x3 = relu(linear(x2));
  const auto wOut = axon::param<3, 16>();
  const auto bOut = axon::param<3, 1>();
  const auto rgbOut = matmul(wOut, x3) + bOut;

  auto evalModule = builder->build();

  // train the network

  const auto target = axon::input<3, 1>();

  const auto loss = axon::mse(target, rgbOut);

  auto gradModule = builder->buildWithGrad(loss);

  auto optim = axon::Optimizer::create();
  if (!optim->prepare(*gradModule, *data)) {
    fprintf(stderr, "failed to prepare optimizer\n");
    return EXIT_FAILURE;
  }

  const auto epochs = 20;

  std::vector<float> parameters(evalModule->numParameters(), 0.0F);

  for (int i = 0; i < epochs; i++) {

    const uint32_t batchSize = 16;

    float lossSum{ 0.0F };

    const auto numBatches = data->rows() / batchSize;

    for (uint32_t j = 0; j < numBatches; j++) {

      optim->zeroGrad();

      for (uint32_t k = 0; k < batchSize; k++) {
        lossSum += optim->exec(loss);
      }

      optim->step(/*lr=*/0.01F, /*momentum=*/0.1F);
    }

    optim->readParameters(parameters.data());

    auto interp = axon::Interpreter::create(*evalModule, parameters.data());

    std::ostringstream pathStream;
    pathStream << "result_" << std::setw(4) << std::setfill('0') << i << ".png";
    const auto path = pathStream.str();

    generateTestImage(*interp, rgbOut[0], rgbOut[1], rgbOut[2], path);

    printf("epoch[%d]: %f\n", i, (lossSum / static_cast<float>(data->rows())));
  }

  return EXIT_SUCCESS;
}
