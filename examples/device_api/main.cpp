#include <axon/Device.h>
#include <axon/DeviceBuffer.h>
#include <axon/DeviceFactory.h>
#include <axon/DeviceProgram.h>
#include <axon/Exception.h>

#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include <stdlib.h>

namespace {

auto
randomVec(const size_t n) -> std::vector<float>
{
  std::mt19937 rng(0);
  std::uniform_real_distribution<float> dist(0.0F, 50.0F);
  std::vector<float> result(n);
  for (size_t i = 0; i < n; i++) {
    result[i] = dist(rng);
  }
  return result;
}

void
printTable(const size_t rows, const size_t cols, const float* data)
{
  for (size_t y = 0; y < rows; y++) {
    for (size_t x = 0; x < cols; x++) {
      const auto value = data[y * cols + x];
      std::cout << std::setfill('0') << std::setprecision(4) << std::fixed << value << ' ';
    }
    std::cout << std::endl;
  }
}

void
runExample()
{
  auto deviceFactory = axon::DeviceFactory::create();

  const auto options = deviceFactory->options();

  if (options.empty()) {
    std::cerr << "no devices found" << std::endl;
    return;
  }

  std::cout << "available devices:" << std::endl;
  for (const auto& opt : options) {
    std::cout << "  - \"" << opt << "\"" << std::endl;
  }

  auto device = deviceFactory->createDevice(/*deviceIndex=*/0);

  const auto rows = 4u;
  const auto cols = 8u;

  const auto inputData = randomVec(rows * cols);
  auto inputBuffer = device->createBuffer();
  inputBuffer->resize(rows * cols * sizeof(float));
  inputBuffer->upload(inputData.data(), inputData.size() * sizeof(float), 0);

  auto outputBuffer = device->createBuffer();
  outputBuffer->resize(cols * sizeof(float));

  auto rowSum = device->createRowSumProgram();
  // for now, you'll have to look up the shader source code
  // to find out what the right indices are for the buffers
  // and the uniforms.
  rowSum->bindBuffer(0, inputBuffer);
  rowSum->bindBuffer(1, outputBuffer);
  rowSum->setUniform(0, rows);
  rowSum->setUniform(1, cols);
  rowSum->setUniform(2, 1.0F / static_cast<float>(rows));
  rowSum->invoke(rows);

  if (!device->wait()) {
    std::cerr << "device timeout occurred";
    return;
  }

  std::vector<float> outputData(cols);
  outputBuffer->download(outputData.data(), cols * sizeof(float), 0);

  std::cout << "input:" << std::endl;
  printTable(rows, cols, inputData.data());

  std::cout << std::endl;

  std::cout << "output:" << std::endl;
  printTable(1, cols, outputData.data());
}

} // namespace

auto
main() -> int
{
  try {
    runExample();
  } catch (const axon::Exception& ex) {
    std::cerr << "exception at " << ex.location().file_name() << " line " << ex.location().line() << std::endl;
    std::cerr << "  " << ex.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
