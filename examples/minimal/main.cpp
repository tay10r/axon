#include <iostream>
#include <random>

#include <stdlib.h>

#include <axon/Dataset.h>
#include <axon/Module.h>
#include <axon/ModuleBuilder.h>
#include <axon/Optimizer.h>
#include <axon/Value.h>

namespace {

class CustomData final : public axon::ProceduralData
{
public:
  void generate(float* row) override
  {
    auto fn = [](const float x) -> float { return x * -5.2F + 0.3F; };

    std::uniform_real_distribution<float> inputDist(-5, 5);

    std::uniform_real_distribution<float> noiseDist(-0.1F, 0.1F);

    row[0] = inputDist(m_rng);

    // y_target
    row[1] = fn(row[0]) + noiseDist(m_rng);
  }

private:
  std::mt19937 m_rng{ 0 };
};

} // namespace

auto
main() -> int
{
  auto builder = axon::ModuleBuilder::create();

  auto w = axon::param();
  auto b = axon::param();
  auto x = axon::Value::input();
  auto yPred = x * w + b;

  auto yTarget = axon::Value::input();
  const auto loss = axon::mse(yPred, yTarget);

  auto m = builder->buildWithGrad(loss);

  // generate training data
  CustomData generator;
  const int rows = 128;
  auto data = axon::Dataset::create(/*rows=*/rows, /*cols=*/2, generator);

  const int batchSize = 16;
  auto optim = axon::Optimizer::create(*m, *data, /*batchSize=*/batchSize, /*seed=*/0);

  const auto epochs = 10;

  for (int i = 0; i < epochs; i++) {

    const auto lossAvg = optim->runEpoch(loss);

    std::cout << "epoch[" << i << "]: " << lossAvg << std::endl;
  }

  return EXIT_SUCCESS;
}
