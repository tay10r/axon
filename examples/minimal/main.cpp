#include <iostream>
#include <random>

#include <stdlib.h>

#include <axon/Dataset.h>
#include <axon/ExprPrinter.h>
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

  // print for debugging
  axon::ExprPrinter printer(&std::cout);
  m->visit(printer);

  // generate training data
  CustomData generator;
  auto data = axon::Dataset::create(/*rows=*/100, /*cols=*/2, generator);

  auto optim = axon::Optimizer::create();
  if (!optim->prepare(*m, *data)) {
    return EXIT_FAILURE;
  }

  const auto epochs = 10;

  for (int i = 0; i < epochs; i++) {

    const int samples = 100;

    optim->zeroGrad();

    float lossSum{ 0.0F };

    for (int j = 0; j < samples; j++) {

      lossSum += optim->exec(loss);
    }

    std::cout << "epoch[" << i << "]: " << (lossSum / static_cast<float>(samples)) << std::endl;

    optim->step(/*lr=*/0.01F, 0);
  }

  return EXIT_SUCCESS;
}
