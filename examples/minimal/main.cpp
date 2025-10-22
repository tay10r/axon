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

    std::uniform_real_distribution<float> dist(0, 1);
    // x
    row[0] = dist(m_rng);
    // y_target
    row[1] = fn(row[0]);
  }

private:
  std::mt19937 m_rng{ 0 };
};

} // namespace

auto
main() -> int
{
  auto builder = axon::ModuleBuilder::create();
  auto w = builder->param();
  auto b = builder->param();
  auto x = builder->input();
  auto yPred = builder->add(builder->mul(x, w), b);

  // loss
  auto yTarget = builder->input();
  auto yError = builder->sub(yPred, yTarget);
  auto loss = builder->mul(yError, yError);

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

  std::mt19937 rng(0);

  std::uniform_real_distribution<float> x_dist(0, 1);

  const auto epochs = 10;

  for (int i = 0; i < epochs; i++) {

    const int samples = 100;

    optim->zeroGrad();

    float lossSum{ 0.0F };

    for (int j = 0; j < samples; j++) {

      lossSum += optim->exec(loss);
    }

    std::cout << "epoch[" << i << "]: " << (lossSum / static_cast<float>(samples)) << std::endl;

    optim->step(/*lr=*/0.01F);
  }

  return EXIT_SUCCESS;
}
