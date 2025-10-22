#include <iostream>
#include <random>

#include <axon/ExprPrinter.h>
#include <axon/Module.h>
#include <axon/ModuleBuilder.h>
#include <axon/Optimizer.h>
#include <axon/Value.h>

auto
main() -> int
{
  auto builder = axon::ModuleBuilder::create();
  auto w = builder->param();
  auto b = builder->param();
  auto x = builder->input(/*index=*/0);
  auto yPred = builder->add(builder->mul(x, w), b);

  // loss
  auto yTarget = builder->input(/*index=*/1);
  auto yError = builder->sub(yPred, yTarget);
  auto loss = builder->mul(yError, yError);

  auto m = builder->buildWithGrad(loss);

  // print for debugging
  axon::ExprPrinter printer(&std::cout);
  m->visit(printer);

  auto optim = axon::Optimizer::create();
  optim->prepare(*m);

  auto fn = [](const float x) -> float { return x * -5.2F + 0.3F; };

  std::mt19937 rng(0);

  std::uniform_real_distribution<float> x_dist(0, 1);

  const auto epochs = 10;

  for (int i = 0; i < epochs; i++) {

    const int samples = 100;

    optim->zeroGrad();

    float lossSum{ 0.0F };

    for (int j = 0; j < samples; j++) {

      const float x = x_dist(rng);

      const float input[2]{ x, fn(x) };

      optim->setInput(input, 2);

      lossSum += optim->exec(*m, loss);
    }

    std::cout << "epoch[" << i << "]: " << (lossSum / static_cast<float>(samples)) << std::endl;

    optim->step(/*lr=*/0.01F);
  }

  return 0;
}
