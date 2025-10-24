#pragma once

#include <memory>

namespace axon {

class Module;
class Value;
class Dataset;

class Optimizer
{
public:
  [[nodiscard]] static auto create(int seed = 0) -> std::unique_ptr<Optimizer>;

  virtual ~Optimizer();

  [[nodiscard]] virtual auto prepare(const Module& gradModule, const Dataset& data) -> bool = 0;

  [[nodiscard]] virtual auto exec(Value loss) -> float = 0;

  virtual void step(float lr = 0.001F, float momentum = 0.9F) = 0;

  virtual void zeroGrad() = 0;

  virtual void readParameters(float* parameters) = 0;
};

} // namespace axon
