#pragma once

#include <memory>

namespace axon {

class Module;
class Value;
class Dataset;

class Optimizer
{
public:
  [[nodiscard]] static auto create(const Module& gradModule, const Dataset& dataset, int batchSize, int seed = 0)
    -> std::unique_ptr<Optimizer>;

  virtual ~Optimizer();

  [[maybe_unused]] virtual auto step(const Value& loss, float lr = 0.001F, float momentum = 0.9F) -> float = 0;

  [[maybe_unused]] virtual auto runEpoch(const Value& loss, float lr = 0.001F, float momentum = 0.9F) -> float = 0;

  virtual void readParameters(float* parameters) = 0;
};

} // namespace axon
