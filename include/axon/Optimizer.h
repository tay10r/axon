#pragma once

#include <memory>

namespace axon {

class Module;
class Value;

class Optimizer
{
public:
  [[nodiscard]] static auto create() -> std::unique_ptr<Optimizer>;

  virtual ~Optimizer();

  virtual void prepare(const Module& m) = 0;

  virtual void setInput(const float* input, const size_t len) = 0;

  [[nodiscard]] virtual auto exec(const Module& m, Value loss) -> float = 0;

  virtual void step(float lr) = 0;

  virtual void zeroGrad() = 0;
};

} // namespace axon
