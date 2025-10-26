#pragma once

#include <memory>

namespace axon {

class Value;
class Module;

class Interpreter
{
public:
  [[nodiscard]] static auto create(const Module& m,
                                   const int batchSize,
                                   const float* parameters,
                                   float* gradient = nullptr) -> std::unique_ptr<Interpreter>;

  virtual ~Interpreter();

  [[nodiscard]] virtual auto getValue(Value value) const -> const float* = 0;

  virtual void exec(const float* input) = 0;
};

} // namespace axon
