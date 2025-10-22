#pragma once

#include <memory>

#include <stdint.h>

namespace axon {

class Module;
class Value;

class ModuleBuilder
{
public:
  [[nodiscard]] static auto create() -> std::unique_ptr<ModuleBuilder>;

  virtual ~ModuleBuilder();

  /**
   * @brief Call this function for creating modules that can be used for shader code generation.
   * */
  [[nodiscard]] virtual auto build() -> std::unique_ptr<Module> = 0;

  /**
   * @brief Call this function for creating modules that can be used for training the network.
   * */
  [[nodiscard]] virtual auto buildWithGrad(Value loss) -> std::unique_ptr<Module> = 0;

  [[nodiscard]] virtual auto input() -> Value = 0;

  [[nodiscard]] virtual auto param() -> Value = 0;

  [[nodiscard]] virtual auto add(Value left, Value right) -> Value = 0;

  [[nodiscard]] virtual auto sub(Value left, Value right) -> Value = 0;

  [[nodiscard]] virtual auto mul(Value left, Value right) -> Value = 0;
};

} // namespace axon
