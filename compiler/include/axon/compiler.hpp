#pragma once

#ifndef AXON_COMPILER_NO_EXTRA_INCLUDES
#include <axon/exception.hpp>
#include <axon/module_builder.hpp>
#endif

#include <memory>
#include <string>

namespace axon {

class Module;
class Value;

class Compiler
{
public:
  struct Options final
  {
    std::string networkName{ "net" };

    std::string outputFile{ "net.h" };

    bool release{ false };

    std::string parametersPath{ "params.bin" };

    std::string exporter{ "c" };
  };

  [[nodiscard]] static auto create(const Options& options) -> std::unique_ptr<Compiler>;

  virtual ~Compiler() = default;

  virtual void buildEvalModule(const std::vector<Value>& outputs) = 0;

  virtual void buildGradModule(const Value& loss) = 0;

  [[nodiscard]] virtual auto getEvalModule() const -> const Module* = 0;

  [[nodiscard]] virtual auto getGradModule() const -> const Module* = 0;
};

} // namespace axon

#ifndef AXON_COMPILER_NO_ENTRY_POINT

void
compile(axon::Compiler& compiler);

#endif /* AXON_COMPILER_NO_ENTRY_POINT */
