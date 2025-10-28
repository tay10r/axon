#include <axon/compiler.hpp>

#include <axon/exception.hpp>
#include <axon/module.hpp>
#include <axon/module_builder.hpp>

namespace axon {

namespace {

class CompilerImpl final : public Compiler
{
public:
  explicit CompilerImpl(const Options& options)
    : m_options(options)
  {
  }

  void buildEvalModule(const std::vector<Value>& outputs) override
  {
    if (m_evalModule) {
      throw Exception("eval model already created");
    }

    m_evalModule = m_builder->build(outputs);
  }

  void buildGradModule(const Value& loss) override
  {
    if (m_gradModule) {
      throw Exception("grad module already created");
    }

    m_gradModule = m_builder->buildWithGrad(loss);
  }

  [[nodiscard]] auto getEvalModule() const -> const Module* override { return m_evalModule.get(); }

  [[nodiscard]] auto getGradModule() const -> const Module* override { return m_gradModule.get(); }

private:
  std::unique_ptr<ModuleBuilder> m_builder{ ModuleBuilder::create() };

  std::unique_ptr<Module> m_gradModule;

  std::unique_ptr<Module> m_evalModule;

  Options m_options;
};

} // namespace

auto
Compiler::create(const Options& options) -> std::unique_ptr<Compiler>
{
  return std::make_unique<CompilerImpl>(options);
}

} // namespace axon
