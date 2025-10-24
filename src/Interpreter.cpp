#include <axon/Interpreter.h>

#include <axon/Expr.h>
#include <axon/ExprVisitor.h>
#include <axon/Module.h>
#include <axon/Value.h>

#include <vector>

#include <math.h>
#include <string.h>

namespace axon {

namespace {

class InterpreterImpl final
  : public Interpreter
  , public ExprVisitor
{
public:
  InterpreterImpl(const Module& m, const float* parameters, float* gradient)
    : m_module(m.copy())
    , m_parameters(parameters)
    , m_gradient(gradient)
    , m_buffer(m_module->numExprs(), 0.0F)
  {
  }

  [[nodiscard]] auto getValue(const Value value) const -> float override { return m_buffer[value.index()]; }

  void exec(const float* input) override
  {
    m_input = input;

    m_ip = 0;

    m_module->visit(*this);

    m_input = nullptr;
  }

  void visit(const InputExpr& e) override { m_buffer.at(m_ip++) = m_input[e.index()]; }

  void visit(const ParamExpr& e) override { m_buffer.at(m_ip++) = m_parameters[e.index()]; }

  void visit(const ConstExpr& e) override { m_buffer.at(m_ip++) = e.value(); }

  void visit(const NegateExpr& e) override { m_buffer.at(m_ip++) = -m_buffer.at(e.operand()); }

  void visit(const RcpExpr& e) override { m_buffer.at(m_ip++) = 1.0F / m_buffer.at(e.operand()); }

  void visit(const SqrtExpr& e) override { m_buffer.at(m_ip++) = sqrtf(m_buffer.at(e.operand())); }

  void visit(const ExpExpr& e) override { m_buffer.at(m_ip++) = expf(m_buffer.at(e.operand())); }

  void visit(const ReLUExpr& e) override { m_buffer.at(m_ip++) = fmaxf(m_buffer.at(e.operand()), 0.0F); }

  void visit(const SigmoidExpr& e) override { m_buffer.at(m_ip++) = 1.0F / (1.0F + expf(-m_buffer.at(e.operand()))); }

  void visit(const HeavisideExpr& e) override
  {
    const auto x = m_buffer.at(e.operand());

    m_buffer.at(m_ip++) = (x > 0.0F) ? 1.0F : 0.0F;
  }

  void visit(const SinExpr& e) override { m_buffer.at(m_ip++) = sinf(m_buffer.at(e.operand())); }

  void visit(const CosExpr& e) override { m_buffer.at(m_ip++) = cosf(m_buffer.at(e.operand())); }

  void visit(const AddExpr& e) override { m_buffer.at(m_ip++) = m_buffer.at(e.left()) + m_buffer.at(e.right()); }

  void visit(const SubExpr& e) override { m_buffer.at(m_ip++) = m_buffer.at(e.left()) - m_buffer.at(e.right()); }

  void visit(const MulExpr& e) override { m_buffer.at(m_ip++) = m_buffer.at(e.left()) * m_buffer.at(e.right()); }

  void visit(const GradAddExpr& e) override
  {
    if (m_gradient) {
      m_gradient[e.paramIndex()] += m_buffer.at(e.valueIndex());
    }

    m_ip++;
  }

  void reset() { m_ip = 0; }

private:
  const float* m_input{};

  std::unique_ptr<Module> m_module;

  const float* m_parameters{};

  float* m_gradient{};

  std::vector<float> m_buffer;

  size_t m_ip = 0;
};

} // namespace

Interpreter::~Interpreter() = default;

auto
Interpreter::create(const Module& m, const float* parameters, float* gradient) -> std::unique_ptr<Interpreter>
{
  return std::make_unique<InterpreterImpl>(m, parameters, gradient);
}

} // namespace axon
