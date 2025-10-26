#include <axon/Interpreter.h>

#include <axon/Exception.h>
#include <axon/Expr.h>
#include <axon/ExprVisitor.h>
#include <axon/Module.h>
#include <axon/Value.h>

#include <sstream>
#include <vector>

#include <assert.h>
#include <math.h>
#include <string.h>

#include "VFloat.h"

namespace axon {

namespace {

#define INVOKE_EXPR(expr)                                                                                              \
  do {                                                                                                                 \
                                                                                                                       \
    assert(m_ip < m_buffer.size());                                                                                    \
    m_buffer[m_ip] = (expr);                                                                                           \
    m_ip++;                                                                                                            \
  } while (0)

template<int N>
class InterpreterImpl final
  : public Interpreter
  , public ExprVisitor
{
public:
  using Float = VFloat<N>;

  InterpreterImpl(const Module& m, const float* parameters, float* gradient)
    : m_module(m.copy())
    , m_parameters(parameters)
    , m_gradient(gradient)
    , m_buffer(m_module->numExprs(), broadcast<N>(0.0F))
  {
  }

  [[nodiscard]] auto getValue(const Value value) const -> const float* override { return m_buffer[value.index()].data; }

  void exec(const float* input) override
  {
    m_input = input;

    m_ip = 0;

    m_module->visit(*this);

    m_input = nullptr;
  }

  void visit(const InputExpr& e) override { INVOKE_EXPR(Float::fromPtr(&m_input[e.index() * N])); }

  void visit(const ParamExpr& e) override { INVOKE_EXPR(broadcast<N>(m_parameters[e.index()])); }

  void visit(const ConstExpr& e) override { INVOKE_EXPR(broadcast<N>(e.value())); }

  void visit(const NegateExpr& e) override { INVOKE_EXPR(-m_buffer[e.operand()]); }

  void visit(const RcpExpr& e) override { INVOKE_EXPR(rcp(m_buffer[e.operand()])); }

  void visit(const SqrtExpr& e) override { INVOKE_EXPR(sqrt(m_buffer[e.operand()])); }

  void visit(const ExpExpr& e) override { INVOKE_EXPR(exp(m_buffer[e.operand()])); }

  void visit(const ReLUExpr& e) override { INVOKE_EXPR(relu(m_buffer[e.operand()])); }

  void visit(const SigmoidExpr& e) override { INVOKE_EXPR(sigmoid(m_buffer[e.operand()])); }

  void visit(const HeavisideExpr& e) override { INVOKE_EXPR(heaviside(m_buffer[e.operand()])); }

  void visit(const SinExpr& e) override { INVOKE_EXPR(sin(m_buffer[e.operand()])); }

  void visit(const CosExpr& e) override { INVOKE_EXPR(cos(m_buffer[e.operand()])); }

  void visit(const AddExpr& e) override { INVOKE_EXPR(m_buffer[e.left()] + m_buffer[e.right()]); }

  void visit(const SubExpr& e) override { INVOKE_EXPR(m_buffer[e.left()] - m_buffer[e.right()]); }

  void visit(const MulExpr& e) override { INVOKE_EXPR(m_buffer[e.left()] * m_buffer[e.right()]); }

  void visit(const GradAddExpr& e) override
  {
    m_gradient[e.paramIndex()] += m_buffer[e.valueIndex()].average();

    m_ip++;
  }

private:
  const float* m_input{};

  std::unique_ptr<Module> m_module;

  const float* m_parameters{};

  float* m_gradient{};

  std::vector<Float> m_buffer;

  size_t m_ip = 0;
};

} // namespace

Interpreter::~Interpreter() = default;

auto
Interpreter::create(const Module& m, const int batchSize, const float* parameters, float* gradient)
  -> std::unique_ptr<Interpreter>
{
  switch (batchSize) {
    case 1:
      return std::make_unique<InterpreterImpl<1>>(m, parameters, gradient);
    case 4:
      return std::make_unique<InterpreterImpl<4>>(m, parameters, gradient);
    case 8:
      return std::make_unique<InterpreterImpl<8>>(m, parameters, gradient);
    case 16:
      return std::make_unique<InterpreterImpl<16>>(m, parameters, gradient);
    default: {
      std::ostringstream stream;
      stream << "unsupported batch size of " << batchSize;
      throw Exception(stream.str());
    }
  }
}

} // namespace axon
