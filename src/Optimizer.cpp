#include <axon/Optimizer.h>

#include <axon/Expr.h>
#include <axon/ExprVisitor.h>
#include <axon/Module.h>
#include <axon/Value.h>

#include <limits>
#include <random>
#include <vector>

#include <assert.h>
#include <math.h>

namespace axon {

namespace {

class Executor final : public ExprVisitor
{
public:
  Executor(const Module& m, const float* parameters, float* gradient)
    : m_buffer(m.numExprs())
    , m_parameters(parameters)
    , m_gradient(gradient)
  {
    //
  }

  [[nodiscard]] auto getValue(const Value& value) const -> float { return m_buffer.at(value.index()); }

  void setInput(const float* input, const size_t len)
  {
    m_input.resize(len);

    for (size_t i = 0; i < len; i++) {
      m_input[i] = input[i];
    }
  }

  void visit(const InputExpr& e) override { m_buffer.at(m_ip++) = m_input.at(e.index()); }

  void visit(const ParamExpr& e) override { m_buffer.at(m_ip++) = m_parameters[e.index()]; }

  void visit(const ConstExpr& e) override { m_buffer.at(m_ip++) = e.value(); }

  void visit(const NegateExpr& e) override { m_buffer.at(m_ip++) = -m_buffer.at(e.operand()); }

  void visit(const RcpExpr& e) override { m_buffer.at(m_ip++) = 1.0F / m_buffer.at(e.operand()); }

  void visit(const SqrtExpr& e) override { m_buffer.at(m_ip++) = sqrtf(m_buffer.at(e.operand())); }

  void visit(const ExpExpr& e) override { m_buffer.at(m_ip++) = expf(m_buffer.at(e.operand())); }

  void visit(const MaxExpr& e) override { m_buffer.at(m_ip++) = fmaxf(m_buffer.at(e.left()), m_buffer.at(e.right())); }

  void visit(const AddExpr& e) override { m_buffer.at(m_ip++) = m_buffer.at(e.left()) + m_buffer.at(e.right()); }

  void visit(const SubExpr& e) override { m_buffer.at(m_ip++) = m_buffer.at(e.left()) - m_buffer.at(e.right()); }

  void visit(const MulExpr& e) override { m_buffer.at(m_ip++) = m_buffer.at(e.left()) * m_buffer.at(e.right()); }

  void visit(const GradAddExpr& e) override { m_gradient[e.paramIndex()] += m_buffer.at(e.valueIndex()); }

  void reset() { m_ip = 0; }

private:
  const float* m_parameters{};

  float* m_gradient{};

  std::vector<float> m_input;

  std::vector<float> m_buffer;

  size_t m_ip{};
};

class OptimizerImpl final : public Optimizer
{
public:
  void prepare(const Module& m) override
  {
    const auto n = m.numParameters();

    m_parameters.resize(n);

    m_gradient.resize(n, 0.0F);

    std::mt19937 rng(0);

    std::uniform_real_distribution<float> dist(-1, 1);

    for (size_t i = 0; i < n; i++) {
      m_parameters[i] = dist(rng);
    }

    m_executor.reset(new Executor(m, m_parameters.data(), m_gradient.data()));
  }

  void setInput(const float* input, const size_t len) override
  {
    assert(m_executor);

    if (!m_executor) {
      return;
    }

    m_executor->setInput(input, len);
  }

  [[nodiscard]] auto exec(const Module& m, Value loss) -> float override
  {
    assert(m_executor);

    if (!m_executor) {
      return std::numeric_limits<float>::infinity();
    }

    m_executor->reset();

    m.visit(*m_executor);

    return m_executor->getValue(loss);
  }

  void step(const float lr) override
  {
    for (size_t i = 0; i < m_parameters.size(); i++) {
      m_parameters[i] -= m_gradient[i] * lr;
    }
  }

  void zeroGrad() override
  {
    for (size_t i = 0; i < m_gradient.size(); i++) {
      m_gradient[i] = 0.0F;
    }
  }

private:
  std::vector<float> m_parameters;

  std::vector<float> m_gradient;

  std::unique_ptr<Executor> m_executor;
};

} // namespace

auto
Optimizer::create() -> std::unique_ptr<Optimizer>
{
  return std::make_unique<OptimizerImpl>();
}

Optimizer::~Optimizer() = default;

} // namespace axon
