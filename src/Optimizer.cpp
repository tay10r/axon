#include <axon/Optimizer.h>

#include <axon/Dataset.h>
#include <axon/Expr.h>
#include <axon/ExprVisitor.h>
#include <axon/Interpreter.h>
#include <axon/Module.h>
#include <axon/Value.h>

#include <algorithm>
#include <limits>
#include <random>
#include <vector>

#include <assert.h>
#include <math.h>
#include <string.h>

namespace axon {

namespace {

class OptimizerImpl final : public Optimizer
{
public:
  OptimizerImpl()
    : m_rng(0)
  {
  }

  [[nodiscard]] auto prepare(const Module& m, const Dataset& dataset) -> bool override
  {
    if (m.numInputs() != dataset.cols()) {
      return false;
    }

    m_cols = m.numInputs();

    m_dataOffset = 0;

    m_data.resize(dataset.rows() * dataset.cols());

    memcpy(m_data.data(), dataset.data(), m_data.size() * sizeof(float));

    m_indices.resize(dataset.rows());

    for (size_t i = 0; i < m_indices.size(); i++) {
      m_indices[i] = static_cast<uint32_t>(i);
    }

    shuffleIndices();

    const auto n = m.numParameters();

    m_parameters.resize(n);

    m_gradient.resize(n, 0.0F);

    std::uniform_real_distribution<float> dist(-1, 1);

    for (size_t i = 0; i < n; i++) {
      m_parameters[i] = dist(m_rng);
    }

    m_interpreter = Interpreter::create(m, m_parameters.data(), m_gradient.data());

    return true;
  }

  [[nodiscard]] auto exec(Value loss) -> float override
  {
    assert(m_interpreter);

    if (!m_interpreter) {
      return std::numeric_limits<float>::infinity();
    }

    const float* data = m_data.data() + m_cols * m_indices[m_dataOffset];

    m_dataOffset = (m_dataOffset + 1) % static_cast<uint32_t>(m_indices.size());

    m_interpreter->exec(data);

    return m_interpreter->getValue(loss);
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

  void readParameters(float* parameters) override
  {
    memcpy(parameters, m_parameters.data(), m_parameters.size() * sizeof(float));
  }

protected:
  void shuffleIndices() { std::shuffle(m_indices.begin(), m_indices.end(), m_rng); }

private:
  std::vector<float> m_parameters;

  std::vector<float> m_gradient;

  std::unique_ptr<Interpreter> m_interpreter;

  std::vector<float> m_data;

  std::vector<uint32_t> m_indices;

  uint32_t m_dataOffset{};

  uint32_t m_cols{};

  std::mt19937 m_rng;
};

} // namespace

auto
Optimizer::create() -> std::unique_ptr<Optimizer>
{
  return std::make_unique<OptimizerImpl>();
}

Optimizer::~Optimizer() = default;

} // namespace axon
