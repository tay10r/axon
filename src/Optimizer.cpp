#include <axon/Optimizer.h>

#include <axon/Dataset.h>
#include <axon/Exception.h>
#include <axon/Expr.h>
#include <axon/ExprVisitor.h>
#include <axon/Interpreter.h>
#include <axon/Module.h>
#include <axon/Value.h>

#include <algorithm>
#include <random>
#include <sstream>
#include <vector>

#include <assert.h>
#include <math.h>
#include <string.h>

#include "VFloat.h"

namespace axon {

namespace {

template<int N>
class OptimizerImpl final : public Optimizer
{
public:
  using Float = VFloat<N>;

  explicit OptimizerImpl(const Module& m, const Dataset& dataset, const int seed)
    : m_rng(static_cast<unsigned int>(seed))
  {
    if (m.numInputs() != dataset.cols()) {
      std::ostringstream stream;
      stream << "The number of inputs to the network (" << m.numInputs() << ")";
      stream << " does not match the number of columns in the dataset (" << dataset.cols() << ")";
      throw Exception(stream.str());
    }

    if ((dataset.rows() % N) != 0) {
      std::ostringstream stream;
      stream << "The number of rows in the dataset (" << dataset.rows() << ") should be divisible";
      stream << " by the batch size (" << N << ")";
      throw Exception(stream.str());
    }

    if (dataset.rows() == 0) {
      throw Exception("dataset is empty");
    }

    m_cols = dataset.cols();

    m_rowOffset = 0;

    m_data.resize(dataset.rows() * dataset.cols());

    memcpy(m_data.data(), dataset.data(), m_data.size() * sizeof(float));

    m_input.resize(N * m.numInputs());

    m_indices.resize(dataset.rows());

    for (size_t i = 0; i < m_indices.size(); i++) {
      m_indices[i] = static_cast<uint32_t>(i);
    }

    shuffleIndices();

    const auto n = m.numParameters();

    m_parameters.resize(n);

    m_gradient.resize(n, 0.0F);

    m_momentum.resize(n, 0.0F);

    initializeParams();

    m_interpreter = Interpreter::create(m, N, m_parameters.data(), m_gradient.data());
  }

  [[nodiscard]] auto step(const Value& loss, const float lr, const float momentum) -> float override
  {
    zeroGrad();

    for (uint32_t column = 0; column < m_cols; column++) {

      auto* dstPtr = &m_input[column * N];

      for (int i = 0; i < N; i++) {

        const auto* row = m_data.data() + m_cols * m_indices[m_rowOffset + static_cast<uint32_t>(i)];

        dstPtr[i] = row[column];
      }
    }

    m_rowOffset = (m_rowOffset + N) % static_cast<uint32_t>(m_indices.size());

    m_interpreter->exec(m_input.data());

    for (size_t i = 0; i < m_momentum.size(); i++) {

      auto g = m_gradient[i];

      const auto gradient_clamp = false;

      if (gradient_clamp) {
        g = std::clamp(g, -1.0F, 1.0F);
      }

      m_momentum[i] = m_momentum[i] * momentum + g * (1.0F - momentum);
    }

    for (size_t i = 0; i < m_parameters.size(); i++) {
      m_parameters[i] -= m_momentum[i] * lr;
    }

    return Float::fromPtr(m_interpreter->getValue(loss)).average();
  }

  [[maybe_unused]] auto runEpoch(const Value& loss, const float lr, const float momentum) -> float override
  {
    const auto numSteps = m_indices.size() / N;

    auto lossSum = 0.0F;

    for (int i = 0; i < static_cast<int>(numSteps); i++) {
      lossSum += step(loss, lr, momentum);
    }

    return lossSum * (1.0F / static_cast<float>(N));
  }

  void readParameters(float* parameters) override
  {
    memcpy(parameters, m_parameters.data(), m_parameters.size() * sizeof(float));
  }

protected:
  void zeroGrad()
  {
    for (size_t i = 0; i < m_gradient.size(); i++) {
      m_gradient[i] = 0.0F;
    }
  }

  void shuffleIndices() { std::shuffle(m_indices.begin(), m_indices.end(), m_rng); }

  void initializeParams()
  {
    std::normal_distribution<float> dist(0, 1.0F);

    const auto stddev = 0.1F;

    for (size_t i = 0; i < m_parameters.size(); i++) {
      m_parameters[i] = dist(m_rng) * stddev;
    }
  }

private:
  std::vector<float> m_parameters;

  std::vector<float> m_gradient;

  std::vector<float> m_momentum;

  std::unique_ptr<Interpreter> m_interpreter;

  std::vector<float> m_data;

  std::vector<uint32_t> m_indices;

  uint32_t m_rowOffset{};

  std::vector<float> m_input;

  uint32_t m_cols{};

  std::mt19937 m_rng;
};

} // namespace

auto
Optimizer::create(const Module& gradModule, const Dataset& dataset, const int batchSize, const int seed)
  -> std::unique_ptr<Optimizer>
{
  switch (batchSize) {
    case 1:
      return std::make_unique<OptimizerImpl<1>>(gradModule, dataset, seed);
    case 4:
      return std::make_unique<OptimizerImpl<4>>(gradModule, dataset, seed);
    case 8:
      return std::make_unique<OptimizerImpl<8>>(gradModule, dataset, seed);
    case 16:
      return std::make_unique<OptimizerImpl<16>>(gradModule, dataset, seed);
    default: {
      std::ostringstream stream;
      stream << "unsupported batch size " << batchSize;
      throw Exception(stream.str());
    }
  }
}

Optimizer::~Optimizer() = default;

} // namespace axon
