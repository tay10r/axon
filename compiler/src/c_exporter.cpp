#include "c_exporter.hpp"

#include <axon/expr.hpp>
#include <axon/expr_visitor.hpp>
#include <axon/module.hpp>

#include <fstream>
#include <sstream>
#include <string_view>

#include <assert.h>
#include <stddef.h>

namespace axon {

namespace {

/* This class is for emitting C code that represents the expressions in a module.
 * */
class CExprWriter final : public ExprVisitor
{
public:
  [[nodiscard]] auto source() const -> std::string { return m_source.str(); }

  void visit(const InputExpr& e) override
  {
    std::ostringstream tmp;
    tmp << "input[" << e.index() << "]";
    addExpr(tmp.str());
  }

  void visit(const ParamExpr& e) override
  {
    std::ostringstream tmp;
    tmp << "parameters[" << e.index() << "]";
    addExpr(tmp.str());
  }

  void visit(const ConstExpr& e) override
  {
    std::ostringstream tmp;
    tmp << e.value();
    addExpr(tmp.str());
  }

  void visit(const NegateExpr& e) override
  {
    std::ostringstream tmp;
    tmp << '-' << tmpName(e.operand());
    addExpr(tmp.str());
  }

  void visit(const RcpExpr& e) override
  {
    std::ostringstream tmp;
    tmp << "1.0F / " << tmpName(e.operand());
    addExpr(tmp.str());
  }

  void visit(const SqrtExpr& e) override
  {
    std::ostringstream tmp;
    tmp << "sqrtf(" << tmpName(e.operand()) << ")";
    addExpr(tmp.str());
  }

  void visit(const ExpExpr& e) override
  {
    std::ostringstream tmp;
    tmp << "expf(" << tmpName(e.operand()) << ")";
    addExpr(tmp.str());
  }

  void visit(const ReLUExpr& e) override
  {
    std::ostringstream tmp;
    tmp << "fmaxf(" << tmpName(e.operand()) << ", 0.0F)";
    addExpr(tmp.str());
  }

  void visit(const SigmoidExpr& e) override
  {
    std::ostringstream tmp;
    tmp << "1.0F / (1.0F + expf(-" << tmpName(e.operand()) << "))";
    addExpr(tmp.str());
  }

  void visit(const HeavisideExpr& e) override
  {
    std::ostringstream tmp;
    tmp << tmpName(e.operand()) << " > 0.0F ? 1.0F : 0.0F";
    addExpr(tmp.str());
  }

  void visit(const SinExpr& e) override
  {
    std::ostringstream tmp;
    tmp << "sinf(" << tmpName(e.operand()) << ")";
    addExpr(tmp.str());
  }

  void visit(const CosExpr& e) override
  {
    std::ostringstream tmp;
    tmp << "cosf(" << tmpName(e.operand()) << ")";
    addExpr(tmp.str());
  }

  void visit(const AddExpr& e) override
  {
    std::ostringstream tmp;
    tmp << tmpName(e.left()) << " + " << tmpName(e.right());
    addExpr(tmp.str());
  }

  void visit(const SubExpr& e) override
  {
    std::ostringstream tmp;
    tmp << tmpName(e.left()) << " - " << tmpName(e.right());
    addExpr(tmp.str());
  }

  void visit(const MulExpr& e) override
  {
    std::ostringstream tmp;
    tmp << tmpName(e.left()) << " * " << tmpName(e.right());
    addExpr(tmp.str());
  }

  void visit(const OutputExpr& e) override
  {
    std::ostringstream tmp;
    tmp << "output[" << e.outputIndex() << "] = " << tmpName(e.valueIndex()) << ';';
    line(tmp.str());
    m_counter++;
  }

protected:
  [[nodiscard]] auto tmpName(const uint32_t value) -> std::string
  {
    std::ostringstream stream;
    stream << "v" << value;
    return stream.str();
  }

  void addExpr(const std::string_view& s)
  {
    std::ostringstream tmp;
    tmp << "const float v" << m_counter << " = " << s << ";";
    line(tmp.str());
    m_counter++;
  }

  void line(const std::string_view& s)
  {
    m_source << "  ";
    m_source << s;
    m_source << '\n';
  }

  void line() { m_source << '\n'; }

private:
  std::ostringstream m_source;

  size_t m_counter{};
};

const char macrosSrc[] = R"(/* performance macros */

#ifndef AXON_RESTRICT
#  ifdef __cplusplus
#    if defined(__clang__) || defined(__GNUC__)
#      define AXON_RESTRICT __restrict__
#    elif defined(_MSC_VER)
#      define AXON_RESTRICT __restrict
#    else
#      define AXON_RESTRICT
#    endif
#  else
#    if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L)
#      define AXON_RESTRICT restrict
#    elif defined(_MSC_VER)
#      define AXON_RESTRICT __restrict
#    elif defined(__clang__) || defined(__GNUC__)
#      define AXON_RESTRICT __restrict__
#    else
#      define AXON_RESTRICT
#    endif
#  endif
#endif
)";

const char rngSrc[] = R"( /* RNG */
/**
 * @brief This is an LCG PRNG
 *
 * @details This acts as a poor man's PRNG if nothing else exists for the user.
 *          If using C++, prefer to use the <random> module instead, or another
 *          mature PRNG library.
 * */
struct axon_rng
{
  uint32_t state;
};

typedef struct axon_rng axon_rng_z;

inline static void
axon_rng_init(axon_rng_z* self, uint32_t seed)
{
  self->state = (seed == 0) ? 1 : seed;
}

inline static uint32_t
axon_rng(axon_rng_z* self)
{
  const uint32_t A = 1664525u;
  const uint32_t C = 1013904223u;
  self->state = A * self->state + C;
  return self->state;
}

inline static uint32_t
axon_rng_range(axon_rng_z* self, const uint32_t a, const uint32_t b)
{
  const uint32_t range = b - a + 1;
  if (range == 0) {
    return axon_rng(self);
  }

  const uint32_t limit = UINT32_MAX - (UINT32_MAX % range);
  uint32_t r;
  do {
    r = axon_rng(self);
  } while (r >= limit);

  return a + (r % range);
}

inline static float
axon_rng_float(axon_rng_z* self)
{
  return axon_rng(self) * (1.0F / 4294967296.0F);
}

inline static void
axon_rng_float_array(axon_rng_z* self, float* AXON_RESTRICT data, const size_t len, const float scale, const float bias)
{
  for (size_t i = 0; i < len; i++) {
    data[i] = axon_rng_float(self) * scale + bias;
  }
}
)";

const char optimizerSrc[] = R"(/* Optimizer */

#ifndef AXON_BUFFER_ALIGN
#define AXON_BUFFER_ALIGN 128 /* This is basically to align to a cache line and avoid cache thrashing. */
#endif

#define AXON_BUFFER_SIZE ((AXON_PARAMETERS * sizeof(float) + (AXON_BUFFER_ALIGN - 1)) / AXON_BUFFER_ALIGN) * AXON_BUFFER_ALIGN

struct axon_opt
{
  float gradient[AXON_BUFFER_SIZE / sizeof(float)];

  float momentum[2][AXON_BUFFER_SIZE / sizeof(float)];

  size_t step;
};

typedef struct axon_opt axon_opt_z;

inline static void
axon_opt_init(axon_opt_z* self, const uint32_t seed)
{
  self->step = 0;
  for (size_t i = 0; i < (AXON_BUFFER_SIZE / sizeof(float)); i++) {
    self->gradient[i] = 0.0F;
    self->momentum[0][i] = 0.0F;
    self->momentum[1][i] = 0.0F;
  }
}

inline static void
axon_opt_step(axon_opt_z* self, const float lr, const float momentum, float* AXON_RESTRICT parameters)
{
  const float* AXON_RESTRICT g = self->gradient;
  const float* AXON_RESTRICT m0 = self->momentum[self->step & 1];
  float* AXON_RESTRICT m1 = self->momentum[(self->step + 1) & 1];

  for (size_t i = 0; i < AXON_PARAMETERS; i++) {
    const float m = m0[i] * momentum + g[i] * (1.0F - momentum);
    m1[i] = m;
    parameters[i] -= m * lr;
  }

  self->step++;
}
)";

class ParamNameWriter final : public ExprVisitor
{
public:
  explicit ParamNameWriter(std::ostream* output)
    : m_output(output)
  {
  }

  void visit(const InputExpr&) override {}

  void visit(const ParamExpr& e) override
  {
    const auto name = e.name();

    if (!name.empty()) {
      (*m_output) << "#define AXON_PARAMETER_" << name << ' ' << e.index() << std::endl;
      m_numNames++;
    }
  }

  void visit(const ConstExpr&) override {}

  void visit(const NegateExpr&) override {}

  void visit(const RcpExpr&) override {}

  void visit(const SqrtExpr&) override {}

  void visit(const ExpExpr&) override {}

  void visit(const ReLUExpr&) override {}

  void visit(const SigmoidExpr&) override {}

  void visit(const HeavisideExpr&) override {}

  void visit(const SinExpr&) override {}

  void visit(const CosExpr&) override {}

  void visit(const AddExpr&) override {}

  void visit(const SubExpr&) override {}

  void visit(const MulExpr&) override {}

  void visit(const OutputExpr&) override {}

  [[nodiscard]] auto numNames() const -> size_t { return m_numNames; }

private:
  std::ostream* m_output;

  size_t m_numNames{};
};

class CExporter final : public Exporter
{
public:
  void exportFull(const Module& evalModule, const Module& gradModule, const std::filesystem::path& outputPath) override
  {
    std::ofstream f(outputPath);
    f << "#pragma once" << std::endl;
    f << std::endl;
    f << "/* Note: This file is automatically generated. Edits may be lost. */" << std::endl;
    f << std::endl;
    f << "#include <stddef.h>" << std::endl;
    f << "#include <stdint.h>" << std::endl;
    f << "#include <math.h>" << std::endl;
    f << "#include <limits.h>" << std::endl;
    f << std::endl;
    f << macrosSrc;
    f << std::endl;
    f << rngSrc;
    f << std::endl;
    f << "#define AXON_PARAMETERS " << evalModule.numParameters() << std::endl;
    f << std::endl;
    f << "#define AXON_EVAL_INPUTS " << evalModule.numInputs() << std::endl;
    f << "#define AXON_EVAL_OUTPUTS " << evalModule.numOutputs() << std::endl;
    f << std::endl;
    f << "#define AXON_GRAD_INPUTS " << gradModule.numInputs() << std::endl;
    f << "#define AXON_GRAD_OUTPUTS " << gradModule.numOutputs() << std::endl;
    f << std::endl;
    ParamNameWriter paramNameWriter(&f);
    evalModule.visit(paramNameWriter);
    if (paramNameWriter.numNames() > 0) {
      f << std::endl;
    }
    f << "inline static void" << std::endl;
    f << "axon_eval(const float* AXON_RESTRICT parameters, const float* AXON_RESTRICT input, float* AXON_RESTRICT "
         "output)"
      << std::endl;
    f << "{" << std::endl;
    {
      CExprWriter writer;
      evalModule.visit(writer);
      f << writer.source();
    }
    f << '}' << std::endl;
    f << std::endl;
    f << "inline static void" << std::endl;
    f << "axon_grad(const float* AXON_RESTRICT parameters, const float* AXON_RESTRICT input, float* AXON_RESTRICT "
         "output)"
      << std::endl;
    f << "{" << std::endl;
    {
      CExprWriter writer;
      gradModule.visit(writer);
      f << writer.source();
    }
    f << '}' << std::endl;
    f << std::endl;
    f << optimizerSrc;
  }

  void exportLean(const Module& evalModule,
                  const std::vector<float>& parameters,
                  const std::filesystem::path& outputDir) override
  {
    //
  }
};

} // namespace

void
registerCExporter()
{
  Exporter::addToRegistry("c", std::make_shared<CExporter>());
}

} // namespace axon
