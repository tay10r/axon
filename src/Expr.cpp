#include <axon/Expr.h>

#include <axon/ExprVisitor.h>

namespace axon {

Expr::~Expr() = default;

InputExpr::InputExpr(const uint32_t index)
  : m_index(index)
{
}

void
InputExpr::accept(ExprVisitor& visitor) const
{
  visitor.visit(*this);
}

auto
InputExpr::index() const -> uint32_t
{
  return m_index;
}

ParamExpr::ParamExpr(const uint32_t index)
  : m_index(index)
{
}

void
ParamExpr::accept(ExprVisitor& visitor) const
{
  visitor.visit(*this);
}

auto
ParamExpr::index() const -> uint32_t
{
  return m_index;
}

ConstExpr::ConstExpr() = default;

ConstExpr::ConstExpr(const float value)
  : m_value(value)
{
}

void
ConstExpr::accept(ExprVisitor& visitor) const
{
  visitor.visit(*this);
}

auto
ConstExpr::value() const -> float
{
  return m_value;
}

UnaryExpr::UnaryExpr(const uint32_t operand)
  : m_operand(operand)
{
}

auto
UnaryExpr::operand() const -> uint32_t
{
  return m_operand;
}

void
NegateExpr::accept(ExprVisitor& visitor) const
{
  visitor.visit(*this);
}

void
RcpExpr::accept(ExprVisitor& visitor) const
{
  visitor.visit(*this);
}

void
SqrtExpr::accept(ExprVisitor& visitor) const
{
  visitor.visit(*this);
}

void
ExpExpr::accept(ExprVisitor& visitor) const
{
  visitor.visit(*this);
}

void
ReLUExpr::accept(ExprVisitor& visitor) const
{
  visitor.visit(*this);
}

void
SigmoidExpr::accept(ExprVisitor& visitor) const
{
  visitor.visit(*this);
}

void
HeavisideExpr::accept(ExprVisitor& visitor) const
{
  visitor.visit(*this);
}

void
SinExpr::accept(ExprVisitor& visitor) const
{
  visitor.visit(*this);
}

void
CosExpr::accept(ExprVisitor& visitor) const
{
  visitor.visit(*this);
}

BinaryExpr::BinaryExpr(const uint32_t l, const uint32_t r)
  : m_left(l)
  , m_right(r)
{
}

auto
BinaryExpr::left() const -> uint32_t
{
  return m_left;
}

auto
BinaryExpr::right() const -> uint32_t
{
  return m_right;
}

void
AddExpr::accept(ExprVisitor& visitor) const
{
  visitor.visit(*this);
}

void
SubExpr::accept(ExprVisitor& visitor) const
{
  visitor.visit(*this);
}

void
MulExpr::accept(ExprVisitor& visitor) const
{
  visitor.visit(*this);
}

GradAddExpr::GradAddExpr(const uint32_t paramIndex, const uint32_t valueIndex)
  : m_paramIndex(paramIndex)
  , m_valueIndex(valueIndex)
{
}

auto
GradAddExpr::paramIndex() const -> uint32_t
{
  return m_paramIndex;
}

auto
GradAddExpr::valueIndex() const -> uint32_t
{
  return m_valueIndex;
}

void
GradAddExpr::accept(ExprVisitor& visitor) const
{
  visitor.visit(*this);
}

} // namespace axon
