#include <axon/ExprPrinter.h>

#include <axon/Expr.h>

#include <ostream>

namespace axon {

#define out (*this->m_output) << "%" << m_counter++ << " <- "

ExprPrinter::ExprPrinter(std::ostream* output)
  : m_output(output)
{
}

void
ExprPrinter::visit(const InputExpr& e)
{
  out << "input [" << e.index() << "]" << '\n';
}

void
ExprPrinter::visit(const ParamExpr& e)
{
  out << "param [" << e.index() << "]" << '\n';
}

void
ExprPrinter::visit(const ConstExpr& e)
{
  out << "const " << e.value() << '\n';
}

void
ExprPrinter::visit(const NegateExpr& e)
{
  out << "negate %" << e.operand() << '\n';
}

void
ExprPrinter::visit(const RcpExpr& e)
{
  out << "rcp %" << e.operand() << '\n';
}

void
ExprPrinter::visit(const SqrtExpr& e)
{
  out << "sqrt %" << e.operand() << '\n';
}

void
ExprPrinter::visit(const ExpExpr& e)
{
  out << "exp %" << e.operand() << '\n';
}

void
ExprPrinter::visit(const ReLUExpr& e)
{
  out << "relu %" << e.operand() << '\n';
}

void
ExprPrinter::visit(const HeavisideExpr& e)
{
  out << "heaviside %" << e.operand() << '\n';
}

void
ExprPrinter::visit(const AddExpr& e)
{
  out << "add %" << e.left() << " %" << e.right() << '\n';
}

void
ExprPrinter::visit(const SubExpr& e)
{
  out << "sub %" << e.left() << " %" << e.right() << '\n';
}

void
ExprPrinter::visit(const MulExpr& e)
{
  out << "mul %" << e.left() << " %" << e.right() << '\n';
}

void
ExprPrinter::visit(const GradAddExpr& e)
{
  out << "grad_add [" << e.paramIndex() << "] %" << e.valueIndex() << '\n';
}

void
ExprPrinter::reset()
{
  m_counter = 0;
}

} // namespace axon
