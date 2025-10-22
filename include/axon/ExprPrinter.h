#pragma once

#include <axon/ExprVisitor.h>

#include <iosfwd>

namespace axon {

class ExprPrinter final : public ExprVisitor
{
public:
  ExprPrinter(std::ostream* output);

  void visit(const InputExpr&) override;

  void visit(const ParamExpr&) override;

  void visit(const ConstExpr&) override;

  void visit(const NegateExpr&) override;

  void visit(const RcpExpr&) override;

  void visit(const SqrtExpr&) override;

  void visit(const ExpExpr&) override;

  void visit(const MaxExpr&) override;

  void visit(const AddExpr&) override;

  void visit(const SubExpr&) override;

  void visit(const MulExpr&) override;

  void visit(const GradAddExpr&) override;

  void reset();

private:
  std::ostream* m_output;

  size_t m_counter{};
};

} // namespace axon
