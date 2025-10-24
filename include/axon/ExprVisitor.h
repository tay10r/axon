#pragma once

namespace axon {

class InputExpr;
class ParamExpr;
class ConstExpr;
class NegateExpr;
class RcpExpr;
class SqrtExpr;
class ExpExpr;
class SinExpr;
class CosExpr;
class ReLUExpr;
class SigmoidExpr;
class HeavisideExpr;
class AddExpr;
class SubExpr;
class MulExpr;
class GradAddExpr;

class ExprVisitor
{
public:
  virtual ~ExprVisitor();

  virtual void visit(const InputExpr&) = 0;

  virtual void visit(const ParamExpr&) = 0;

  virtual void visit(const ConstExpr&) = 0;

  virtual void visit(const NegateExpr&) = 0;

  virtual void visit(const RcpExpr&) = 0;

  virtual void visit(const SqrtExpr&) = 0;

  virtual void visit(const ExpExpr&) = 0;

  virtual void visit(const ReLUExpr&) = 0;

  virtual void visit(const SigmoidExpr&) = 0;

  virtual void visit(const HeavisideExpr&) = 0;

  virtual void visit(const SinExpr&) = 0;

  virtual void visit(const CosExpr&) = 0;

  virtual void visit(const AddExpr&) = 0;

  virtual void visit(const SubExpr&) = 0;

  virtual void visit(const MulExpr&) = 0;

  virtual void visit(const GradAddExpr&) = 0;
};

} // namespace axon
