#pragma once

#include <stdint.h>

namespace axon {

class ExprVisitor;

class Expr
{
public:
  virtual ~Expr();

  virtual void accept(ExprVisitor& visitor) const = 0;
};

class InputExpr final : public Expr
{
public:
  explicit InputExpr(uint32_t index);

  void accept(ExprVisitor& visitor) const override;

  [[nodiscard]] auto index() const -> uint32_t { return m_index; }

private:
  uint32_t m_index;
};

class ParamExpr final : public Expr
{
public:
  explicit ParamExpr(uint32_t index);

  void accept(ExprVisitor& visitor) const override;

  [[nodiscard]] auto index() const -> uint32_t { return m_index; }

private:
  uint32_t m_index;
};

class ConstExpr final : public Expr
{
public:
  ConstExpr();

  ConstExpr(float value);

  void accept(ExprVisitor& visitor) const override;

  [[nodiscard]] auto value() const -> float { return m_value; }

private:
  float m_value{ 0.0F };
};

class UnaryExpr : public Expr
{
public:
  explicit UnaryExpr(uint32_t operand);

  [[nodiscard]] auto operand() const -> uint32_t { return m_operand; }

private:
  uint32_t m_operand;
};

class NegateExpr final : public UnaryExpr
{
public:
  using UnaryExpr::UnaryExpr;

  void accept(ExprVisitor& visitor) const override;
};

class RcpExpr final : public UnaryExpr
{
public:
  using UnaryExpr::UnaryExpr;

  void accept(ExprVisitor& visitor) const override;
};

class SqrtExpr final : public UnaryExpr
{
public:
  using UnaryExpr::UnaryExpr;

  void accept(ExprVisitor& visitor) const override;
};

class ExpExpr final : public UnaryExpr
{
public:
  using UnaryExpr::UnaryExpr;

  void accept(ExprVisitor& visitor) const override;
};

class ReLUExpr final : public UnaryExpr
{
public:
  using UnaryExpr::UnaryExpr;

  void accept(ExprVisitor& visitor) const override;
};

class SigmoidExpr final : public UnaryExpr
{
public:
  using UnaryExpr::UnaryExpr;

  void accept(ExprVisitor& visitor) const override;
};

class HeavisideExpr final : public UnaryExpr
{
public:
  using UnaryExpr::UnaryExpr;

  void accept(ExprVisitor& visitor) const override;
};

class SinExpr final : public UnaryExpr
{
public:
  using UnaryExpr::UnaryExpr;

  void accept(ExprVisitor& visitor) const override;
};

class CosExpr final : public UnaryExpr
{
public:
  using UnaryExpr::UnaryExpr;

  void accept(ExprVisitor& visitor) const override;
};

class BinaryExpr : public Expr
{
public:
  BinaryExpr(uint32_t l, uint32_t r);

  [[nodiscard]] auto left() const -> uint32_t { return m_left; }

  [[nodiscard]] auto right() const -> uint32_t { return m_right; }

private:
  uint32_t m_left;

  uint32_t m_right;
};

class AddExpr final : public BinaryExpr
{
public:
  using BinaryExpr::BinaryExpr;

  void accept(ExprVisitor& visitor) const override;
};

class SubExpr final : public BinaryExpr
{
public:
  using BinaryExpr::BinaryExpr;

  void accept(ExprVisitor& visitor) const override;
};

class MulExpr final : public BinaryExpr
{
public:
  using BinaryExpr::BinaryExpr;

  void accept(ExprVisitor& visitor) const override;
};

/**
 * @brief This is a special function only intended for its side effect of setting output values.
 * */
class OutputExpr final : public Expr
{
public:
  /**
   * @brief Constructs an "output" expression.
   *
   * @param outputIndex The index of the output element to set.
   *
   * @param valueIndex The index of the value to assign the output element.
   * */
  explicit OutputExpr(uint32_t outputIndex, uint32_t valueIndex);

  [[nodiscard]] auto outputIndex() const -> uint32_t;

  [[nodiscard]] auto valueIndex() const -> uint32_t;

  void accept(ExprVisitor& visitor) const override;

private:
  uint32_t m_outputIndex;

  uint32_t m_valueIndex;
};

} // namespace axon
