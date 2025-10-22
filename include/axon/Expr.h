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

  [[nodiscard]] auto index() const -> uint32_t;

private:
  uint32_t m_index;
};

class ParamExpr final : public Expr
{
public:
  explicit ParamExpr(uint32_t index);

  void accept(ExprVisitor& visitor) const override;

  [[nodiscard]] auto index() const -> uint32_t;

private:
  uint32_t m_index;
};

class ConstExpr final : public Expr
{
public:
  ConstExpr();

  ConstExpr(float value);

  void accept(ExprVisitor& visitor) const override;

  [[nodiscard]] auto value() const -> float;

private:
  float m_value{ 0.0F };
};

class UnaryExpr : public Expr
{
public:
  explicit UnaryExpr(uint32_t operand);

  [[nodiscard]] auto operand() const -> uint32_t;

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

class HeavisideExpr final : public UnaryExpr
{
public:
  using UnaryExpr::UnaryExpr;

  void accept(ExprVisitor& visitor) const override;
};

class BinaryExpr : public Expr
{
public:
  BinaryExpr(uint32_t l, uint32_t r);

  [[nodiscard]] auto left() const -> uint32_t;

  [[nodiscard]] auto right() const -> uint32_t;

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
 * @brief This is a special function only intended for its side effect of adding to a gradient.
 *        It is considered an in-place operation on the gradient of a parameter.
 * */
class GradAddExpr final : public Expr
{
public:
  /**
   * @brief Constructs a gradiant add expression.
   *
   * @param paramIndex The index of the parameter that the gradient is for.
   *
   * @param valueIndex The index of the value to add to the gradient.
   * */
  explicit GradAddExpr(uint32_t paramIndex, uint32_t valueIndex);

  [[nodiscard]] auto paramIndex() const -> uint32_t;

  [[nodiscard]] auto valueIndex() const -> uint32_t;

  void accept(ExprVisitor& visitor) const override;

private:
  uint32_t m_paramIndex;

  uint32_t m_valueIndex;
};

} // namespace axon
