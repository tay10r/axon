#pragma once

#include <stdint.h>

namespace axon {

class ExprVisitor;

class Module
{
public:
  virtual ~Module();

  virtual void visit(ExprVisitor& visitor) const = 0;

  virtual void reverseVisit(ExprVisitor& visitor) const = 0;

  virtual void reverseVisitFrom(ExprVisitor& visitor, uint32_t startOffset) const = 0;

  [[nodiscard]] virtual auto numParameters() const -> uint32_t = 0;

  [[nodiscard]] virtual auto numExprs() const -> uint32_t = 0;
};

} // namespace axon
