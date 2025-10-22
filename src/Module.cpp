#include <axon/Module.h>

#include <axon/Expr.h>
#include <axon/ExprVisitor.h>
#include <axon/ModuleBuilder.h>
#include <axon/Value.h>

#include <map>
#include <vector>

#include <assert.h>

namespace axon {

Module::~Module() = default;

namespace {

class ModuleBuilderImpl;
class GradModuleInserter;

class ModuleImpl final : public Module
{
  friend ModuleBuilderImpl;

  friend GradModuleInserter;

public:
  void visit(ExprVisitor& visitor) const override
  {
    for (size_t i = 0; i < m_exprs.size(); i++) {
      m_exprs[i]->accept(visitor);
    }
  }

  void reverseVisit(ExprVisitor& visitor) const override
  {
    for (size_t i = m_exprs.size(); i > 0; i--) {
      m_exprs[i - 1]->accept(visitor);
    }
  }

  void reverseVisitFrom(ExprVisitor& visitor, const uint32_t startOffset) const override
  {
    for (size_t i = startOffset + 1; i > 0; i--) {
      m_exprs[i - 1]->accept(visitor);
    }
  }

  [[nodiscard]] auto numParameters() const -> uint32_t override { return m_numParameters; }

  [[nodiscard]] auto numExprs() const -> uint32_t override { return static_cast<uint32_t>(m_exprs.size()); }

private:
  std::vector<std::shared_ptr<Expr>> m_exprs;

  uint32_t m_numParameters{};
};

class GradModuleInserter final : public ExprVisitor
{
public:
  explicit GradModuleInserter(ModuleImpl* m)
    : m_module(m)
  {
  }

  void registerGrad(const Expr* expr, Expr* gradExpr)
  {
    std::unique_ptr<Expr> tmp(gradExpr);

    auto it = m_gradMap.find(expr);
    if (it == m_gradMap.end()) {
      it = m_gradMap.emplace(expr, Value()).first;
    }

    m_module->m_exprs.emplace_back(std::move(tmp));

    it->second = Value(static_cast<uint32_t>(m_module->m_exprs.size() - 1));
  }

  void visit(const ConstExpr&) override {}

  void visit(const InputExpr&) override {}

  void visit(const ParamExpr& e) override
  {
    const auto grad = findGrad(&e);

    (void)push(new GradAddExpr(e.index(), grad.index()));
  }

  void visit(const NegateExpr& e) override
  {
    (void)e;
    //
  }

  void visit(const RcpExpr& e) override
  {
    const auto op = e.operand();
    (void)op;
    //
  }

  void visit(const SqrtExpr& e) override
  {
    const auto op = e.operand();
    (void)op;
    //
  }

  void visit(const ExpExpr& e) override
  {
    const auto op = e.operand();
    (void)op;
    //
  }

  void visit(const MaxExpr& e) override
  {
    const auto l = e.left();
    const auto r = e.right();
    (void)l;
    (void)r;

    //
  }

  void visit(const AddExpr& e) override
  {
    const auto grad = findGrad(&e);
    const auto l = e.left();
    const auto r = e.right();
    registerGrad(m_module->m_exprs.at(r).get(), grad);
    registerGrad(m_module->m_exprs.at(l).get(), grad);
  }

  void visit(const SubExpr& e) override
  {
    const auto grad = findGrad(&e);
    const auto l = e.left();
    const auto r = e.right();

    registerGrad(m_module->m_exprs.at(l).get(), grad);
    registerGrad(m_module->m_exprs.at(r).get(), new NegateExpr(grad.index()));
  }

  void visit(const MulExpr& e) override
  {
    const auto grad = findGrad(&e);
    const auto l = e.left();
    const auto r = e.right();
    registerGrad(m_module->m_exprs.at(l).get(), new MulExpr(grad.index(), r));
    registerGrad(m_module->m_exprs.at(r).get(), new MulExpr(grad.index(), l));
  }

  void visit(const GradAddExpr&) override { assert(false); /* technically should be unreachable */ }

protected:
  void registerGrad(const Expr* expr, Value value)
  {
    auto it = m_gradMap.find(expr);

    assert(it == m_gradMap.end());

    if (it == m_gradMap.end()) {
      it = m_gradMap.emplace(expr, Value()).first;
    }

    it->second = value;
  }

  [[nodiscard]] auto findGrad(const Expr* expr) -> Value { return m_gradMap.at(expr); }

  [[nodiscard]] auto push(Expr* e) -> Value
  {
    std::unique_ptr<Expr> tmp(e);

    m_module->m_exprs.emplace_back(std::move(tmp));

    return Value(static_cast<uint32_t>(m_module->m_exprs.size() - 1));
  }

private:
  ModuleImpl* m_module;

  std::map<const Expr*, Value> m_gradMap;
};

class ModuleBuilderImpl final : public ModuleBuilder
{
public:
  [[nodiscard]] auto build() -> std::unique_ptr<Module> override { return std::make_unique<ModuleImpl>(*m_module); }

  [[nodiscard]] auto buildWithGrad(const Value loss) -> std::unique_ptr<Module> override
  {
    auto m = std::make_unique<ModuleImpl>(*m_module);

    GradModuleInserter g(m.get());

    // Seed the auto grad algorithm
    g.registerGrad(m->m_exprs.at(loss.index()).get(), new ConstExpr(1.0F));

    // NOTE: Do not use the module copy, in case the fact that we are appending to it
    //       causes problems.
    m_module->reverseVisitFrom(g, loss.index());

    return m;
  }

  [[nodiscard]] auto input(const uint32_t index) -> Value override { return push(new InputExpr(index)); }

  [[nodiscard]] auto param() -> Value override
  {
    const uint32_t param = m_module->m_numParameters;

    m_module->m_numParameters++;

    return push(new ParamExpr(param));
  }

  [[nodiscard]] auto add(Value left, Value right) -> Value override
  {
    return push(new AddExpr(left.index(), right.index()));
  }

  [[nodiscard]] auto sub(Value left, Value right) -> Value override
  {
    return push(new SubExpr(left.index(), right.index()));
  }

  [[nodiscard]] auto mul(Value left, Value right) -> Value override
  {
    return push(new MulExpr(left.index(), right.index()));
  }

protected:
  template<typename DerivedExpr>
  [[nodiscard]] auto push(DerivedExpr* expr) -> Value
  {
    std::shared_ptr<DerivedExpr> ptr(expr);

    m_module->m_exprs.emplace_back(std::move(ptr));

    Value value(static_cast<uint32_t>(m_module->m_exprs.size() - 1));

    return value;
  }

private:
  std::unique_ptr<ModuleImpl> m_module{ new ModuleImpl() };
};

} // namespace

auto
ModuleBuilder::create() -> std::unique_ptr<ModuleBuilder>
{
  return std::make_unique<ModuleBuilderImpl>();
}

ModuleBuilder::~ModuleBuilder() = default;

} // namespace axon
