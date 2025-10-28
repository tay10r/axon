#include <axon/compiler.hpp>

/* This is a very, very simple example of a neural network.
 * It is simply the equation:
 *   y = mx + b
 * However, the principle is the same when extending this to larger networks.
 * */
void
compile(axon::Compiler& compiler)
{
  /* First thing we do is define the computational graph for the neural net. */

  const auto w = axon::param("m");
  const auto b = axon::param("b");
  const auto x = axon::input();
  const auto y = w * x + b;

  /* Then we tell the compiler to take the operations it has recorded from the
   * operations above and produce and "eval" module. A "module" kind of like
   * a mathematical program, with instructions to execute. An "eval module" is
   * the module representation of the neural network.
   *
   * We pass a list of values that are considered outputs, which we may want to
   * use in the application that this network is for.
   */
  compiler.buildEvalModule({ y });

  /* Here we describe how to compute the "loss" for the network.
   * The loss is what indicates how loss the network got to the
   * target result. It is also used to figure out how to change
   * the parameters of the network to produce better results in
   * future calls.
   * */
  const auto target = axon::input();
  const auto loss = axon::mse(target, y);

  /* With the "loss" term defined, we can tell the compiler
   * to construct the "grad module". This module is for computing
   * the "gradient" of the neural network. The gradient tells us
   * how we have to change each parameter in order to get better
   * results. This process is called "training" or "optimization".
   * The loss term is essential for computing the gradient.
   * */
  compiler.buildGradModule(loss);
}
