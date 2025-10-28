About
=====

This is a C++ library for designing and training neural networks, with a focus on smaller networks (fewer than 10k parameters).
The reason for building this library was to make it easier to deploy and train neural networks in resource constrained environments,
such as single GPU kernels or embedded systems.

### Example

```cpp
void
compile(axon::Compiler& compiler)

    // a 3x3 trainable parameter
    const auto w = axon::param<3, 3>(/*name=*/"w");

    // a 3x1 input vector
    const auto x = axon::input<3, 1>();

    // a 3x1 trainable parameter
    const auto b = axon::input<3, 1>(/*name=*/"b");

    const auto y = w * x + b;

    // This represents the computational graph above, and can be used for inference.
    compiler.buildEvalModule({ y });

    // Target values are considered inputs
    const auto target = axon::input<3, 1>();

    // Compute the MSE loss between the target value and predicted value
    const auto loss = axon::mse(y, target);

    // This module is used for computing the gradients with respect to the
    // given loss value.
    compiler.buildGradModule(loss);
}
```

More examples can be found in the `examples/` directory.
