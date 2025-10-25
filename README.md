About
=====

This is a C++ library for designing and training neural networks, with a focus on smaller networks (fewer than 10k parameters).
The reason for building this library was to make it easier to deploy and train neural networks in resource constrained environments,
such as single GPU kernels or embedded systems.

### Example

```cpp
int main() {

    // This is used to build the neural network.
    // Note that a global pointer is also associated
    // with this object and can be explicity set (or unset)
    // with axon::ModuleBuilder::setCurrent()
    auto builder = axon::ModuleBuilder::create();

    // a 3x3 trainable parameter
    const auto w = axon::param<3, 3>();

    // a 3x1 input vector
    const auto x = axon::input<3, 1>();

    // a 3x1 trainable parameter
    const auto b = axon::input<3, 1>();

    const auto y = w * x + b;

    // This represents the computational graph above, and can be used for inference.
    const auto evalModule = builder->build();

    // Target values are considered inputs
    const auto target = axon::input<3, 1>();

    // Compute the MSE loss between the target value and predicted value
    const auto loss = axon::mse(y, target);

    // This module is used for computing the gradients with respect to the
    // given loss value.
    const auto gradModule = builder->buildWithGrad(loss);

    return 0;
}
```

More examples can be found in the `examples/` directory.