# Rust Neural Network Implementation

A neural network implementation written from scratch in Rust, developed as a learning project to simultaneously explore deep learning concepts and Rust programming.

## Project Overview

This project was primarily developed in 2021 and 2022 as an educational exercise with two main goals:
1. Learn neural network fundamentals by implementing everything from scratch
2. Gain more practical experience with Rust

The implementation includes:
- Multi-layer neural network architecture
- Custom linear algebra implementations (later integrated with BLAS for performance)
- Various activation functions and optimizers
- Different weight initialization schemes
- MNIST dataset training capability
- Framework for building simple neural network architectures

## Technical Details

### Core Features
- **Linear Algebra**: Initially implemented matrix operations from scratch to explore the mathematical foundations. Later optimized using BLAS.
- **Network Architecture**: Supports arbitrary layer configurations with customizable activation functions
- **Training**: Implements backpropagation and gradient descent
- **Activation Functions**: Includes ReLU, Leaky ReLU, ELU, and Sigmoid (as well as my own custom one called JeLU - get it?)
- **Optimizers**: A variety of optimizers including Adam and momentum
- **Weight Initialization**: Supports various schemes including Xavier initialization
- **Gradient Checking**: An algorithm for validating that gradient descent is working correctly
- **Cost Functions**: Cost functions including quadratic cost and cross-entropy loss
- **Training Logging**: Training logging to track training data as training progresses

### Example Usage

```rust
let mut nn = NeuralNetworkBuilder::new()
        .with_input_layer(784)
        .with_hidden_layer(
            100,
            Initializer::HeForReLUAndVariants,
            ActivationFunction::LeakyReLU(0.1),
        )
        .with_hidden_layer(
            100,
            Initializer::HeForReLUAndVariants,
            ActivationFunction::LeakyReLU(0.1),
        )
        .with_hidden_layer(
            100,
            Initializer::HeForReLUAndVariants,
            ActivationFunction::LeakyReLU(0.1),
        )
        .with_hidden_layer(
            50,
            Initializer::HeForReLUAndVariants,
            ActivationFunction::LeakyReLU(0.1),
        )
        .with_output_layer(
            10,
            Initializer::XavierNormalHOMLForSigmoid,
            ActivationFunction::Softmax,
        )
        .with_cost_fn(CostFunc::CrossEntropy)
        .build();

    nn.train_stochastic(
        &training_data,
        10_000,
        &Optimizer::Adam(AdamConfig::default()),
        5000,
        Some(&check_options),
        Some(early_stop_config),
        Some(10),
        Some(session_logger),
    );
```

## Project Status

This is an experimental learning project and remains in a work-in-progress state. While functional for basic tasks like MNIST classification, it was primarily developed for personal learning and experimentation rather than production use.

The code contains a bunch of experiments in the `test/d.*` directories (crates), but the notable ones are:
- test7-nn-mnist-classifier
- test6-nn (which has an earlier iteration of the same)

The other `test/d.*` directories contain other non-neural-network experiments (traditional ML).

All the Linear Algebra is contained in the `common` directory (crate).

Run it by doing this:

```sh
cd test7-nn-mnist-classifier
cargo run
```

## Shout Outs

Shout out to: 
- [Michael Nielsen's online book about neural networks](http://neuralnetworksanddeeplearning.com/)
- [3Blue1Brown](https://www.youtube.com/@3blue1brown)

Both of which are phenominal resources for learning the fundamental math of how Neural Networks work.

## Future Possibilities

While the project's main educational goals have been met, potential areas for expansion include:
- Support accelerated computing using CUDA or Metal
- Re-organize the repo to focus it on the Neural Networks only (not the other traditional ML experiments) and export the Neural Network stuff so it can be used as a library (crate) from outside code
- An explanation of the math, if I can find a nice and somewhat painless way of rendering it online in a GitHub repo
- Support assitional model types
- Cleanup for improved dev experience

## License

WTFPL but if you get rich with this, feel free to throw me some cash (or equity is nice, too).

---

Note: This project is primarily educational in nature. It is not intended for production use.
