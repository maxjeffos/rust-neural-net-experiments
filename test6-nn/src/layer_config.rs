use crate::activation::ActivationFunction;

#[derive(Debug, Clone)]
pub struct LayerConfig {
    // Optional because the input layer doesn't have an activation function
    pub activation_function: Option<ActivationFunction>,
    // weights: Matrix,
    // biases: ColumnVector,
    pub initializer: Option<String>,
}

impl LayerConfig {
    pub fn new(activation_function: Option<ActivationFunction>) -> Self {
        Self {
            activation_function,
            initializer: None,
        }
    }

    pub fn new_with_initializer(
        activation_function: Option<ActivationFunction>,
        initializer: Option<String>,
    ) -> Self {
        Self {
            activation_function,
            initializer,
        }
    }
}
