use std::collections::HashMap;

use crate::activation::ActivationFunction;
use crate::initializer::get_init_weights_and_biases;
use crate::layer_info::LayerInfo;
use crate::{Initializer, SimpleNeuralNetwork};
use common::column_vec_of_random_values_from_distribution;
use common::linalg::{ColumnVector, Matrix};

pub struct NeuralNetworkBuilder {
    input_layer_size: Option<usize>,
    hidden_layers_info: Vec<HiddenLayerConfig>,
    output_layer_info: Option<OutputLayerConfig>,
}

#[derive(Debug, Clone)]
pub struct HiddenLayerConfig {
    size: usize,
    weights_and_biases: Initializer, // rename this field to initializer
    activation_function: ActivationFunction,
}

#[derive(Debug, Clone)]
pub struct OutputLayerConfig {
    size: usize,
    weights_and_biases: Initializer,
    activation_function: ActivationFunction,
}

impl NeuralNetworkBuilder {
    pub fn new() -> Self {
        Self {
            input_layer_size: None,
            hidden_layers_info: Vec::new(),
            output_layer_info: None,
        }
    }

    pub fn with_input_layer(mut self, size: usize) -> Self {
        self.input_layer_size = Some(size);
        self
    }

    pub fn with_hidden_layer(
        mut self,
        size: usize,
        weights_and_biases: Initializer,
        activation_function: ActivationFunction,
    ) -> Self {
        if self.input_layer_size.is_none() {
            panic!("Input layer size must be set before adding hidden layers");
        }

        if let Initializer::Manual(weights, bias_v) = &weights_and_biases {
            if weights.num_rows() != size {
                panic!(
                  "The number of rows in the weights matrix ({}) does not match the size of the layer ({})",
                  weights.num_rows(),
                  size
              );
            }
            let previous_row_size = if self.hidden_layers_info.is_empty() {
                self.input_layer_size.unwrap()
            } else {
                self.hidden_layers_info.last().unwrap().size
            };

            println!("previous row size: {}", previous_row_size);

            if weights.num_columns() != previous_row_size {
                panic!(
                  "The number of columns in the weights matrix ({}) does not match the size of the previous layer ({})",
                  weights.num_columns(),
                  previous_row_size
              );
            }

            if bias_v.num_elements() != size {
                panic!(
                  "The number of elements in the bias vector ({}) does not match the size of the layer ({})",
                  bias_v.num_elements(),
                  size
              );
            }
        }

        self.hidden_layers_info.push(HiddenLayerConfig {
            size,
            weights_and_biases,
            activation_function,
        });
        self
    }

    pub fn with_output_layer(
        mut self,
        size: usize,
        weights_and_biases: Initializer,
        activation_function: ActivationFunction,
    ) -> Self {
        if self.input_layer_size.is_none() {
            panic!("Input layer size must be set before adding the output layer");
        }

        if let Initializer::Manual(weights, bias_v) = &weights_and_biases {
            if weights.num_rows() != size {
                panic!(
                  "The number of rows in the weights matrix ({}) does not match the size of the layer ({})",
                  weights.num_rows(),
                  size
              );
            }
            let previous_row_size = if self.hidden_layers_info.is_empty() {
                self.input_layer_size.unwrap()
            } else {
                self.hidden_layers_info.last().unwrap().size
            };

            println!("previous row size: {}", previous_row_size);

            if weights.num_columns() != previous_row_size {
                panic!(
                  "The number of columns in the weights matrix ({}) does not match the size of the previous layer ({})",
                  weights.num_columns(),
                  previous_row_size
              );
            }

            if bias_v.num_elements() != size {
                panic!(
                  "The number of elements in the bias vector ({}) does not match the size of the layer ({})",
                  bias_v.num_elements(),
                  size
              );
            }
        }

        self.output_layer_info = Some(OutputLayerConfig {
            size,
            weights_and_biases,
            activation_function,
        });
        self
    }

    pub fn build(self) -> SimpleNeuralNetwork {
        // first setup the sizes
        let mut sizes = Vec::new();
        let mut layer_infos = HashMap::new();
        layer_infos.insert(0, LayerInfo::new_with_initializer(None, None));

        if let Some(input_layer_size) = self.input_layer_size {
            sizes.push(self.input_layer_size.unwrap());
        } else {
            panic!("Input layer size not specified");
        }

        self.hidden_layers_info.iter().for_each(|layer_info| {
            sizes.push(layer_info.size);
        });

        if let Some(output_layer_info) = &self.output_layer_info {
            sizes.push(output_layer_info.size);
        } else {
            panic!("Output layer size not specified");
        }

        // initial weights and biases
        let mut weights = HashMap::new();
        let mut biases = HashMap::new();

        let mut l = 1; // input layer is l 0 and doesn't have weights/biases

        for h in self.hidden_layers_info {
            let initializer_str = format!("{}", &h.weights_and_biases);

            match h.weights_and_biases {
                Initializer::RandomBasic => {
                    let weights_m = Matrix::new_matrix_with_random_values_from_normal_distribution(
                        sizes[l],
                        sizes[l - 1],
                        0.0,
                        1.0,
                    );

                    let bias_v = column_vec_of_random_values_from_distribution(0.0, 1.0, sizes[l]);

                    weights.insert(l, weights_m);
                    biases.insert(l, bias_v);
                }
                Initializer::Manual(weights_m, bias_v) => {
                    weights.insert(l, weights_m);
                    biases.insert(l, bias_v);
                }
                Initializer::Xavier => {
                    let num_nodes_in_previous_layer = sizes[l - 1];
                    let min = -1.0 / (num_nodes_in_previous_layer as f64).sqrt();
                    let max = 1.0 / (num_nodes_in_previous_layer as f64).sqrt();

                    let weights_m = Matrix::new_matrix_with_random_values_from_uniform_distribution(
                        sizes[l],
                        sizes[l - 1],
                        min,
                        max,
                    );

                    let bias_v = ColumnVector::new_zero_vector(sizes[l]);

                    weights.insert(l, weights_m);
                    biases.insert(l, bias_v);
                }
                Initializer::XavierNormalized => {
                    let num_nodes_in_this_layer = sizes[l];
                    let num_nodes_in_next_layer = sizes[l + 1];

                    println!(
                        "XavierNormalized: num_nodes_in_this_layer: {}",
                        num_nodes_in_this_layer
                    );
                    println!(
                        "XavierNormalized: num_nodes_in_next_layer: {}",
                        num_nodes_in_next_layer
                    );

                    let x = 6.0_f64.sqrt()
                        / (num_nodes_in_this_layer as f64 + num_nodes_in_next_layer as f64).sqrt();
                    let min = -x;
                    let max = x;

                    println!("using XavierNormalized with +/- {}", x);

                    let weights_m = Matrix::new_matrix_with_random_values_from_uniform_distribution(
                        sizes[l],
                        sizes[l - 1],
                        min,
                        max,
                    );

                    let bias_v = ColumnVector::new_zero_vector(sizes[l]);

                    weights.insert(l, weights_m);
                    biases.insert(l, bias_v);
                }
                Initializer::XavierNormalHOMLForSigmoid => {
                    let (weights_m, bias_v) = get_init_weights_and_biases(
                        l,
                        &sizes,
                        Initializer::XavierNormalHOMLForSigmoid,
                    );
                    weights.insert(l, weights_m);
                    biases.insert(l, bias_v);
                }
                Initializer::HeForReLUAndVariants => {
                    let (weights_m, bias_v) =
                        get_init_weights_and_biases(l, &sizes, Initializer::HeForReLUAndVariants);
                    weights.insert(l, weights_m);
                    biases.insert(l, bias_v);
                }
            }

            layer_infos.insert(
                l,
                LayerInfo::new_with_initializer(Some(h.activation_function), Some(initializer_str)),
            );
            l += 1;
        }

        // l is now the output layer
        let output_layer_info = self.output_layer_info.unwrap();
        let initializer_str = format!("{}", &output_layer_info.weights_and_biases);
        match output_layer_info.weights_and_biases {
            Initializer::RandomBasic => {
                let weights_m = Matrix::new_matrix_with_random_values_from_normal_distribution(
                    sizes[l],
                    sizes[l - 1],
                    0.0,
                    1.0,
                );

                let bias_v = column_vec_of_random_values_from_distribution(0.0, 1.0, sizes[l]);

                weights.insert(l, weights_m);
                biases.insert(l, bias_v);
            }
            Initializer::Manual(weights_m, bias_v) => {
                weights.insert(l, weights_m);
                biases.insert(l, bias_v);
            }
            Initializer::Xavier => {
                let num_nodes_in_previous_layer = sizes[l - 1];
                let min = -1.0 / (num_nodes_in_previous_layer as f64).sqrt();
                let max = 1.0 / (num_nodes_in_previous_layer as f64).sqrt();

                let weights_m = Matrix::new_matrix_with_random_values_from_uniform_distribution(
                    sizes[l],
                    sizes[l - 1],
                    min,
                    max,
                );

                let bias_v = ColumnVector::new_zero_vector(sizes[l]);

                weights.insert(l, weights_m);
                biases.insert(l, bias_v);
            }
            Initializer::XavierNormalized => {
                // WARNING - I don't know if this is right for the output layer
                let num_nodes_in_this_layer = sizes[l];

                let x = 6.0_f64.sqrt() / (num_nodes_in_this_layer as f64).sqrt();
                let min = -x;
                let max = x;

                println!("using XavierNormalized with +/- {}", x);

                let weights_m = Matrix::new_matrix_with_random_values_from_uniform_distribution(
                    sizes[l],
                    sizes[l - 1],
                    min,
                    max,
                );

                let bias_v = ColumnVector::new_zero_vector(sizes[l]);

                weights.insert(l, weights_m);
                biases.insert(l, bias_v);
            }
            Initializer::XavierNormalHOMLForSigmoid => {
                let (weights_m, bias_v) =
                    get_init_weights_and_biases(l, &sizes, Initializer::XavierNormalHOMLForSigmoid);
                weights.insert(l, weights_m);
                biases.insert(l, bias_v);
            }
            Initializer::HeForReLUAndVariants => {
                let (weights_m, bias_v) =
                    get_init_weights_and_biases(l, &sizes, Initializer::HeForReLUAndVariants);
                weights.insert(l, weights_m);
                biases.insert(l, bias_v);
            }
        }
        layer_infos.insert(
            l,
            LayerInfo::new_with_initializer(
                Some(output_layer_info.activation_function),
                Some(initializer_str),
            ),
        );

        SimpleNeuralNetwork {
            sizes,
            weights,
            biases,
            layer_infos,
        }
    }
}

#[cfg(test)]
mod test_nn_builder {
    use super::*;
    use common::column_vector;
    use common::linalg::{MatrixShape, RowsMatrixBuilder};
    use std::panic;

    #[test]
    fn test_nn_builder_manual_wb_values() {
        // 2 x 3 x 1
        // W0: 3x2 (l1size x l0size)
        // W1: 1x3 (l2size x l1size)

        let weights_l1 = RowsMatrixBuilder::new()
            .with_row(&[0.2, 0.2])
            .with_row(&[0.4, 0.4])
            .with_row(&[0.6, 0.6])
            .build();

        let weights_l2 = RowsMatrixBuilder::new().with_row(&[0.5, 0.5, 0.5]).build();

        let bias_v_l1 = column_vector![0.1, 0.1, 0.1];
        let bias_v_l2 = column_vector![0.1];

        let nn = NeuralNetworkBuilder::new()
            .with_input_layer(2)
            .with_hidden_layer(
                3,
                Initializer::Manual(weights_l1, bias_v_l1),
                ActivationFunction::Sigmoid,
            )
            .with_output_layer(
                1,
                Initializer::Manual(weights_l2, bias_v_l2),
                ActivationFunction::Sigmoid,
            )
            .build();

        assert_eq!(nn.num_layers(), 3);

        let l1w = nn.weights.get(&1).unwrap();
        assert_eq!(l1w.shape(), MatrixShape::new(3, 2));
        assert_eq!(*l1w.data, vec![0.2, 0.2, 0.4, 0.4, 0.6, 0.6]);

        let l1b = nn.biases.get(&1).unwrap();
        assert_eq!(l1b.num_elements(), 3);
        assert_eq!(*l1b.get_data_as_slice(), vec![0.1, 0.1, 0.1]);

        let l2w = nn.weights.get(&2).unwrap();
        assert_eq!(l2w.shape(), MatrixShape::new(1, 3));
        assert_eq!(*l2w.data, vec![0.5, 0.5, 0.5]);

        let l2b = nn.biases.get(&2).unwrap();
        assert_eq!(l2b.num_elements(), 1);
        assert_eq!(*l2b.get_data_as_slice(), vec![0.1]);
    }

    #[test]
    #[should_panic]
    fn cannot_add_hiddlen_layer_before_input_layer() {
        let _ = NeuralNetworkBuilder::new().with_hidden_layer(
            3,
            Initializer::RandomBasic,
            ActivationFunction::Sigmoid,
        );
    }

    #[test]
    #[should_panic]
    fn cannot_add_output_layer_before_input_layer() {
        let _ = NeuralNetworkBuilder::new().with_output_layer(
            1,
            Initializer::RandomBasic,
            ActivationFunction::Sigmoid,
        );
    }

    #[test]
    fn panics_on_hidden_layer_with_invalid_weight_or_bias_dimensions() {
        // TODO: writing these tests is making it obvious that the builder should be returning an errors rather than panicking
        // ex you could have NNBuilderError::InvalidWeightsMatrixShape
        // ex you could have NNBuilderError::InvalidBiasVectorLength
        // ex you could have NNBuilderError::InvalidLayerOrder or MissingInputLayer would probably be better
        // then in the tests above, I could make sure that, for example, the panic is a result of the invalid weight matrix and not the ok-length bias vector, etc

        // All these test scenarios verify that the builder panics when passed a weights matrix or bias vector that is not the correct dimensions.
        // For a NN with the layer sizes 2x3x4x1,
        // the correct shapes of the weight matricies should be:
        // l0: no weights matrix
        // l1w 3x2
        // l2w: 4x3
        // l3w: 1x4

        let l1w_ok = Matrix::new_zero_matrix(3, 2);
        let l1b_ok = ColumnVector::new_zero_vector(3);
        let l2w_ok = Matrix::new_zero_matrix(4, 3);
        let l2b_ok = ColumnVector::new_zero_vector(4);

        // panics if the first hidden layer has an invalid weight matrix shape
        assert_eq!(
            panic::catch_unwind(|| {
                let _ = NeuralNetworkBuilder::new()
                    .with_input_layer(2)
                    .with_hidden_layer(
                        3,
                        Initializer::Manual(
                            Matrix::new_zero_matrix(1, 3), // should be 3x2,
                            l1b_ok.clone(),
                        ),
                        ActivationFunction::Sigmoid,
                    );
            })
            .is_err(),
            true
        );

        // panics if the first hidden layer has an invalid bias vector length
        assert_eq!(
            panic::catch_unwind(|| {
                let _ = NeuralNetworkBuilder::new()
                    .with_input_layer(2)
                    .with_hidden_layer(
                        3,
                        Initializer::Manual(
                            l1w_ok.clone(),
                            ColumnVector::new_zero_vector(2), // should be 3
                        ),
                        ActivationFunction::Sigmoid,
                    );
            })
            .is_err(),
            true
        );

        // panics if a non-first hidden layer has an invalid weight matrix shape
        let builder = NeuralNetworkBuilder::new()
            .with_input_layer(2)
            .with_hidden_layer(
                3,
                Initializer::Manual(l1w_ok.clone(), l1b_ok.clone()),
                ActivationFunction::Sigmoid,
            );

        assert_eq!(
            panic::catch_unwind(|| {
                let _ = builder.with_hidden_layer(
                    4,
                    Initializer::Manual(
                        Matrix::new_zero_matrix(4, 2), // should be 4x3
                        l2b_ok.clone(),
                    ),
                    ActivationFunction::Sigmoid,
                );
            })
            .is_err(),
            true
        );

        // panics if a non-first hidden layer has an invalid bias vector length
        let builder = NeuralNetworkBuilder::new()
            .with_input_layer(2)
            .with_hidden_layer(
                3,
                Initializer::Manual(l1w_ok.clone(), l1b_ok.clone()),
                ActivationFunction::Sigmoid,
            );
        assert_eq!(
            panic::catch_unwind(|| {
                let _ = builder.with_hidden_layer(
                    4,
                    Initializer::Manual(
                        l2w_ok.clone().clone(),
                        ColumnVector::new_zero_vector(2), // should be 4
                    ),
                    ActivationFunction::Sigmoid,
                );
            })
            .is_err(),
            true
        );

        // panics if the output layer has an invalid weight matrix shape
        let builder = NeuralNetworkBuilder::new()
            .with_input_layer(2)
            .with_hidden_layer(
                3,
                Initializer::Manual(l1w_ok.clone(), l1b_ok.clone()),
                ActivationFunction::Sigmoid,
            )
            .with_hidden_layer(
                4,
                Initializer::Manual(l2w_ok.clone(), l2b_ok.clone()),
                ActivationFunction::Sigmoid,
            );

        assert_eq!(
            panic::catch_unwind(|| {
                let _ = builder.with_output_layer(
                    1,
                    Initializer::Manual(
                        Matrix::new_zero_matrix(2, 4), // should be 1x4
                        ColumnVector::new_zero_vector(1),
                    ),
                    ActivationFunction::Sigmoid,
                );
            })
            .is_err(),
            true
        );

        // panics if the output layer has an invalid bias vector length
        let builder = NeuralNetworkBuilder::new()
            .with_input_layer(2)
            .with_hidden_layer(
                3,
                Initializer::Manual(l1w_ok, l1b_ok),
                ActivationFunction::Sigmoid,
            )
            .with_hidden_layer(
                4,
                Initializer::Manual(l2w_ok, l2b_ok),
                ActivationFunction::Sigmoid,
            );

        assert_eq!(
            panic::catch_unwind(|| {
                let _ = builder.with_output_layer(
                    1,
                    Initializer::Manual(
                        Matrix::new_zero_matrix(1, 4),
                        ColumnVector::new_zero_vector(3), // should be 1
                    ),
                    ActivationFunction::Sigmoid,
                );
            })
            .is_err(),
            true
        );
    }
}
