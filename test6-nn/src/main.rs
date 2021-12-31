use common::linalg::square;
use common::linalg::{ColumnVector, Matrix, RowsMatrixBuilder};
use common::sigmoid::{sigmoid_prime_vector, sigmoid_vector};
use common::{column_vec_of_random_values_from_distribution, column_vector};

fn z(
    weights_matrix: &Matrix,
    inputs_vector: &ColumnVector,
    biases_vector: &ColumnVector,
) -> ColumnVector {
    weights_matrix.mult_vector(inputs_vector).add(biases_vector)
}

pub struct SimpleNeuralNetwork {
    sizes: Vec<usize>,

    // A vec of weight matricies - one for each inter-layer step
    // For each weight matrix, a row corresponds to the input neurons in the previous layer
    // and a particular neuron in the next layer.
    // Thus, the dimensions will be [rows x columns] [# neurons in the previous layer x # number neurons in the next layer]
    weights: Vec<Matrix>,

    // A vec of ColumnVectors - one for inter-layer step.
    // Each vector will have length equal to the number of neurons in the next layer.
    biases: Vec<ColumnVector>,
}

// Note that 3B1B does not do the divide by 2 and he ends up with a 2 in the derivitive function.
// Neilson does the divide by 2
// I'm doing the divide by 2
fn quadratic_cost(expected: &ColumnVector, actual: &ColumnVector) -> f64 {
    if expected.num_elements() != actual.num_elements() {
        panic!("expected and actual outputs must have the same length");
    }

    expected
        .iter_with(actual)
        .map(|(exp, act)| exp - act)
        .map(square)
        .sum::<f64>()
        / 2.0
}

impl SimpleNeuralNetwork {
    pub fn new(sizes: Vec<usize>) -> Self {
        let mut biases = Vec::new();

        for i in 1..sizes.len() {
            let biases_column_vector =
                column_vec_of_random_values_from_distribution(0.0, 1.0, sizes[i]);
            biases.push(biases_column_vector);
        }

        let mut weights = Vec::new();

        for l in 1..sizes.len() {
            let weights_matrix = Matrix::new_matrix_with_random_values_from_normal_distribution(
                sizes[l],
                sizes[l - 1],
                0.0,
                1.0,
            );
            weights.push(weights_matrix);
        }

        Self {
            sizes,
            weights,
            biases,
        }
    }

    pub fn num_layers(&self) -> usize {
        self.sizes.len()
    }

    pub fn feed_forward(&self, input_activations: &ColumnVector) -> ColumnVector {
        let mut activation_vector = input_activations.clone();

        for i_step in 0..self.num_layers() - 1 {
            let z_vec = z(
                &self.weights[i_step],
                &activation_vector,
                &self.biases[i_step],
            );
            activation_vector = sigmoid_vector(&z_vec);
        }

        activation_vector
    }

    /// Feed forward capturing the intermediate z values and activation values.
    /// # Arguments
    /// * `input_activations` - The input activations
    /// # Returns
    /// A Vec<FeedForwardIntermediateValues> - one for each layer other than the first (input) layer (simiar to weights and biases).
    fn feed_forward_capturing_intermediates(
        &self,
        input_activations: &ColumnVector,
    ) -> Vec<FeedForwardIntermediateValues> {
        let mut intermediates = Vec::new();
        let mut activation_vector = input_activations.clone();

        for l in 0..self.num_layers() {
            // println!("feed_forward layer: {}", l);

            if l == 0 {
                intermediates.push(FeedForwardIntermediateValues {
                    // Filling the layer 0 z vector with NAN (Not a Number) because it is never used and isn't a valid computation.
                    z_vector: ColumnVector::fill_new(f64::NAN, activation_vector.num_elements()),
                    activations_vector: activation_vector.clone(),
                });
            } else {
                let z_vec = z(
                    &self.weights[l - 1],
                    &activation_vector,
                    &self.biases[l - 1],
                );
                // println!("feed_forward - z_vec: {:?}", z_vec);
                activation_vector = sigmoid_vector(&z_vec);

                intermediates.push(FeedForwardIntermediateValues {
                    z_vector: z_vec.clone(),
                    activations_vector: activation_vector.clone(),
                });
            }
        }

        intermediates
    }

    pub fn cost_for_single_training_example(
        &self,
        inputs: &ColumnVector,
        expected_outputs: &ColumnVector,
    ) -> f64 {
        if inputs.num_elements() != self.sizes[0] {
            panic!(
                "inputs must have the same number of elements as the number of neurons in the input layer"
            );
        }

        let outputs = self.feed_forward(&inputs);

        if outputs.num_elements() != expected_outputs.num_elements() {
            panic!("outputs and expected_outputs must have the same length");
        }

        quadratic_cost(expected_outputs, &outputs)
    }

    /// Computes the cost of the neural network across the entire setup `all_inputs` and `expected_outputs`.
    /// Both `all_inputs` and `expected_outputs` should be matricies consisting of one column vector per input / expected output.
    /// So the number of rows in `all_inputs` should equal the number of neurons in the input layer
    /// and the number of rows in the `expected_outputs` should be the number of neurons in the output layer.
    /// The number of columns in both `all_inputs` and `all_outputs` should both match and equal the number of training examples.
    pub fn cost_for_training_set_iterative_impl(
        &self,
        all_inputs: &Matrix,
        expected_outputs: &Matrix,
    ) -> f64 {
        let mut cost = 0.0;
        for i_column in 0..all_inputs.num_columns() {
            let next_input_vector = all_inputs.extract_column_vector(i_column);
            let next_desired_output_vector = expected_outputs.extract_column_vector(i_column);

            let next_cost = self
                .cost_for_single_training_example(&next_input_vector, &next_desired_output_vector);
            cost += next_cost;
        }

        cost / (all_inputs.num_columns() as f64)
    }

    // Backprop Equation (the one that is unlabeled but follows after BP1a. I assume then ment to label it BP1b)
    // from the Neilson book
    fn error_last_layer(
        &self,
        output_activations_vector: &ColumnVector,
        expected_outputs_vector: &ColumnVector,
        output_layer_z_vector: &ColumnVector,
    ) -> ColumnVector {
        let mut part1 = output_activations_vector.minus(expected_outputs_vector);
        let part2 = sigmoid_prime_vector(output_layer_z_vector);
        part1.hadamard_product_in_place(&part2);
        part1
    }

    // Backprop Equation BP2 from the Neilson book
    fn error_any_layer_but_last(
        &self,
        layer: usize,
        error_vector_for_plus_one_layer: &ColumnVector,
        this_layer_z_vector: &ColumnVector,
    ) -> ColumnVector {
        // there's once less weight matrix than layer since the input layer doesn't have a weight matrix.
        // so if we are on layer 2, weights[2] will be the weights for layer 3 (which is what we want in EQ 2)
        let weight_matrix = self.weights.get(layer).unwrap();
        let weight_matrix_transpose = weight_matrix.transpose();

        let mut part1 = weight_matrix_transpose.mult_vector(error_vector_for_plus_one_layer);
        let part2 = sigmoid_prime_vector(this_layer_z_vector);

        part1.hadamard_product_in_place(&part2);
        part1
    }

    /// Returns a Vec of column vectors representing the errors at each neuron at each layer from L-1 to 1
    /// where layer L-1 is the output layer and layer 0 is the input layer.
    fn backpropate_errors(
        &self,
        expected_outputs_vector: &ColumnVector,
        feedforward_intermediate_values: &Vec<FeedForwardIntermediateValues>,
    ) -> Vec<ColumnVector> {
        // loop through the layers from back to front, and compute the error at each one.
        // Create a column vector representing the errors at each layer
        let mut error_vectors = Vec::<ColumnVector>::new();

        let last_layer_index = self.num_layers() - 1;

        for l in (1..self.num_layers()).rev() {
            if l == last_layer_index {
                let last_layer_z_values = &feedforward_intermediate_values.last().unwrap().z_vector;
                let last_layer_activations_vector = &feedforward_intermediate_values
                    .last()
                    .unwrap()
                    .activations_vector;

                let layer_errors_vector = self.error_last_layer(
                    &last_layer_activations_vector,
                    expected_outputs_vector,
                    &last_layer_z_values,
                );

                error_vectors.push(layer_errors_vector);
            } else {
                // we're working backwards from the last layer to the input layer, but
                // filling up the errors_vectors in reverse order (ex [0] is first, etc)
                let error_vector_for_plus_one_layer = error_vectors.last().unwrap(); // this will get the most recently done layer, which will be the l+1 th layer

                // we're going to need the z vector for the input layer, too, but how's that even possible?
                // we can't get the z vector for the input layer, because their are no weights/biases associated to it...
                // turns out we don't do it... we only do this down to layer 2 - see http://neuralnetworksanddeeplearning.com/chap2.html#the_backpropagation_algorithm

                let this_layer_z_vector = &feedforward_intermediate_values.get(l).unwrap().z_vector;

                // println!("this_layer_z_vector:\n{}", this_layer_z_vector);

                let this_layer_errors_vector = self.error_any_layer_but_last(
                    l,
                    error_vector_for_plus_one_layer,
                    this_layer_z_vector,
                );
                error_vectors.push(this_layer_errors_vector);
            }
        }

        // println!("\nerrors_vectors");
        // let mut layer = 2_usize;
        // for ev in error_vectors.iter() {
        //     println!("  - layer {}: \n{}", layer, ev);
        //     println!("{}", ev);
        //     layer -= 1;
        // }

        error_vectors

        // error_vectors is in order from L down to l+1 (i.e. the output layer down to the 1th layer (not the 0th layer)).
        // depending on what comes next, we may or may not wat to reverse the order to keep things consistent with how we normally order things
    }

    fn update_weights_and_biases(
        &mut self,
        error_vectors_for_each_training_example: &Vec<Vec<ColumnVector>>,
        intermediates_for_each_training_example: &Vec<Vec<FeedForwardIntermediateValues>>,
        learning_rate: f64,
    ) {
        // error_vectors_for_each_training_example is like per training example, per layer, a column vector of errors per neuron

        let num_training_examples = error_vectors_for_each_training_example.len();

        // for layers L-1 down to 1, update the weights and biases

        // the layers_index_in_errors_vec is in order from L to layer 0
        let mut layer_index_in_errors_vec = 0; // L-1

        for l in (1..self.num_layers()).rev() {
            // println!("l: {}", l);

            // let mut weights_partials_vector_avg = Matrix::new_column_vector(&vec![0.0; self.sizes[l]]);
            let mut weights_partials_matrix_avg =
                Matrix::new_zero_matrix(self.sizes[l], self.sizes[l - 1]);
            let mut bias_partials_vector_avg = ColumnVector::new_zero_vector(self.sizes[l]);

            for i_training_ex in 0..num_training_examples {
                let error_vectors_for_this_training_example =
                    &error_vectors_for_each_training_example[i_training_ex];

                let intermediates_for_this_training_example =
                    &intermediates_for_each_training_example[i_training_ex];

                let this_layer_errors_vector = error_vectors_for_this_training_example
                    .get(layer_index_in_errors_vec)
                    .unwrap();

                let previous_layer_activations_vector = &intermediates_for_this_training_example
                    .get(l - 1)
                    .unwrap()
                    .activations_vector;
                let previous_layer_activations_vector_transpose =
                    previous_layer_activations_vector.transpose();

                let weights_grad = this_layer_errors_vector
                    .mult_matrix(&previous_layer_activations_vector_transpose);

                weights_partials_matrix_avg.add_in_place(&weights_grad);
                bias_partials_vector_avg.plus_in_place(this_layer_errors_vector);
            }

            let learning_rate_over_num_training_examples =
                learning_rate / (num_training_examples as f64);

            weights_partials_matrix_avg
                .multiply_by_scalar_in_place(learning_rate_over_num_training_examples);
            bias_partials_vector_avg
                .multiply_by_scalar_in_place(learning_rate_over_num_training_examples);

            // -1 because one less than num layers
            let weights = self.weights.get_mut(l - 1).unwrap();
            weights.subtract_in_place(&weights_partials_matrix_avg);

            // -1 because one less than num layers
            let biases = self.biases.get_mut(l - 1).unwrap();
            biases.minus_in_place(&bias_partials_vector_avg);

            layer_index_in_errors_vec += 1;
        }
    }

    pub fn train(
        &mut self,
        training_inputs: &Matrix,
        expected_outputs: &Matrix,
        epocs: usize,
        learning_rate: f64,
    ) {
        if training_inputs.num_columns() != expected_outputs.num_columns() {
            panic!("the number of training inputs must match the number of training outputs");
        }

        let initial_cost =
            self.cost_for_training_set_iterative_impl(&training_inputs, &expected_outputs);

        println!("initial cost across entire training set: {}", initial_cost,);

        for _ in 0..epocs {
            let mut error_vectors_for_each_training_example = Vec::new();
            let mut intermediates_for_each_training_example = Vec::new();

            for i_training_example in 0..training_inputs.num_columns() {
                let next_training_inputs_vector =
                    training_inputs.extract_column_vector(i_training_example);
                let next_training_expected_output_vector =
                    expected_outputs.extract_column_vector(i_training_example);

                let intermediates =
                    self.feed_forward_capturing_intermediates(&next_training_inputs_vector);

                let error_vectors_for_this_training_example =
                    self.backpropate_errors(&next_training_expected_output_vector, &intermediates);

                intermediates_for_each_training_example.push(intermediates);
                error_vectors_for_each_training_example
                    .push(error_vectors_for_this_training_example);
            }

            self.update_weights_and_biases(
                &error_vectors_for_each_training_example,
                &intermediates_for_each_training_example,
                learning_rate,
            );
        }

        let final_cost =
            self.cost_for_training_set_iterative_impl(&training_inputs, &expected_outputs);
        println!(
            "\ncost across entire training set after {} epocs: {}",
            epocs, final_cost,
        );
    }
}

struct FeedForwardIntermediateValues {
    z_vector: ColumnVector,
    activations_vector: ColumnVector,
}

fn main() {
    // 3x2 network
    // single weights / biases
    // weights matrix -> 2x3

    let inputs = column_vector![0.0, 0.5, 1.0];

    let weights = RowsMatrixBuilder::new()
        .with_row(&[0.5, 0.5, 0.5])
        .with_row(&[1.0, 1.0, 1.0])
        .build();

    let biases = column_vector![0.1, 0.1];

    let nn = SimpleNeuralNetwork {
        sizes: vec![3, 2],
        weights: vec![weights],
        biases: vec![biases],
    };

    let outputs = nn.feed_forward(&inputs);

    println!("outputs: {:?}", outputs);

    // println!("gaussian");
    // let normal = Normal::new(0.0, 1.0).unwrap();
    // for _ in 0..10 {
    //     let v = normal.sample(&mut rand::thread_rng());
    //     println!("{} is from a N(0, 1) distribution", v)
    // }

    // let weights_vector = column_vector![1.0, 2.0, 3.0];
    // let activations_vector = column_vector![5.0, 7.0, 11.0];
    // // let bias = 13.0;

    // // let z = z(&weights_vector, &activations_vector, &biases);
    // // println!("z = {:?}", z);

    // let mut nn = SimpleNeuralNetwork::new(vec![2, 3, 1]);
}

#[cfg(test)]
mod tests {
    use super::*;
    use common::linalg::ColumnsMatrixBuilder;
    use common::scalar_valued_multivariable_point::ScalarValuedMultivariablePoint;
    use float_cmp::approx_eq;
    use time_test::time_test;

    pub fn get_simple_two_layer_nn_for_test() -> SimpleNeuralNetwork {
        let num_neurons_layer_0 = 3;
        let num_neurons_layer_1 = 2;

        // Layer 0 (input): 3 neurons
        // Layer 1 (output): 2 neurons
        // weights matrix -> 2x3

        let weights = RowsMatrixBuilder::new()
            .with_row(&[0.5, 0.5, 0.5])
            .with_row(&[1.0, 1.0, 1.0])
            .build();

        let biases = column_vector![0.1, 0.1];

        let nn = SimpleNeuralNetwork {
            sizes: vec![num_neurons_layer_0, num_neurons_layer_1],
            weights: vec![weights],
            biases: vec![biases],
        };

        nn
    }

    pub fn get_simple_get_2_3_1_nn_for_test() -> SimpleNeuralNetwork {
        let num_neurons_layer_0 = 2;
        let num_neurons_layer_1 = 3;
        let num_neurons_layer_2 = 1;

        // 3x2
        let weights_0 = RowsMatrixBuilder::new()
            .with_row(&[0.2, 0.2])
            .with_row(&[0.4, 0.4])
            .with_row(&[0.6, 0.6])
            .build();

        // 1x3
        let weights_1 = RowsMatrixBuilder::new().with_row(&[0.5, 0.5, 0.5]).build();

        let biases_0 = column_vector![0.1, 0.1, 0.1];
        let biases_1 = column_vector![0.1];

        let weights = vec![weights_0, weights_1];
        let biases = vec![biases_0, biases_1];

        let nn = SimpleNeuralNetwork {
            sizes: vec![
                num_neurons_layer_0,
                num_neurons_layer_1,
                num_neurons_layer_2,
            ],
            weights,
            biases,
        };

        nn
    }

    pub fn get_three_layer_multiple_output_nn_for_test() -> SimpleNeuralNetwork {
        let num_neurons_layer_0 = 3;
        let num_neurons_layer_1 = 5;
        let num_neurons_layer_2 = 2;

        // W0: 5x3
        // W1: 2x5

        let weights_0 = RowsMatrixBuilder::new()
            .with_row(&[0.2, 0.2, 0.2])
            .with_row(&[0.4, 0.4, 0.4])
            .with_row(&[0.6, 0.6, 0.6])
            .with_row(&[0.8, 0.8, 0.8])
            .with_row(&[1.0, 1.0, 1.0])
            .build();

        let weights_1 = RowsMatrixBuilder::new()
            .with_row(&[0.5, 0.5, 0.5, 0.5, 0.5])
            .with_row(&[0.6, 0.6, 0.6, 0.6, 0.6])
            .build();

        let biases_0 = column_vector![0.1, 0.1, 0.1, 0.1, 0.1];
        let biases_1 = column_vector![0.1, 0.1];

        let weights = vec![weights_0, weights_1];
        let biases = vec![biases_0, biases_1];

        let nn = SimpleNeuralNetwork {
            sizes: vec![
                num_neurons_layer_0,
                num_neurons_layer_1,
                num_neurons_layer_2,
            ],
            weights,
            biases,
        };

        nn
    }

    #[test]
    fn test_z_vec() {
        // Layer N: 3 neurons
        // L N-1: 2 neurons
        // weights matrix -> 2x3
        let weights = RowsMatrixBuilder::new()
            .with_row(&[1.0, 2.0, 3.0])
            .with_row(&[4.0, 5.0, 6.0])
            .build();

        let biases = column_vector![0.5, 0.5];
        let inputs = column_vector![1.0, 2.0, 3.0];

        let outputs = z(&weights, &inputs, &biases);

        assert_eq!(outputs.num_elements(), 2);
        assert_eq!(outputs.get(0), 14.5);
        assert_eq!(outputs.get(1), 32.5);
    }

    #[test]
    fn feed_forward_works_simple_two_layer() {
        let num_neurons_layer_0 = 3;
        let num_neurons_layer_1 = 2;

        // Layer N: 3 neurons
        // L N-1: 2 neurons
        // weights matrix -> 2x3

        let inputs = column_vector![0.0, 0.5, 1.0];

        let weights = RowsMatrixBuilder::new()
            .with_row(&[0.5, 0.5, 0.5])
            .with_row(&[1.0, 1.0, 1.0])
            .build();

        let biases = column_vector![0.1, 0.1];

        let nn = SimpleNeuralNetwork {
            sizes: vec![num_neurons_layer_0, num_neurons_layer_1],
            weights: vec![weights],
            biases: vec![biases],
        };

        let outputs = nn.feed_forward(&inputs);
        assert_eq!(outputs.num_elements(), 2);
        assert_eq!(outputs.get(0), 0.7005671424739729);
        assert_eq!(outputs.get(1), 0.8320183851339245);
    }

    #[test]
    fn feed_forward_works_simple_three_layer() {
        let num_neurons_layer_0 = 3;
        let num_neurons_layer_1 = 5;
        let num_neurons_layer_2 = 2;

        // W0: 5x3
        // W1: 2x5

        let inputs = column_vector![0.0, 0.5, 1.0];

        let weights_0 = RowsMatrixBuilder::new()
            .with_row(&[0.2, 0.2, 0.2])
            .with_row(&[0.4, 0.4, 0.4])
            .with_row(&[0.6, 0.6, 0.6])
            .with_row(&[0.8, 0.8, 0.8])
            .with_row(&[1.0, 1.0, 1.0])
            .build();

        let weights_1 = RowsMatrixBuilder::new()
            .with_row(&[0.5, 0.5, 0.5, 0.5, 0.5])
            .with_row(&[0.6, 0.6, 0.6, 0.6, 0.6])
            .build();

        let biases_0 = column_vector![0.1, 0.1, 0.1, 0.1, 0.1];
        let biases_1 = column_vector![0.1, 0.1];

        // manually compute the correct output to use in later assertion
        let z0 = weights_0.mult_vector(&inputs).plus(&biases_0);
        let sz0 = sigmoid_vector(&z0);
        println!("sz0: {:?}", sz0);

        let z1 = weights_1.mult_vector(&sz0).plus(&biases_1);
        let sz1 = sigmoid_vector(&z1);
        println!("sz1: {:?}", sz1);

        println!("\n now doing with feed_forward");

        let weights = vec![weights_0, weights_1];
        let biases = vec![biases_0, biases_1];

        let nn = SimpleNeuralNetwork {
            sizes: vec![
                num_neurons_layer_0,
                num_neurons_layer_1,
                num_neurons_layer_2,
            ],
            weights,
            biases,
        };

        let outputs = nn.feed_forward(&inputs);

        println!("outputs: {:?}", outputs);
        // assert_eq!(1, 0);

        assert_eq!(outputs.num_elements(), 2);
        assert_eq!(outputs.get(0), sz1.get(0));
        assert_eq!(outputs.get(1), sz1.get(1));

        // the actual outputs, which should be the same as the manually computed outputs
        assert_eq!(outputs.get(0), 0.8707823298624764);
        assert_eq!(outputs.get(1), 0.9063170030285769);
    }

    #[test]
    fn feed_forward_works_simple_three_layer_using_feed_forward_capturing() {
        // This is basically a copy of the above test except that it uses the new version of feed_forward which captures all the intermediate values.
        // Once the dust settles on my backprop implementation, I can probably just use the capturing one for all the training stuff, but
        // I'll still need the orig one for the post-training prediction phase.
        let num_neurons_layer_0 = 3;
        let num_neurons_layer_1 = 5;
        let num_neurons_layer_2 = 2;

        // W0: 5x3
        // W1: 2x5

        let inputs = column_vector![0.0, 0.5, 1.0];

        let weights_0 = RowsMatrixBuilder::new()
            .with_row(&[0.2, 0.2, 0.2])
            .with_row(&[0.4, 0.4, 0.4])
            .with_row(&[0.6, 0.6, 0.6])
            .with_row(&[0.8, 0.8, 0.8])
            .with_row(&[1.0, 1.0, 1.0])
            .build();

        let weights_1 = RowsMatrixBuilder::new()
            .with_row(&[0.5, 0.5, 0.5, 0.5, 0.5])
            .with_row(&[0.6, 0.6, 0.6, 0.6, 0.6])
            .build();

        let biases_0 = column_vector![0.1, 0.1, 0.1, 0.1, 0.1];
        let biases_1 = column_vector![0.1, 0.1];

        // manually compute the correct output to use in later assertion
        let z0 = weights_0.mult_vector(&inputs).plus(&biases_0);
        let sz0 = sigmoid_vector(&z0);
        println!("sz0: {:?}", sz0);

        let z1 = weights_1.mult_vector(&sz0).plus(&biases_1);
        let sz1 = sigmoid_vector(&z1);
        println!("sz1: {:?}", sz1);

        println!("\n now doing with feed_forward");

        let weights = vec![weights_0, weights_1];
        let biases = vec![biases_0, biases_1];

        let nn = SimpleNeuralNetwork {
            sizes: vec![
                num_neurons_layer_0,
                num_neurons_layer_1,
                num_neurons_layer_2,
            ],
            weights,
            biases,
        };

        let intermediates = nn.feed_forward_capturing_intermediates(&inputs);
        assert_eq!(intermediates.len(), 3);
        let final_step_values = intermediates.last().unwrap();
        let outputs = final_step_values.activations_vector.clone();

        println!("outputs: {:?}", outputs);
        // assert_eq!(1, 0);

        assert_eq!(outputs.num_elements(), 2);
        assert_eq!(outputs.get(0), sz1.get(0));
        assert_eq!(outputs.get(1), sz1.get(1));

        // the actual outputs, which should be the same as the manually computed outputs
        assert_eq!(outputs.get(0), 0.8707823298624764);
        assert_eq!(outputs.get(1), 0.9063170030285769);
    }

    #[test]
    pub fn test_quadratic_cost_fn() {
        let inputs = column_vector![0.0, 0.5, 1.0];
        let targets = column_vector![0.0, 0.5, 1.0];
        let cost = quadratic_cost(&inputs, &targets);
        assert_eq!(cost, 0.0);

        let inputs = column_vector![4.0, 4.0];
        let targets = column_vector![2.0, 2.0];
        let cost = quadratic_cost(&inputs, &targets);
        assert_eq!(cost, 4.0);
    }

    #[test]
    pub fn test_cost_for_single_training_example_single_output_neuron() {
        let nn = get_simple_get_2_3_1_nn_for_test();

        // let's go for y(x0, x1) = x0 + x1;
        let input_example = column_vector![1.0, 1.0];
        let expected_output = column_vector![3.0];

        let c0 = nn.cost_for_single_training_example(&input_example, &expected_output);

        // manually compute the expected cost for the single neuron in the last layer
        let a_output = nn.feed_forward(&input_example);
        let a_output_value = a_output.into_value();

        let manual_cost = (3.0 - a_output_value).powi(2) / 2.0;

        println!("a_output: \n{}", a_output_value);
        println!("manual_cost: \n{}", manual_cost);

        assert_eq!(c0, manual_cost);
    }

    #[test]
    pub fn test_cost_for_single_training_example_multiple_output_neurons() {
        let nn = get_three_layer_multiple_output_nn_for_test();

        let input_example = column_vector![0.0, 0.5, 1.0];
        let expected_output_vector = column_vector![2.0, 2.0];

        let c0 = nn.cost_for_single_training_example(&input_example, &expected_output_vector);

        // manually compute the expected cost for the single neuron in the last layer
        // this uses a different method to compute the cost, but it is mathematically equivalent to that used
        // in cost_for_single_training_example
        let actual_output_vector = nn.feed_forward(&input_example);
        let diff_vec = expected_output_vector.minus(&actual_output_vector);
        let length_of_diff_vector = diff_vec.vec_length();
        let length_of_diff_vector_squared = length_of_diff_vector * length_of_diff_vector;
        let over_two = length_of_diff_vector_squared / 2.0;

        println!("\nactual_output_vector: \n{}", actual_output_vector);
        println!("diff_vec: \n{}", diff_vec);
        println!("length_of_diff_vector: \n{:?}", length_of_diff_vector);
        println!(
            "length_of_diff_vector_squared: \n{:?}",
            length_of_diff_vector_squared
        );
        println!("over_two: \n{:?}", over_two);

        println!("c0: \n{}", c0);

        assert_eq!(c0, over_two);
    }

    #[test]
    pub fn test_cost_for_training_set_iterative_impl() {
        let mut nn = get_three_layer_multiple_output_nn_for_test();

        let input_examples = ColumnsMatrixBuilder::new()
            .with_column(&[0.0, 0.5, 0.9])
            .with_column(&[0.0, 0.5, 1.0])
            .with_column(&[0.0, 0.5, 1.1])
            .build();

        let expected_outputs_vector = ColumnsMatrixBuilder::new()
            .with_column(&[2.0, 2.0])
            .with_column(&[2.0, 2.0])
            .with_column(&[2.0, 2.0])
            .build();

        let c = nn.cost_for_training_set_iterative_impl(&input_examples, &expected_outputs_vector);

        println!("\nc: \n{}", c);

        let c0 = nn.cost_for_single_training_example(
            &input_examples.extract_column_vector(0),
            &expected_outputs_vector.extract_column_vector(0),
        );

        let c1 = nn.cost_for_single_training_example(
            &input_examples.extract_column_vector(1),
            &expected_outputs_vector.extract_column_vector(1),
        );

        let c2 = nn.cost_for_single_training_example(
            &input_examples.extract_column_vector(2),
            &expected_outputs_vector.extract_column_vector(2),
        );

        println!("\nc0: {}", c0);
        println!("c1: {}", c1);
        println!("c2: {}", c2);

        println!("\nc0: {}", c0);

        let c_avg = (c0 + c1 + c2) / 3.0;
        assert_eq!(c, c_avg);

        // from here on, I'm testing the towards_backprop stuff

        println!("\nIn the test - doing the pre backprop stuff: {}", c0);
        let mut error_vectors_for_each_training_example = Vec::new();
        let mut intermediates_for_each_training_example = Vec::new();

        for i_training_example in 0..input_examples.num_columns() {
            println!("\nfor the {}th training example", i_training_example);
            let next_inputs_vector = input_examples.extract_column_vector(i_training_example);
            let next_expected_outputs_vector =
                expected_outputs_vector.extract_column_vector(i_training_example);
            let intermediates = nn.feed_forward_capturing_intermediates(&next_inputs_vector);
            println!("for input {}", i_training_example);
            println!("intermediates");
            for i_intermediate in 0..intermediates.len() {
                println!("intermediate {}:", i_intermediate);
                println!(" - z_vector:");
                println!("{}", intermediates[i_intermediate].z_vector);
                println!(" - activations_vector:");
                println!("{}", intermediates[i_intermediate].activations_vector);
            }
            let error_vectors_for_this_training_example =
                nn.backpropate_errors(&next_expected_outputs_vector, &intermediates);
            error_vectors_for_each_training_example.push(error_vectors_for_this_training_example);
            intermediates_for_each_training_example.push(intermediates);
        }

        // now we have the errors back propagated
        // update the weights and biases from layers L-1 to 1, where layer L-1 is the output and layer 0 is the input

        nn.update_weights_and_biases(
            &error_vectors_for_each_training_example,
            &intermediates_for_each_training_example,
            0.1,
        );

        // assert_eq!(1, 0);
    }

    const ORANGE: f64 = 0.0;
    const BLUE: f64 = 1.0;

    fn get_data_set_1() -> Vec<ScalarValuedMultivariablePoint> {
        // fake data roughly based on https://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=gauss&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=3,1&seed=0.22934&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false
        let training_data = vec![
            ScalarValuedMultivariablePoint::new_3d(-2.0, -2.0, ORANGE),
            ScalarValuedMultivariablePoint::new_3d(-2.0, -2.0, ORANGE),
            ScalarValuedMultivariablePoint::new_3d(-2.0, -2.0, ORANGE),
            ScalarValuedMultivariablePoint::new_3d(-2.0, -2.0, ORANGE),
            ScalarValuedMultivariablePoint::new_3d(-2.0, -2.0, ORANGE),
            ScalarValuedMultivariablePoint::new_3d(-2.0, -2.0, ORANGE),
            ScalarValuedMultivariablePoint::new_3d(-2.0, -2.0, ORANGE),
            ScalarValuedMultivariablePoint::new_3d(-2.0, -2.0, ORANGE),
            ScalarValuedMultivariablePoint::new_3d(-2.0, -2.0, ORANGE),
            ScalarValuedMultivariablePoint::new_3d(-2.0, -2.0, ORANGE),
            ScalarValuedMultivariablePoint::new_3d(2.0, 2.0, BLUE),
            ScalarValuedMultivariablePoint::new_3d(2.0, 2.0, BLUE),
            ScalarValuedMultivariablePoint::new_3d(2.0, 2.0, BLUE),
            ScalarValuedMultivariablePoint::new_3d(2.0, 2.0, BLUE),
            ScalarValuedMultivariablePoint::new_3d(2.0, 2.0, BLUE),
            ScalarValuedMultivariablePoint::new_3d(2.0, 2.0, BLUE),
            ScalarValuedMultivariablePoint::new_3d(2.0, 2.0, BLUE),
            ScalarValuedMultivariablePoint::new_3d(2.0, 2.0, BLUE),
            ScalarValuedMultivariablePoint::new_3d(2.0, 2.0, BLUE),
            ScalarValuedMultivariablePoint::new_3d(2.0, 2.0, BLUE),
        ];
        training_data
    }

    #[test]
    fn test_nn() {
        time_test!();
        let training_data = get_data_set_1();

        // 2 x 3 x 1
        let num_neurons_layer_0 = 2;
        let num_neurons_layer_1 = 3;
        let num_neurons_layer_2 = 1;

        // W0: 3x2 (l1size x l0size)
        // W1: 1x3 (l2size x l1size)

        let weights_0 = RowsMatrixBuilder::new()
            .with_row(&[0.2, 0.2])
            .with_row(&[0.4, 0.4])
            .with_row(&[0.6, 0.6])
            .build();

        let weights_1 = RowsMatrixBuilder::new().with_row(&[0.5, 0.5, 0.5]).build();

        let biases_0 = column_vector![0.1, 0.1, 0.1];
        let biases_1 = column_vector![0.1];

        let weights = vec![weights_0, weights_1];
        let biases = vec![biases_0, biases_1];

        let mut nn = SimpleNeuralNetwork {
            sizes: vec![
                num_neurons_layer_0,
                num_neurons_layer_1,
                num_neurons_layer_2,
            ],
            weights,
            biases,
        };

        // need a matrix with rows = size of input x num data points
        let mut training_inputs = ColumnsMatrixBuilder::new();
        let mut expected_outputs = ColumnsMatrixBuilder::new();

        for i_data in 0..training_data.len() {
            training_inputs.push_column(&training_data[i_data].independant);
            expected_outputs.push_column(&[training_data[i_data].dependant]);
        }

        let training_inputs = training_inputs.build();
        let expected_outputs = expected_outputs.build();

        let epocs = 10000;
        let learning_rate = 2.0;

        nn.train(&training_inputs, &expected_outputs, epocs, learning_rate);

        // predict
        let prediction_input = column_vector![2.0, 2.0];
        let expected_output = column_vector![1.0];
        let predicted_output_0 = nn.feed_forward(&prediction_input).into_value();
        println!("predicted_output_0: {}", &predicted_output_0);
        let cost_of_predicted_0 =
            nn.cost_for_single_training_example(&prediction_input, &expected_output);
        println!("cost_of_predicted_0: \n{}", &cost_of_predicted_0);

        // predict
        println!("second prediction");
        let prediction_input = column_vector![-2.0, -2.0];
        let expected_output = column_vector![0.0];
        let predicted_output_1 = nn.feed_forward(&prediction_input).into_value();
        println!("predicted_output_1: {}", &predicted_output_1);
        let cost_of_predicted_1 =
            nn.cost_for_single_training_example(&prediction_input, &expected_output);
        println!("cost_of_predicted_1: \n{}", &cost_of_predicted_1);

        assert!(approx_eq!(f64, predicted_output_0, BLUE, epsilon = 0.01));
        assert!(approx_eq!(f64, predicted_output_1, ORANGE, epsilon = 0.01));

        assert!(approx_eq!(f64, cost_of_predicted_0, 0.0, epsilon = 0.0001));
        assert!(approx_eq!(f64, cost_of_predicted_1, 0.0, epsilon = 0.0001));
    }

    #[test]
    fn test_nn_using_constructor_for_random_initial_weights_and_biases() {
        time_test!();
        // try the same data set as before but use the NN constructor to initialize with random weights/biases
        let training_data = get_data_set_1();

        // 2 x 3 x 1
        let mut nn = SimpleNeuralNetwork::new(vec![2, 3, 1]);

        println!("initial weights:");
        for w in nn.weights.iter() {
            println!("{}", w);
        }

        println!("initial biases:");
        for b in nn.biases.iter() {
            println!("{}", b);
        }

        // need a matrix with rows = size of input x num data points
        let mut training_inputs = ColumnsMatrixBuilder::new();
        let mut expected_outputs = ColumnsMatrixBuilder::new();

        for i_data in 0..training_data.len() {
            training_inputs.push_column(&training_data[i_data].independant);
            expected_outputs.push_column(&[training_data[i_data].dependant]);
        }

        let training_inputs = training_inputs.build();
        let expected_outputs = expected_outputs.build();

        let epocs = 7000;
        let learning_rate = 4.0;

        nn.train(&training_inputs, &expected_outputs, epocs, learning_rate);

        // predict
        let prediction_input = column_vector![2.0, 2.0];
        let expected_output = column_vector![1.0];
        let predicted_output_0 = nn.feed_forward(&prediction_input).into_value();
        println!("predicted_output_0: {}", &predicted_output_0);
        let cost_of_predicted_0 =
            nn.cost_for_single_training_example(&prediction_input, &expected_output);
        println!("cost_of_predicted_0: \n{}", &cost_of_predicted_0);

        // predict
        println!("second prediction");
        let prediction_input = column_vector![-2.0, -2.0];
        let expected_output = column_vector![0.0];
        let predicted_output_1 = nn.feed_forward(&prediction_input).into_value();
        println!("predicted_output_1: {}", &predicted_output_1);
        let cost_of_predicted_1 =
            nn.cost_for_single_training_example(&prediction_input, &expected_output);
        println!("cost_of_predicted_1: \n{}", &cost_of_predicted_1);

        assert!(approx_eq!(f64, predicted_output_0, BLUE, epsilon = 0.025));
        assert!(approx_eq!(f64, predicted_output_1, ORANGE, epsilon = 0.025));

        assert!(approx_eq!(f64, cost_of_predicted_0, 0.0, epsilon = 0.0001));
        assert!(approx_eq!(f64, cost_of_predicted_1, 0.0, epsilon = 0.0001));
    }

    #[test]
    fn test_nn_using_more_hidden_layers_with_more_neurons() {
        time_test!();

        // try more layers and more neurons in the hidden layers to see if I can improve number of epocs
        let training_data = get_data_set_1();

        // 2 x 16 x 16 x 1
        let mut nn = SimpleNeuralNetwork::new(vec![2, 16, 16, 1]);

        println!("initial weights:");
        for w in nn.weights.iter() {
            println!("{}", w);
        }

        println!("initial biases:");
        for b in nn.biases.iter() {
            println!("{}", b);
        }

        // need a matrix with rows = size of input x num data points
        let mut training_inputs = ColumnsMatrixBuilder::new();
        let mut expected_outputs = ColumnsMatrixBuilder::new();

        for i_data in 0..training_data.len() {
            training_inputs.push_column(&training_data[i_data].independant);
            expected_outputs.push_column(&[training_data[i_data].dependant]);
        }

        let training_inputs = training_inputs.build();
        let expected_outputs = expected_outputs.build();

        let epocs = 2500;
        let learning_rate = 2.0;

        nn.train(&training_inputs, &expected_outputs, epocs, learning_rate);

        // predict
        let prediction_input = column_vector![2.0, 2.0];
        let expected_output = column_vector![1.0];
        let predicted_output_0 = nn.feed_forward(&prediction_input).into_value();
        println!("predicted_output_0: {}", &predicted_output_0);
        let cost_of_predicted_0 =
            nn.cost_for_single_training_example(&prediction_input, &expected_output);
        println!("cost_of_predicted_0: \n{}", &cost_of_predicted_0);

        // predict
        println!("second prediction");
        let prediction_input = column_vector![-2.0, -2.0];
        let expected_output = column_vector![0.0];
        let predicted_output_1 = nn.feed_forward(&prediction_input).into_value();
        println!("predicted_output_1: {}", &predicted_output_1);
        let cost_of_predicted_1 =
            nn.cost_for_single_training_example(&prediction_input, &expected_output);
        println!("cost_of_predicted_1: \n{}", &cost_of_predicted_1);

        assert!(approx_eq!(f64, predicted_output_0, BLUE, epsilon = 0.01));
        assert!(approx_eq!(f64, predicted_output_1, ORANGE, epsilon = 0.01));

        assert!(approx_eq!(f64, cost_of_predicted_0, 0.0, epsilon = 0.0001));
        assert!(approx_eq!(f64, cost_of_predicted_1, 0.0, epsilon = 0.0001));
    }
}
