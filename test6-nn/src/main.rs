use std::collections::HashMap;

use common::datapoints::NDTrainingDataPoint;
use common::linalg::{euclidian_distance, euclidian_length, square};
use common::linalg::{ColumnVector, Matrix, MatrixShape, RowsMatrixBuilder};
use common::sigmoid::{sigmoid_prime_vector, sigmoid_vector};
use common::{column_vec_of_random_values_from_distribution, column_vector};

type LayerIndex = usize;

struct FeedForwardIntermediates {
    z_v: ColumnVector,
    activation_v: ColumnVector,
}

impl FeedForwardIntermediates {
    fn new_from(
        maybe_z_v: Option<&ColumnVector>,
        activation_v: &ColumnVector,
    ) -> FeedForwardIntermediates {
        let z_v = match maybe_z_v {
            Some(z_v) => z_v.clone(),
            None => ColumnVector::fill_new(f64::NAN, activation_v.num_elements()),
        };

        FeedForwardIntermediates {
            z_v,
            activation_v: activation_v.clone(),
        }
    }
}

pub struct CheckOptions {
    gradient_checking: bool,
    cost_decreasing_check: bool,
}

impl CheckOptions {
    pub fn no_checks() -> Self {
        Self {
            gradient_checking: false,
            cost_decreasing_check: false,
        }
    }

    pub fn all_checks() -> Self {
        Self {
            gradient_checking: true,
            cost_decreasing_check: true,
        }
    }
}

const GRADIENT_CHECK_EPSILON: f64 = 0.0001; // recommended value from Andrew Ng
const GRADIENT_CHECK_TWICE_EPSILON: f64 = 2.0 * GRADIENT_CHECK_EPSILON;
const GRADIENT_CHECK_EPSILON_SQUARED: f64 = GRADIENT_CHECK_EPSILON * GRADIENT_CHECK_EPSILON;

fn z(weight_matrix: &Matrix, bias_v: &ColumnVector, input_v: &ColumnVector) -> ColumnVector {
    weight_matrix.mult_vector(input_v).add(bias_v)
}

pub struct SimpleNeuralNetwork {
    sizes: Vec<LayerIndex>,

    // A HashMap of the weights keyed by the layer index.
    // The dimensions will be [rows x columns] [# neurons in the previous layer x # number neurons in the next layer]
    weights: HashMap<LayerIndex, Matrix>,

    // A HashMap of the biases keyed by the layer index.
    // The dimension of each ColumnVector will be [# number neurons in the layer]
    biases: HashMap<LayerIndex, ColumnVector>,
}

// Note that 3B1B does not do the divide by 2 and he ends up with a 2 in the derivitive function.
// Neilson does the divide by 2
// I'm doing the divide by 2
fn quadratic_cost(desired_v: &ColumnVector, actual_v: &ColumnVector) -> f64 {
    if desired_v.num_elements() != actual_v.num_elements() {
        panic!("expected and actual outputs must have the same length");
    }

    desired_v
        .iter_with(actual_v)
        .map(|(exp, act)| exp - act)
        .map(square)
        .sum::<f64>()
        / 2.0
}

impl SimpleNeuralNetwork {
    pub fn new(sizes: Vec<usize>) -> Self {
        let mut biases = HashMap::new();
        let mut weights = HashMap::new();

        for l in 1..sizes.len() {
            let biases_column_vector =
                column_vec_of_random_values_from_distribution(0.0, 1.0, sizes[l]);
            biases.insert(l, biases_column_vector);

            let weights_matrix = Matrix::new_matrix_with_random_values_from_normal_distribution(
                sizes[l],
                sizes[l - 1],
                0.0,
                1.0,
            );
            weights.insert(l, weights_matrix);
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

    pub fn get_weight_matrix_shape(&self, layer_index: LayerIndex) -> MatrixShape {
        if layer_index == 0 {
            panic!("not valid for input layer (because it has no weights/biases");
        }
        MatrixShape::new(self.sizes[layer_index], self.sizes[layer_index - 1])
    }

    pub fn feed_forward(&self, input_activations: &ColumnVector) -> ColumnVector {
        let mut activation_v = input_activations.clone();

        for l in 1..self.sizes.len() {
            let z_v = z(
                self.weights.get(&l).unwrap(),
                self.biases.get(&l).unwrap(),
                &activation_v,
            );
            activation_v = sigmoid_vector(&z_v);
        }

        activation_v
    }

    /// Feed forward capturing the intermediate z vectors and activation vectors.
    /// # Arguments
    /// * `input_activations` - The input activations
    /// # Returns
    /// A HashMap<LayerIndex, FeedForwardIntermediateValues> with an entry for each layer. For the input layer, only the activation_v is populated since z_v doesn't make sense for it.
    fn feed_forward_capturing(
        &self,
        input_activations: &ColumnVector,
    ) -> HashMap<LayerIndex, FeedForwardIntermediates> {
        let mut intermediates = HashMap::new();
        let mut activation_v = input_activations.clone();

        for l in 0..self.num_layers() {
            if l == 0 {
                intermediates.insert(l, FeedForwardIntermediates::new_from(None, &activation_v));
            } else {
                let z_v = z(
                    self.weights.get(&l).unwrap(),
                    self.biases.get(&l).unwrap(),
                    &activation_v,
                );
                activation_v = sigmoid_vector(&z_v);
                intermediates.insert(
                    l,
                    FeedForwardIntermediates::new_from(Some(&z_v), &activation_v),
                );
            }
        }

        intermediates
    }

    pub fn cost_single_tr_ex(&self, tr_ex: &NDTrainingDataPoint) -> f64 {
        if tr_ex.input_v.num_elements() != self.sizes[0] {
            panic!(
                "input_v must have the same number of elements as the number of neurons in the input layer"
            );
        }

        let outputs = self.feed_forward(&tr_ex.input_v);

        if outputs.num_elements() != tr_ex.desired_output_v.num_elements() {
            panic!("input_v and desired_output_v must have the same length");
        }

        quadratic_cost(&tr_ex.desired_output_v, &outputs)
    }

    pub fn cost_training_set(&self, training_data: &Vec<NDTrainingDataPoint>) -> f64 {
        training_data
            .iter()
            .map(|tr_ex| self.cost_single_tr_ex(tr_ex))
            .sum::<f64>()
            / training_data.len() as f64
    }

    // returning a Vec<f64> instead of a ColumnVector here because I don't think we'll do any math with it in its raw form.
    // So using it as a Vec will probably give me maximum flexibility.
    fn unroll_weights_and_biases(&self) -> Vec<f64> {
        let mut unrolled_vec = Vec::new();
        for l in 1..self.num_layers() {
            unrolled_vec.extend_from_slice(&self.weights.get(&l).unwrap().data);
            unrolled_vec.extend_from_slice(&self.biases.get(&l).unwrap().get_data_as_slice());
        }
        unrolled_vec
    }

    fn unroll_gradients(
        &self,
        gradients: &HashMap<LayerIndex, (Matrix, ColumnVector)>,
    ) -> Vec<f64> {
        let mut unrolled_vec = Vec::new();
        for l in 1..self.num_layers() {
            let this_layer_gradients = gradients.get(&l).unwrap();
            unrolled_vec.extend_from_slice(this_layer_gradients.0.data.as_slice()); // .0 is the weights matrix
            unrolled_vec.extend_from_slice(this_layer_gradients.1.get_data_as_slice());
            // .0 is the weights matrix
        }
        unrolled_vec
    }

    /// reshape theta_v in accordance with the sizes of the layers
    fn reshape_weights_and_biases(&self, big_theta_v: &[f64]) -> SimpleNeuralNetwork {
        // we know the number of layers and the size of each one
        // so we know what size of weights and biases we need
        // just need to pull things out correctly form big_theta_v.

        let mut weights = HashMap::new();
        let mut biases = HashMap::new();

        let mut ptr: usize = 0;

        for l in 1..self.num_layers() {
            let w_shape = self.get_weight_matrix_shape(l);
            let w_data = &big_theta_v[ptr..(ptr + w_shape.data_length())];
            let w = Matrix::new_with_shape_and_values(&w_shape, w_data);
            ptr += w_shape.data_length();
            let b = ColumnVector::new(&big_theta_v[ptr..(ptr + self.sizes[l])]);
            ptr += self.sizes[l];

            weights.insert(l, w);
            biases.insert(l, b);
        }

        SimpleNeuralNetwork {
            sizes: self.sizes.clone(),
            weights,
            biases,
        }
    }

    /// Used for gradient checking
    fn approximate_cost_gradient(&self, training_data: &Vec<NDTrainingDataPoint>) -> Vec<f64> {
        let mut big_theta_v = self.unroll_weights_and_biases();
        let mut gradient = Vec::new();

        for i in 0..big_theta_v.len() {
            let orig_i_value = big_theta_v[i];

            big_theta_v[i] = orig_i_value + GRADIENT_CHECK_EPSILON;
            let temp_nn = self.reshape_weights_and_biases(&big_theta_v);
            let cost_plus_epsilon = temp_nn.cost_training_set(&training_data);

            big_theta_v[i] = orig_i_value - GRADIENT_CHECK_EPSILON;
            let temp_nn = self.reshape_weights_and_biases(&big_theta_v);
            let cost_minus_epsilon = temp_nn.cost_training_set(training_data);

            big_theta_v[i] = orig_i_value; // important - restore the orig value

            let approx_cost_fn_partial_derivative =
                (cost_plus_epsilon - cost_minus_epsilon) / GRADIENT_CHECK_TWICE_EPSILON;
            gradient.push(approx_cost_fn_partial_derivative);
        }
        gradient
    }

    // Backprop Equation (the one that is unlabeled but follows after BP1a. I assume then ment to label it BP1b)
    // from the Neilson book
    fn err_last_layer(
        &self,
        output_activation_v: &ColumnVector,
        desired_output_v: &ColumnVector,
        output_layer_z_v: &ColumnVector,
    ) -> ColumnVector {
        output_activation_v
            .minus(desired_output_v)
            .hadamard_product_chaining(&sigmoid_prime_vector(output_layer_z_v))
    }

    // Backprop Equation BP2 from the Neilson book
    fn err_non_last_layer(
        &self,
        layer: LayerIndex,
        plus_one_layer_error_v: &ColumnVector,
        this_layer_z_v: &ColumnVector,
    ) -> ColumnVector {
        let weight_matrix = self.weights.get(&(layer + 1)).unwrap();

        weight_matrix
            .transpose()
            .mult_vector(plus_one_layer_error_v)
            .hadamard_product_chaining(&sigmoid_prime_vector(this_layer_z_v))
    }

    /// Returns a Vec of column vectors representing the errors at each neuron at each layer from L-1 to 1
    /// where layer L-1 is the output layer and layer 0 is the input layer.
    /// Because we are going backwards and because we're only doing down to laye 1, I'm using a HashMap to keep track of the error vectors
    /// and the layers they correspond to to reduce confusion.
    fn backprop(
        &self,
        desired_output_v: &ColumnVector,
        intermediates: &HashMap<LayerIndex, FeedForwardIntermediates>,
    ) -> HashMap<LayerIndex, ColumnVector> {
        // loop through the layers from back to front, and compute the error at each one.
        // Create a column vector representing the errors at each layer

        let mut error_vectors = HashMap::new();

        let last_layer_index = self.num_layers() - 1;

        for l in (1..self.num_layers()).rev() {
            let z_v = &intermediates.get(&l).unwrap().z_v;

            let err_v = if l == last_layer_index {
                let activations_v = &intermediates[&last_layer_index].activation_v;
                self.err_last_layer(&activations_v, desired_output_v, &z_v)
            } else {
                let error_vector_for_plus_one_layer = error_vectors.get(&(l + 1)).unwrap();
                println!("in backprop, l = {}", l);
                self.err_non_last_layer(l, error_vector_for_plus_one_layer, z_v)
            };

            error_vectors.insert(l, err_v);
        }

        error_vectors
    }

    fn compute_gradients(
        &mut self,
        per_tr_ex_data: &Vec<(
            HashMap<usize, FeedForwardIntermediates>,
            HashMap<usize, ColumnVector>,
        )>, // outer Vec is per training example
    ) -> HashMap<LayerIndex, (Matrix, ColumnVector)> {
        let mut gradients = HashMap::new();
        let num_training_examples = per_tr_ex_data.len();

        for l in (1..self.num_layers()).rev() {
            let mut weights_partials_matrix_avg =
                Matrix::new_zero_matrix(self.sizes[l], self.sizes[l - 1]);
            let mut bias_partials_vector_avg = ColumnVector::new_zero_vector(self.sizes[l]);

            per_tr_ex_data
                .iter()
                .for_each(|(intermediates, error_vectors)| {
                    let prev_layer_activations_v =
                        &intermediates.get(&(l - 1)).unwrap().activation_v;

                    let this_layer_err_v = error_vectors.get(&l).unwrap();

                    let prev_layer_activations_v_transpose = prev_layer_activations_v.transpose();

                    let weights_grad =
                        this_layer_err_v.mult_matrix(&prev_layer_activations_v_transpose);

                    weights_partials_matrix_avg.add_in_place(&weights_grad);
                    bias_partials_vector_avg.plus_in_place(this_layer_err_v);
                });

            weights_partials_matrix_avg.divide_by_scalar_in_place(num_training_examples as f64);
            bias_partials_vector_avg.divide_by_scalar_in_place(num_training_examples as f64);

            gradients.insert(l, (weights_partials_matrix_avg, bias_partials_vector_avg));
        }

        gradients
    }

    pub fn train(
        &mut self,
        training_data: &Vec<NDTrainingDataPoint>,
        epocs: usize,
        learning_rate: f64,
        check_options: Option<CheckOptions>,
    ) {
        let initial_cost = self.cost_training_set(&training_data);
        println!("initial cost across entire training set: {}", initial_cost);

        // here's what this does:
        // for epocs
        //     for each training example
        //         feed forward, capturing intermediates
        //         backpropagate to compute errors at each neuron of each layer
        //     compute gradients for w and b
        //     update weights and biases

        let mut epocs_count = 0;
        let mut prev_cost = initial_cost;

        let check_options = check_options.unwrap_or(CheckOptions::no_checks());

        loop {
            if epocs_count >= epocs {
                println!("stopping after {} epocs", epocs_count);
                break;
            }

            let mut per_tr_ex_data = Vec::new();

            training_data.iter().for_each(|tr_ex| {
                let intermediates = self.feed_forward_capturing(&tr_ex.input_v);
                let errors = self.backprop(&tr_ex.desired_output_v, &intermediates);
                per_tr_ex_data.push((intermediates, errors));
            });

            // note: compute_gradients takes data for ALL training examples
            let mut gradients = self.compute_gradients(&per_tr_ex_data);

            if check_options.gradient_checking {
                let approx_gradients_big_v = self.approximate_cost_gradient(training_data);
                // unroll the actual gradients
                let d_vec = self.unroll_gradients(&gradients);

                let ed = euclidian_distance(&approx_gradients_big_v, &d_vec);
                println!("ed: {}", ed);

                if ed > GRADIENT_CHECK_EPSILON_SQUARED {
                    panic!("failed gradient check");
                }

                let normalized_distance = euclidian_distance(&approx_gradients_big_v, &d_vec)
                    / (euclidian_length(&approx_gradients_big_v) + euclidian_length(&d_vec));

                if normalized_distance > GRADIENT_CHECK_EPSILON_SQUARED {
                    panic!("failed gradient check");
                }
            }

            // update the weights and biases
            // TODO: extract to method for easy testing
            gradients
                .iter_mut()
                .for_each(|(layer_index, (weights_grad, bias_grad))| {
                    let layer_index = *layer_index;

                    weights_grad.multiply_by_scalar_in_place(learning_rate);
                    bias_grad.multiply_by_scalar_in_place(learning_rate);

                    let weights = self.weights.get_mut(&layer_index).unwrap();
                    let biases = self.biases.get_mut(&layer_index).unwrap();

                    weights.subtract_in_place(&weights_grad);
                    biases.minus_in_place(&bias_grad);
                });

            if check_options.cost_decreasing_check {
                let cost = self.cost_training_set(&training_data);
                if cost > prev_cost {
                    panic!(
                        "cost increased from {} to {} on epoc {}",
                        prev_cost, cost, epocs_count
                    );
                }
                prev_cost = cost;
            }

            epocs_count += 1;

            // TODO: stop on convergence
        }

        let final_cost = self.cost_training_set(&training_data);
        println!(
            "\ncost across entire training set after {} epocs: {}",
            epocs, final_cost,
        );
    }
}

fn main() {
    // 3x2 network
    // single weights / biases
    // weights matrix -> 2x3

    let inputs = column_vector![0.0, 0.5, 1.0];

    let weight_m_layer_1 = RowsMatrixBuilder::new()
        .with_row(&[0.5, 0.5, 0.5])
        .with_row(&[1.0, 1.0, 1.0])
        .build();

    let bias_v_layer_1 = column_vector![0.1, 0.1];

    let mut weights = HashMap::new();
    weights.insert(1, weight_m_layer_1);

    let mut biases = HashMap::new();
    biases.insert(1, bias_v_layer_1);

    let nn = SimpleNeuralNetwork {
        sizes: vec![3, 2],
        weights,
        biases,
    };

    let outputs = nn.feed_forward(&inputs);

    println!("outputs: {:?}", outputs);
}

#[cfg(test)]
mod tests {
    use super::*;
    use common::scalar_valued_multivariable_point::ScalarValuedMultivariablePoint;
    use float_cmp::approx_eq;
    use time_test::time_test;

    pub fn get_simple_two_layer_nn_for_test() -> SimpleNeuralNetwork {
        let num_neurons_layer_0 = 3;
        let num_neurons_layer_1 = 2;

        // Layer 0 (input): 3 neurons
        // Layer 1 (output): 2 neurons
        // weights matrix -> 2x3

        let weight_m_layer_1 = RowsMatrixBuilder::new()
            .with_row(&[0.5, 0.5, 0.5])
            .with_row(&[1.0, 1.0, 1.0])
            .build();

        let bias_v_layer_1 = column_vector![0.1, 0.1];

        let mut weights = HashMap::new();
        weights.insert(1, weight_m_layer_1);

        let mut biases = HashMap::new();
        biases.insert(1, bias_v_layer_1);

        let nn = SimpleNeuralNetwork {
            sizes: vec![num_neurons_layer_0, num_neurons_layer_1],
            weights,
            biases,
        };

        nn
    }

    pub fn get_simple_get_2_3_1_nn_for_test() -> SimpleNeuralNetwork {
        let num_neurons_layer_0 = 2;
        let num_neurons_layer_1 = 3;
        let num_neurons_layer_2 = 1;

        // 3x2
        let weights_l1 = RowsMatrixBuilder::new()
            .with_row(&[0.2, 0.2])
            .with_row(&[0.4, 0.4])
            .with_row(&[0.6, 0.6])
            .build();

        // 1x3
        let weights_l2 = RowsMatrixBuilder::new().with_row(&[0.5, 0.5, 0.5]).build();

        let biases_l1 = column_vector![0.1, 0.1, 0.1];
        let biases_l2 = column_vector![0.1];

        let mut weights = HashMap::new();
        weights.insert(1, weights_l1);
        weights.insert(2, weights_l2);

        let mut biases = HashMap::new();
        biases.insert(1, biases_l1);
        biases.insert(2, biases_l2);

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

        let weights_l1 = RowsMatrixBuilder::new()
            .with_row(&[0.2, 0.2, 0.2])
            .with_row(&[0.4, 0.4, 0.4])
            .with_row(&[0.6, 0.6, 0.6])
            .with_row(&[0.8, 0.8, 0.8])
            .with_row(&[1.0, 1.0, 1.0])
            .build();

        let weights_l2 = RowsMatrixBuilder::new()
            .with_row(&[0.5, 0.5, 0.5, 0.5, 0.5])
            .with_row(&[0.6, 0.6, 0.6, 0.6, 0.6])
            .build();

        let biases_l1 = column_vector![0.1, 0.1, 0.1, 0.1, 0.1];
        let biases_l2 = column_vector![0.1, 0.1];

        let mut weights = HashMap::new();
        weights.insert(1, weights_l1);
        weights.insert(2, weights_l2);

        let mut biases = HashMap::new();
        biases.insert(1, biases_l1);
        biases.insert(2, biases_l2);

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

        let bias_v = column_vector![0.5, 0.5];
        let input_v = column_vector![1.0, 2.0, 3.0];

        let weighted_sum_v = z(&weights, &bias_v, &input_v);

        assert_eq!(weighted_sum_v.num_elements(), 2);
        assert_eq!(weighted_sum_v.get(0), 14.5);
        assert_eq!(weighted_sum_v.get(1), 32.5);
    }

    #[test]
    fn feed_forward_works_simple_two_layer() {
        let num_neurons_layer_0 = 3;
        let num_neurons_layer_1 = 2;

        // Layer N: 3 neurons
        // L N-1: 2 neurons
        // weights matrix -> 2x3

        let input_v = column_vector![0.0, 0.5, 1.0];

        let weight_m_l1 = RowsMatrixBuilder::new()
            .with_row(&[0.5, 0.5, 0.5])
            .with_row(&[1.0, 1.0, 1.0])
            .build();

        let biase_v_l1 = column_vector![0.1, 0.1];

        let mut weights = HashMap::new();
        weights.insert(1, weight_m_l1);

        let mut biases = HashMap::new();
        biases.insert(1, biase_v_l1);

        let nn = SimpleNeuralNetwork {
            sizes: vec![num_neurons_layer_0, num_neurons_layer_1],
            weights,
            biases,
        };

        let outputs = nn.feed_forward(&input_v);
        assert_eq!(outputs.num_elements(), 2);
        assert_eq!(outputs.get(0), 0.7005671424739729);
        assert_eq!(outputs.get(1), 0.8320183851339245);

        // now see if I can do this from the unrolled big_theta_v
        let big_theta_v = nn.unroll_weights_and_biases();
        let reshaped = nn.reshape_weights_and_biases(&big_theta_v);
        let ff_fn_output_v = reshaped.feed_forward(&input_v);
        assert_eq!(ff_fn_output_v.num_elements(), 2);
        assert_eq!(ff_fn_output_v.get(0), 0.7005671424739729);
        assert_eq!(ff_fn_output_v.get(1), 0.8320183851339245);
    }

    #[test]
    fn feed_forward_works_simple_three_layer() {
        let num_neurons_layer_0 = 3;
        let num_neurons_layer_1 = 5;
        let num_neurons_layer_2 = 2;

        // W0: 5x3
        // W1: 2x5

        let inputs = column_vector![0.0, 0.5, 1.0];

        let weights_l1 = RowsMatrixBuilder::new()
            .with_row(&[0.2, 0.2, 0.2])
            .with_row(&[0.4, 0.4, 0.4])
            .with_row(&[0.6, 0.6, 0.6])
            .with_row(&[0.8, 0.8, 0.8])
            .with_row(&[1.0, 1.0, 1.0])
            .build();

        let weights_l2 = RowsMatrixBuilder::new()
            .with_row(&[0.5, 0.5, 0.5, 0.5, 0.5])
            .with_row(&[0.6, 0.6, 0.6, 0.6, 0.6])
            .build();

        let bias_v_l1 = column_vector![0.1, 0.1, 0.1, 0.1, 0.1];
        let bias_v_l2 = column_vector![0.1, 0.1];

        // manually compute the correct output to use in later assertion
        let z0 = weights_l1.mult_vector(&inputs).plus(&bias_v_l1);
        let sz0 = sigmoid_vector(&z0);
        println!("sz0: {:?}", sz0);

        let z1 = weights_l2.mult_vector(&sz0).plus(&bias_v_l2);
        let sz1 = sigmoid_vector(&z1);
        println!("sz1: {:?}", sz1);

        println!("\n now doing with feed_forward");

        let mut weights = HashMap::new();
        weights.insert(1, weights_l1);
        weights.insert(2, weights_l2);

        let mut biases = HashMap::new();
        biases.insert(1, bias_v_l1);
        biases.insert(2, bias_v_l2);

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

        let weights_l1 = RowsMatrixBuilder::new()
            .with_row(&[0.2, 0.2, 0.2])
            .with_row(&[0.4, 0.4, 0.4])
            .with_row(&[0.6, 0.6, 0.6])
            .with_row(&[0.8, 0.8, 0.8])
            .with_row(&[1.0, 1.0, 1.0])
            .build();

        let weights_l2 = RowsMatrixBuilder::new()
            .with_row(&[0.5, 0.5, 0.5, 0.5, 0.5])
            .with_row(&[0.6, 0.6, 0.6, 0.6, 0.6])
            .build();

        let bias_v_l1 = column_vector![0.1, 0.1, 0.1, 0.1, 0.1];
        let bias_v_l2 = column_vector![0.1, 0.1];

        // manually compute the correct output to use in later assertion
        let z0 = weights_l1.mult_vector(&inputs).plus(&bias_v_l1);
        let sz0 = sigmoid_vector(&z0);
        println!("sz0: {:?}", sz0);

        let z1 = weights_l2.mult_vector(&sz0).plus(&bias_v_l2);
        let sz1 = sigmoid_vector(&z1);
        println!("sz1: {:?}", sz1);

        println!("\n now doing with feed_forward");

        let mut weights = HashMap::new();
        weights.insert(1, weights_l1);
        weights.insert(2, weights_l2);

        let mut biases = HashMap::new();
        biases.insert(1, bias_v_l1);
        biases.insert(2, bias_v_l2);

        let nn = SimpleNeuralNetwork {
            sizes: vec![
                num_neurons_layer_0,
                num_neurons_layer_1,
                num_neurons_layer_2,
            ],
            weights,
            biases,
        };

        let intermediates = nn.feed_forward_capturing(&inputs);
        assert_eq!(intermediates.len(), 3);
        let final_step_values = intermediates.get(&2).unwrap();
        let outputs = final_step_values.activation_v.clone();

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
    pub fn test_cost_single_tr_ex_single_output_neuron() {
        let nn = get_simple_get_2_3_1_nn_for_test();

        // let's go for y(x0, x1) = x0 + x1;
        let tr_ex = NDTrainingDataPoint {
            input_v: column_vector![1.0, 1.0],
            desired_output_v: column_vector![3.0],
        };

        let c0 = nn.cost_single_tr_ex(&tr_ex);

        // manually compute the expected cost for the single neuron in the last layer
        let a_output = nn.feed_forward(&tr_ex.input_v);
        let a_output_value = a_output.into_value();

        let manual_cost = (3.0 - a_output_value).powi(2) / 2.0;

        println!("a_output: \n{}", a_output_value);
        println!("manual_cost: \n{}", manual_cost);

        assert_eq!(c0, manual_cost);
    }

    #[test]
    pub fn test_cost_single_tr_ex_multiple_output_neurons() {
        let nn = get_three_layer_multiple_output_nn_for_test();

        let tr_ex = NDTrainingDataPoint {
            input_v: column_vector![0.0, 0.5, 1.0],
            desired_output_v: column_vector![2.0, 2.0],
        };

        let c0 = nn.cost_single_tr_ex(&tr_ex);

        // manually compute the expected cost for the single neuron in the last layer
        // this uses a different method to compute the cost, but it is mathematically equivalent to that used
        // in cost_single_tr_ex
        let actual_output_vector = nn.feed_forward(&tr_ex.input_v);
        let diff_vec = &tr_ex.desired_output_v.minus(&actual_output_vector);
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

        let mut training_data = Vec::new();

        training_data.push(NDTrainingDataPoint {
            input_v: column_vector![0.0, 0.5, 0.9],
            desired_output_v: column_vector![2.0, 2.0],
        });
        training_data.push(NDTrainingDataPoint {
            input_v: column_vector![0.0, 0.5, 1.0],
            desired_output_v: column_vector![2.0, 2.0],
        });
        training_data.push(NDTrainingDataPoint {
            input_v: column_vector![0.0, 0.5, 1.1],
            desired_output_v: column_vector![2.0, 2.0],
        });

        let c = nn.cost_training_set(&training_data);
        let c0 = nn.cost_single_tr_ex(&training_data[0]);
        let c1 = nn.cost_single_tr_ex(&training_data[1]);
        let c2 = nn.cost_single_tr_ex(&training_data[2]);
        let c_avg = (c0 + c1 + c2) / 3.0;
        assert_eq!(c, c_avg);

        // now do the same cost test with big_theta_v
        let big_theta_v = nn.unroll_weights_and_biases();
        let reshaped = nn.reshape_weights_and_biases(&big_theta_v);
        let cost_big_theta_v = reshaped.cost_training_set(&training_data);
        assert_eq!(cost_big_theta_v, c);

        // from here on, I'm testing the towards_backprop stuff
        println!("\nIn the test - doing the pre backprop stuff: {}", c0);
        let mut error_vectors_for_each_training_example = Vec::new();
        let mut intermediates_for_each_training_example = Vec::new();

        for i_tr_ex in 0..training_data.len() {
            println!("\nfor the {}th training example", i_tr_ex);
            let inputs_v = &training_data[i_tr_ex].input_v;
            let desired_output_v = &training_data[i_tr_ex].desired_output_v;
            let intermediates = nn.feed_forward_capturing(&inputs_v);
            assert_eq!(intermediates.len(), nn.num_layers());

            println!("for input {}", i_tr_ex);
            println!("intermediates");
            for i_intermediate in 0..intermediates.len() {
                println!("intermediate {}:", i_intermediate);
                println!(" - z_vector:");
                println!("{}", intermediates.get(&i_intermediate).unwrap().z_v);
                println!(" - activations_vector:");
                println!(
                    "{}",
                    intermediates.get(&i_intermediate).unwrap().activation_v
                );
            }
            let err_vectors_this_tr_ex = nn.backprop(&desired_output_v, &intermediates);
            error_vectors_for_each_training_example.push(err_vectors_this_tr_ex);
            intermediates_for_each_training_example.push(intermediates);
        }
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

    fn get_data_set_1b() -> Vec<NDTrainingDataPoint> {
        // fake data roughly based on https://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=gauss&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=3,1&seed=0.22934&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false
        let training_data = vec![
            NDTrainingDataPoint::new(column_vector![-2.0, -2.0], column_vector![ORANGE]),
            NDTrainingDataPoint::new(column_vector![-2.0, -2.0], column_vector![ORANGE]),
            NDTrainingDataPoint::new(column_vector![-2.0, -2.0], column_vector![ORANGE]),
            NDTrainingDataPoint::new(column_vector![-2.0, -2.0], column_vector![ORANGE]),
            NDTrainingDataPoint::new(column_vector![-2.0, -2.0], column_vector![ORANGE]),
            NDTrainingDataPoint::new(column_vector![-2.0, -2.0], column_vector![ORANGE]),
            NDTrainingDataPoint::new(column_vector![-2.0, -2.0], column_vector![ORANGE]),
            NDTrainingDataPoint::new(column_vector![-2.0, -2.0], column_vector![ORANGE]),
            NDTrainingDataPoint::new(column_vector![-2.0, -2.0], column_vector![ORANGE]),
            NDTrainingDataPoint::new(column_vector![-2.0, -2.0], column_vector![ORANGE]),
            NDTrainingDataPoint::new(column_vector![2.0, 2.0], column_vector![BLUE]),
            NDTrainingDataPoint::new(column_vector![2.0, 2.0], column_vector![BLUE]),
            NDTrainingDataPoint::new(column_vector![2.0, 2.0], column_vector![BLUE]),
            NDTrainingDataPoint::new(column_vector![2.0, 2.0], column_vector![BLUE]),
            NDTrainingDataPoint::new(column_vector![2.0, 2.0], column_vector![BLUE]),
            NDTrainingDataPoint::new(column_vector![2.0, 2.0], column_vector![BLUE]),
            NDTrainingDataPoint::new(column_vector![2.0, 2.0], column_vector![BLUE]),
            NDTrainingDataPoint::new(column_vector![2.0, 2.0], column_vector![BLUE]),
            NDTrainingDataPoint::new(column_vector![2.0, 2.0], column_vector![BLUE]),
            NDTrainingDataPoint::new(column_vector![2.0, 2.0], column_vector![BLUE]),
        ];
        training_data
    }

    #[test]
    fn test_nn() {
        time_test!();
        let training_data = get_data_set_1b();

        // 2 x 3 x 1
        let num_neurons_layer_0 = 2;
        let num_neurons_layer_1 = 3;
        let num_neurons_layer_2 = 1;

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

        let mut weights = HashMap::new();
        weights.insert(1, weights_l1);
        weights.insert(2, weights_l2);

        let mut biases = HashMap::new();
        biases.insert(1, bias_v_l1);
        biases.insert(2, bias_v_l2);

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

        let epocs = 2000;
        let learning_rate = 0.9;

        nn.train(
            &training_data,
            epocs,
            learning_rate,
            Some(CheckOptions::all_checks()),
        );

        // predict
        let tr_ex = NDTrainingDataPoint {
            input_v: column_vector![2.0, 2.0],
            desired_output_v: column_vector![1.0],
        };

        let predicted_output_0 = nn.feed_forward(&tr_ex.input_v).into_value();
        println!("predicted_output_0: {}", &predicted_output_0);
        let cost_of_predicted_0 = nn.cost_single_tr_ex(&tr_ex);
        println!("cost_of_predicted_0: \n{}", &cost_of_predicted_0);

        // predict
        let tr_ex = NDTrainingDataPoint {
            input_v: column_vector![-2.0, -2.0],
            desired_output_v: column_vector![0.0],
        };

        let predicted_output_1 = nn.feed_forward(&tr_ex.input_v).into_value();
        println!("predicted_output_1: {}", &predicted_output_1);
        let cost_of_predicted_1 = nn.cost_single_tr_ex(&tr_ex);
        println!("cost_of_predicted_1: \n{}", &cost_of_predicted_1);

        assert!(approx_eq!(f64, predicted_output_0, BLUE, epsilon = 0.05));
        assert!(approx_eq!(f64, predicted_output_1, ORANGE, epsilon = 0.05));

        assert!(approx_eq!(f64, cost_of_predicted_0, 0.0, epsilon = 0.001));
        assert!(approx_eq!(f64, cost_of_predicted_1, 0.0, epsilon = 0.001));
    }

    #[test]
    fn test_nn_using_constructor_for_random_initial_weights_and_biases() {
        time_test!();
        // try the same data set as before but use the NN constructor to initialize with random weights/biases
        let training_data = get_data_set_1b();

        // 2 x 3 x 1
        let mut nn = SimpleNeuralNetwork::new(vec![2, 3, 1]);

        println!("initial weights:");
        for (l, w) in nn.weights.iter() {
            println!("{}", w);
        }

        println!("initial biases:");
        for (l, b) in nn.biases.iter() {
            println!("{}", b);
        }

        let epocs = 7000;
        let learning_rate = 0.9;

        let check_options = CheckOptions {
            gradient_checking: false,
            cost_decreasing_check: true,
        };

        nn.train(&training_data, epocs, learning_rate, Some(check_options));

        // predict
        let tr_ex = NDTrainingDataPoint {
            input_v: column_vector![2.0, 2.0],
            desired_output_v: column_vector![1.0],
        };

        let predicted_output_0 = nn.feed_forward(&tr_ex.input_v).into_value();
        println!("predicted_output_0: {}", &predicted_output_0);
        let cost_of_predicted_0 = nn.cost_single_tr_ex(&tr_ex);
        println!("cost_of_predicted_0: \n{}", &cost_of_predicted_0);

        // predict
        println!("second prediction");
        let tr_ex = NDTrainingDataPoint {
            input_v: column_vector![-2.0, -2.0],
            desired_output_v: column_vector![0.0],
        };
        let predicted_output_1 = nn.feed_forward(&tr_ex.input_v).into_value();
        println!("predicted_output_1: {}", &predicted_output_1);
        let cost_of_predicted_1 = nn.cost_single_tr_ex(&tr_ex);
        println!("cost_of_predicted_1: \n{}", &cost_of_predicted_1);

        assert!(approx_eq!(f64, predicted_output_0, BLUE, epsilon = 0.025));
        assert!(approx_eq!(f64, predicted_output_1, ORANGE, epsilon = 0.025));

        assert!(approx_eq!(f64, cost_of_predicted_0, 0.0, epsilon = 0.0002));
        assert!(approx_eq!(f64, cost_of_predicted_1, 0.0, epsilon = 0.0002));
    }

    #[test]
    fn test_nn_using_more_hidden_layers_with_more_neurons() {
        time_test!();

        // try more layers and more neurons in the hidden layers to see if I can improve number of epocs
        let training_data = get_data_set_1b();

        // 2 x 16 x 16 x 1
        let mut nn = SimpleNeuralNetwork::new(vec![2, 16, 16, 1]);

        println!("initial weights:");
        nn.weights.values().for_each(|w| {
            println!("{}", w);
        });

        println!("initial biases:");
        nn.biases.values().for_each(|b| {
            println!("{}", b);
        });

        let epocs = 2500;
        let learning_rate = 0.9;

        let check_options = CheckOptions {
            gradient_checking: false,
            cost_decreasing_check: true,
        };

        nn.train(&training_data, epocs, learning_rate, Some(check_options));

        // predict
        let tr_ex = NDTrainingDataPoint {
            input_v: column_vector![2.0, 2.0],
            desired_output_v: column_vector![1.0],
        };
        let predicted_output_0 = nn.feed_forward(&tr_ex.input_v).into_value();
        println!("predicted_output_0: {}", &predicted_output_0);
        let cost_of_predicted_0 = nn.cost_single_tr_ex(&tr_ex);
        println!("cost_of_predicted_0: \n{}", &cost_of_predicted_0);

        // predict
        println!("second prediction");
        let tr_ex = NDTrainingDataPoint {
            input_v: column_vector![-2.0, -2.0],
            desired_output_v: column_vector![0.0],
        };
        let predicted_output_1 = nn.feed_forward(&tr_ex.input_v).into_value();
        println!("predicted_output_1: {}", &predicted_output_1);
        let cost_of_predicted_1 = nn.cost_single_tr_ex(&tr_ex);
        println!("cost_of_predicted_1: \n{}", &cost_of_predicted_1);

        assert!(approx_eq!(f64, predicted_output_0, BLUE, epsilon = 0.025));
        assert!(approx_eq!(f64, predicted_output_1, ORANGE, epsilon = 0.025));

        assert!(approx_eq!(f64, cost_of_predicted_0, 0.0, epsilon = 0.0001));
        assert!(approx_eq!(f64, cost_of_predicted_1, 0.0, epsilon = 0.0001));
    }

    #[test]
    fn test_get_weight_matrix_shape() {
        let nn = SimpleNeuralNetwork::new(vec![2, 3, 2]);
        let weight_matrix_l1_shape = nn.get_weight_matrix_shape(1);
        let weight_matrix_l2_shape = nn.get_weight_matrix_shape(2);
        assert_eq!(weight_matrix_l1_shape.rows, 3);
        assert_eq!(weight_matrix_l1_shape.columns, 2);
        assert_eq!(weight_matrix_l2_shape.rows, 2);
        assert_eq!(weight_matrix_l2_shape.columns, 3);
    }

    #[test]
    fn test_unroll_weights_and_biases() {
        let mut nn = SimpleNeuralNetwork::new(vec![2, 3, 2]);
        let weight_matrix_l1_shape = nn.get_weight_matrix_shape(1);
        let weight_matrix_l2_shape = nn.get_weight_matrix_shape(2);

        assert_eq!(weight_matrix_l1_shape, MatrixShape::new(3, 2));
        assert_eq!(weight_matrix_l2_shape, MatrixShape::new(2, 3));

        // the initial weights and biases are random
        // overwrite them with known values so we can test the unroll

        // input layer - no weights or biases

        // layer l = 1
        let w1 = nn.weights.get_mut(&1).unwrap();

        println!("weight_matrix_l1_shape: {:?}", weight_matrix_l1_shape);
        println!("w1 shape: {} x {}", w1.num_rows(), w1.num_columns());
        println!("w1: \n{}", w1);

        w1.set(0, 0, 1.0);
        w1.set(0, 1, 2.0);
        w1.set(1, 0, 3.0);
        w1.set(1, 1, 4.0);
        w1.set(2, 0, 5.0);
        w1.set(2, 1, 6.0);

        println!("w1: \n{}", w1);

        let b1 = nn.biases.get_mut(&1).unwrap();
        b1.set(0, 7.0);
        b1.set(1, 8.0);
        b1.set(2, 9.0);

        // layer l = 2 (output layer)
        let w2 = nn.weights.get_mut(&2).unwrap();
        let b2 = nn.biases.get_mut(&2).unwrap();

        w2.set(0, 0, 10.0);
        w2.set(0, 1, 11.0);
        w2.set(0, 2, 12.0);
        w2.set(1, 0, 13.0);
        w2.set(1, 1, 14.0);
        w2.set(1, 2, 15.0);

        b2.set(0, 16.0);
        b2.set(1, 17.0);

        let big_theta_v = nn.unroll_weights_and_biases();
        assert_eq!(big_theta_v.len(), 17);
        assert_eq!(
            big_theta_v,
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0, 17.0
            ]
        );
    }

    #[test]
    fn test_unroll_gradients() {
        let mut nn = SimpleNeuralNetwork::new(vec![2, 3, 2]);
        let weight_matrix_l1_shape = nn.get_weight_matrix_shape(1);
        let weight_matrix_l2_shape = nn.get_weight_matrix_shape(2);

        assert_eq!(weight_matrix_l1_shape, MatrixShape::new(3, 2));
        assert_eq!(weight_matrix_l2_shape, MatrixShape::new(2, 3));

        let mut gradients = HashMap::new();

        // for layer 1
        let mut w1 = Matrix::new_zero_matrix(3, 2);
        let mut b1 = ColumnVector::new_zero_vector(3);

        w1.set(0, 0, 0.1);
        w1.set(0, 1, 0.2);
        w1.set(1, 0, 0.3);
        w1.set(1, 1, 0.4);
        w1.set(2, 0, 0.5);
        w1.set(2, 1, 0.6);

        b1.set(0, 0.7);
        b1.set(1, 0.8);
        b1.set(2, 0.9);

        // for layer 2
        let mut w2 = Matrix::new_zero_matrix(2, 3);
        let mut b2 = ColumnVector::new_zero_vector(2);

        w2.set(0, 0, 0.1);
        w2.set(0, 1, 0.11);
        w2.set(0, 2, 0.12);
        w2.set(1, 0, 0.13);
        w2.set(1, 1, 0.14);
        w2.set(1, 2, 0.15);

        b2.set(0, 0.16);
        b2.set(1, 0.17);

        gradients.insert(1, (w1, b1));
        gradients.insert(2, (w2, b2));

        let big_d_vec = nn.unroll_gradients(&gradients);
        assert_eq!(big_d_vec.len(), 17);
        assert_eq!(
            big_d_vec,
            vec![
                0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15,
                0.16, 0.17
            ]
        );
    }

    #[test]
    fn test_reshape_weights_and_biases() {
        let nn = SimpleNeuralNetwork::new(vec![2, 3, 2]);

        let big_theta_v = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0,
        ];

        let temp_nn = nn.reshape_weights_and_biases(&big_theta_v);

        assert_eq!(temp_nn.num_layers(), 3);

        let w1 = temp_nn.weights.get(&1).unwrap();
        let b1 = temp_nn.biases.get(&1).unwrap();
        assert_eq!(w1.shape(), MatrixShape::new(3, 2));
        assert_eq!(w1.data.as_slice(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(b1.num_elements(), 3);
        assert_eq!(b1.get_data_as_slice(), &[7.0, 8.0, 9.0]);

        let w2 = temp_nn.weights.get(&2).unwrap();
        let b2 = temp_nn.biases.get(&2).unwrap();
        assert_eq!(w2.shape(), MatrixShape::new(2, 3));
        assert_eq!(w2.data.as_slice(), &[10.0, 11.0, 12.0, 13.0, 14.0, 15.0]);
        assert_eq!(b2.num_elements(), 2);
        assert_eq!(b2.get_data_as_slice(), &[16.0, 17.0]);
    }
}
