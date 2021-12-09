use common::column_vector;
use common::matrix::{ColumnsMatrixBuilder, Matrix, RowsMatrixBuilder};

use rand::distributions::{Distribution, Standard, Uniform};
use rand::Rng;
use rand_distr::Normal;

fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}

fn sigmoid_vector(z_vector: &Matrix) -> Matrix {
    if z_vector.columns != 1 {
        panic!("this function is only applicable to column vectors");
    }
    if z_vector.rows == 0 {
        panic!("this function is only valid for column vectors with at least one element (row)")
    }

    let mut matrix = z_vector.clone();

    for m in 0..matrix.rows {
        matrix.set(m, 0, sigmoid(z_vector.get(m, 0)));
    }

    matrix
}

/// Compute the derivative of the sigmoid function at the given z
fn sigmoid_prime(z: f64) -> f64 {
    sigmoid(z) * (1.0 - sigmoid(z))
}

fn sigmoid_prime_vector(z_vector: &Matrix) -> Matrix {
    if z_vector.columns != 1 {
        panic!("this function is only applicable to column vectors");
    }
    if z_vector.rows == 0 {
        panic!("this function is only valid for column vectors with at least one element (row)")
    }

    let mut matrix = z_vector.clone();

    for m in 0..matrix.rows {
        matrix.set(m, 0, sigmoid_prime(z_vector.get(m, 0)));
    }

    matrix
}

fn z(weights_matrix: &Matrix, inputs_vector: &Matrix, biases_vector: &Matrix) -> Matrix {
    weights_matrix.multiply(inputs_vector).plus(biases_vector)
}

pub struct SimpleNeuralNetwork {
    sizes: Vec<usize>,

    // A vec of weight matricies - one for each inter-layer step
    // For each weight matrix, a row corresponds to the input neurons in the previous layer
    // and a particular neuron in the next layer.
    // Thus, the dimensions will be [rows x columns] [# neurons in the previous layer x # number neurons in the next layer]
    weights: Vec<Matrix>,

    // A vec of row vector matricies - one for inter-layer step
    // Each vector will have length equal to the number of neurons in the next layer
    biases: Vec<Matrix>,
}

impl SimpleNeuralNetwork {
    pub fn new(sizes: Vec<usize>) -> Self {
        let mut biases = Vec::new();

        for i in 1..sizes.len() {
            let mut biases_row_vector =
                column_vec_of_random_values_from_distribution(0.0, 1.0, sizes[i]);
            biases.push(biases_row_vector);
        }
        println!("{:?}", biases);

        let mut weights = Vec::new();

        for l in 1..sizes.len() {
            let mut weights_matrix = Matrix::new_matrix_with_random_values_from_normal_distribution(
                sizes[l],
                sizes[l - 1],
                0.0,
                1.0,
            );
            weights.push(weights_matrix);
        }

        println!("{:?}", weights);

        Self {
            sizes,
            weights,
            biases,
        }
    }

    pub fn num_layers(&self) -> usize {
        self.sizes.len()
    }

    pub fn feed_forward(&self, input_activations: &Matrix) -> Matrix {
        let mut activation_vector = input_activations.clone();
        // println!("feed_forward - weights length: {}", self.weights.len());
        // println!("feed_forward - biases length: {}", self.biases.len());

        // println!("self.num_layers: {:?}", self.num_layers());
        // println!("self.sizes: {:?}", self.sizes.len());

        for i_step in 0..self.num_layers() - 1 {
            // println!("feed_forward step: {}", i_step);
            let z_vec = z(
                &self.weights[i_step],
                &activation_vector,
                &self.biases[i_step],
            );
            // println!("feed_forward - z_vec: {:?}", z_vec);
            activation_vector = sigmoid_vector(&z_vec);
        }

        activation_vector
    }

    /// Feed forward capturing the intermediate z values and activation values
    /// # Arguments
    /// * `input_activations` - The input activations
    /// # Returns
    /// A Vec<FeedForwardIntermediateValues> - one for each layer other than the first (input) layer (simiar to weights and biases)
    fn feed_forward_capturing_intermediates(
        &self,
        input_activations: &Matrix,
    ) -> Vec<FeedForwardIntermediateValues> {
        let mut intermediates = Vec::new();

        let mut activation_vector = input_activations.clone();
        // println!("feed_forward - weights length: {}", self.weights.len());
        // println!("feed_forward - biases length: {}", self.biases.len());

        // println!("self.num_layers: {:?}", self.num_layers());
        // println!("self.sizes: {:?}", self.sizes.len());

        for l in 0..self.num_layers() {
            // println!("feed_forward layer: {}", l);

            if l == 0 {
                intermediates.push(FeedForwardIntermediateValues {
                    // Filling the layer 0 z vector with NaN because it is never used and isn't a valid computation
                    z_vector: Matrix::new_column_vector(
                        vec![f64::NAN; activation_vector.rows].as_slice(),
                    ),
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

    // here it would be nice to be able to type the inputs and outputs as Vector (column vector type Matrix as opposed to just any Matrix)
    pub fn cost_for_single_training_example(
        &self,
        inputs: &Matrix,
        expected_outputs: &Matrix,
    ) -> f64 {
        if inputs.columns != 1 || expected_outputs.columns != 1 {
            panic!("both inpts and expected_outputs must be column vectors");
        }

        let outputs = self.feed_forward(&inputs);

        let diff_vec = expected_outputs.minus(&outputs);
        let length_of_diff_vector = diff_vec.vec_length();
        let square_of_length_of_diff = length_of_diff_vector * length_of_diff_vector;

        // Note that 3B1B does not do the divide by zero and he ends up with a 2 in the derivitive function.
        // I'm doing the divide by 2
        square_of_length_of_diff / 2.0
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
        for i_column in 0..all_inputs.columns {
            let next_input_vector = all_inputs.extract_column_vector(i_column);
            let next_desired_output_vector = expected_outputs.extract_column_vector(i_column);

            let next_cost = self
                .cost_for_single_training_example(&next_input_vector, &next_desired_output_vector);
            cost += next_cost;
        }

        cost / (all_inputs.columns as f64)
    }

    // Backprop Equation 1 from the Neilson book
    // all inputs and the output are Column Vectors
    fn error_last_layer(
        &self,
        output_activations_vector: &Matrix,
        expected_outputs_vector: &Matrix,
        output_layer_z_vector: &Matrix,
    ) -> Matrix {
        // all inputs and the output are Column Vectors
        let mut part1 = output_activations_vector.minus(expected_outputs_vector);
        let part2 = sigmoid_prime_vector(output_layer_z_vector);
        part1.hadamard_product_in_place(&part2);
        part1
    }

    // Backprop Equation 2 from the Neilson book
    // all inputs and the output are Column Vectors
    fn error_any_layer_but_last(
        &self,
        layer: usize,
        error_vector_for_plus_one_layer: &Matrix,
        this_layer_z_vector: &Matrix,
    ) -> Matrix {
        // println!("in error_any_layer_but_last");
        // println!("  - layer: {}", layer);
        // println!("  - weights.length(): {}", self.weights.len());

        // there's once less weight matrix than layer since the input layer doesn't have a weight matrix.
        // so if we are on layer 2, weights[2] will be the weights for layer 3 (which is what we want in EQ 2)
        let weight_matrix = self.weights.get(layer).unwrap();
        let weight_matrix_transpose = weight_matrix.transpose();

        let mut part1 = weight_matrix_transpose.multiply(error_vector_for_plus_one_layer);
        let part2 = sigmoid_prime_vector(this_layer_z_vector);

        part1.hadamard_product_in_place(&part2);
        part1
    }

    /// Returns a Vec of column vectors representing the errors at each neuron at each layer from L-1 to 1
    /// where layer L-1 is the output layer and layer 0 is the input layer.
    fn backpropate_errors(
        &self,
        expected_outputs_vector: &Matrix,
        feedforward_intermediate_values: &Vec<FeedForwardIntermediateValues>,
    ) -> Vec<Matrix> {
        // loop through the layers from back to front, and compute the error at each one.
        // Create a column vector representing the errors at each layer
        let mut error_vectors = Vec::<Matrix>::new(); // each Matrix is a Column Vector

        let last_layer_index = self.num_layers() - 1;

        for l in (1..self.num_layers()).rev() {
            // println!("l: {}", l);
            if l == last_layer_index {
                let last_layer_z_values = &feedforward_intermediate_values.last().unwrap().z_vector;
                let last_layer_activations_vector = &feedforward_intermediate_values
                    .last()
                    .unwrap()
                    .activations_vector;

                // println!("last_layer_z_values:\n{}", last_layer_z_values);
                // println!(
                //     "last_layer_activations_vector:\n{}",
                //     last_layer_activations_vector
                // );

                // println!("expected_outputs_vector:\n{}", expected_outputs_vector);

                let layer_errors_vector = self.error_last_layer(
                    &last_layer_activations_vector,
                    expected_outputs_vector,
                    &last_layer_z_values,
                );

                error_vectors.push(layer_errors_vector);
            } else {
                // println!("any other layer - l: {}", l);
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
        error_vectors_for_each_training_example: &Vec<Vec<Matrix>>,
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

            let mut bias_partials_vector_avg = Matrix::new_zero_matrix(self.sizes[l], 1);

            for i_training_ex in 0..num_training_examples {
                let error_vectors_for_this_training_example =
                    &error_vectors_for_each_training_example[i_training_ex];
                // let error_vector_for_this_layer = &error_vectors_for_this_training_example[layer_index_in_errors_vec];

                let intermediates_for_this_training_example =
                    &intermediates_for_each_training_example[i_training_ex];

                let this_layer_errors_vector = error_vectors_for_this_training_example
                    .get(layer_index_in_errors_vec)
                    .unwrap();
                // println!("\nthis_layer_errors_vector:\n{}", this_layer_errors_vector);

                let previous_layer_activations_vector = &intermediates_for_this_training_example
                    .get(l - 1)
                    .unwrap()
                    .activations_vector;
                let previous_layer_activations_vector_transpose =
                    previous_layer_activations_vector.transpose();

                // println!(
                //     "\nprevious_layer_activations_vector:\n{}",
                //     previous_layer_activations_vector
                // );
                // println!(
                //     "\nprevious_layer_activations_vector_transpose:\n{}",
                //     previous_layer_activations_vector_transpose
                // );

                // println!("\norig weights matrix:\n{}", &self.weights[l - 1]); // -1 because one less than num layers

                let weights_grad =
                    this_layer_errors_vector.multiply(&previous_layer_activations_vector_transpose);
                // println!("\nweights_grad:\n{}", weights_grad);

                // println!(
                //     "\nweights_partials_vector_avg:\n{}",
                //     weights_partials_matrix_avg
                // );

                weights_partials_matrix_avg.add_in_place(&weights_grad);

                // now do the similar thing for the biases
                bias_partials_vector_avg.add_in_place(this_layer_errors_vector);
            }

            weights_partials_matrix_avg.divide_by_scalar_in_place(num_training_examples as f64);
            bias_partials_vector_avg.divide_by_scalar_in_place(num_training_examples as f64);

            weights_partials_matrix_avg.multiply_by_scalar_in_place(learning_rate);
            bias_partials_vector_avg.multiply_by_scalar_in_place(learning_rate);

            // println!(
            //     "\nweights_partials_vector_avg:\n{}",
            //     weights_partials_matrix_avg
            // );
            // println!(
            //     "\nbias_partials_vector_avg:\n{}",
            //     bias_partials_vector_avg
            // );

            // update weights
            // let old_weights_matrix = self.weights.get(l - 1).unwrap(); // -1 because one less than num layers
            // let new_weights_matrix = old_weights_matrix
            //     .minus(&weights_partials_matrix_avg.multiply_by_scalar(learning_rate));
            // let old_biases_vector = self.biases.get(l - 1).unwrap(); // -1 because one less than num layers
            // println!("\nold_weights_matrix:\n{}", old_weights_matrix);
            // println!("\nold_biases_matrix:\n{}", old_biases_vector);

            let weights = self.weights.get_mut(l - 1).unwrap();
            // weights.subtract_in_place(&weights_partials_matrix_avg.multiply_by_scalar(learning_rate));
            weights.subtract_in_place(&weights_partials_matrix_avg);
            // println!("\nupdated weights matrix:\n{}", weights);

            let biases = self.biases.get_mut(l - 1).unwrap();
            // biases.subtract_in_place(&bias_partials_vector_avg.multiply_by_scalar(learning_rate));
            biases.subtract_in_place(&bias_partials_vector_avg);
            // println!("\nupdated biases vector:\n{}", biases);

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
        if training_inputs.columns != expected_outputs.columns {
            panic!("the number of training inputs must match the number of training outputs");
        }

        let initial_cost =
            self.cost_for_training_set_iterative_impl(&training_inputs, &expected_outputs);

        println!("initial cost across entire training set: {}", initial_cost,);

        for _ in 0..epocs {
            let mut error_vectors_for_each_training_example = Vec::new();
            let mut intermediates_for_each_training_example = Vec::new();

            for i_training_example in 0..training_inputs.columns {
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
    z_vector: Matrix,           // column vector
    activations_vector: Matrix, // column vector
}

fn column_vec_of_random_values(min: f64, max: f64, size: usize) -> Matrix {
    let mut rng = rand::thread_rng();

    let mut values = Vec::new();
    for _ in 0..size {
        let x = rng.gen_range(min..max);
        values.push(x);
    }
    Matrix::new_column_vector(&values)
}

fn column_vec_of_random_values_from_distribution(mean: f64, std_dev: f64, size: usize) -> Matrix {
    let mut rng = rand::thread_rng();
    let normal = Normal::new(mean, 1.0).unwrap();

    let mut values = Vec::new();
    for _ in 0..size {
        let x = normal.sample(&mut rng);
        values.push(x);
    }
    Matrix::new_column_vector(&values)
}

fn main() {
    println!("Hello, world!");
    println!("sigmoid(0.85): {}", sigmoid(0.85));
    println!("sigmoid(1.6): {}", sigmoid(1.6));

    let v = column_vector![0.4, 0.7, 1.0, 1.3, 1.6];
    let sv = sigmoid_prime_vector(&v);
    println!("v: {:?}", sv);

    let res = column_vector![
        0.24026074574152914,
        0.22171287329310904,
        0.19661193324148185,
        0.16829836246906024,
        0.13976379193306102
    ];

    let weights_1 = RowsMatrixBuilder::new()
        .with_row(&[0.5, 0.5, 0.5, 0.5, 0.5])
        .with_row(&[0.6, 0.6, 0.6, 0.6, 0.6])
        .build();

    let biases_1 = column_vector![0.1, 0.1];

    let step2z = weights_1.multiply(&res).plus(&biases_1);
    println!("step2z: {:?}", step2z);
    let sstepz = sigmoid_vector(&step2z);
    println!("sstepz: \n{}", sstepz);

    // let inputs = column_vector![1.0, 2.0, 3.0];
    // let mut weights = Matrix::empty_with_num_cols(3);
    // weights.push_row(&[1.0, 2.0, 3.0]);
    // weights.push_row(&[4.0, 5.0, 6.0]);

    // let mut weights = Matrix::new_zero_matrix(2, 3);
    // weights.set(0, 0, 1.0);
    // weights.set(0, 1, 2.0);
    // weights.set(0, 2, 3.0);
    // weights.set(1, 0, 4.0);
    // weights.set(1, 1, 5.0);
    // weights.set(1, 2, 6.0);

    // let outputs = weights.multiply(&inputs);
    // println!("outputs: {:?}", outputs);

    let num_neurons_layer_0 = 3;
    let num_neurons_layer_1 = 2;

    // Layer N: 3 neurons
    // L N-1: 2 neurons
    // weights matrix -> 2x3

    let inputs = column_vector![0.0, 0.5, 1.0];

    let mut weights = Matrix::new_zero_matrix(2, 3);
    weights.set(0, 0, 0.5);
    weights.set(0, 1, 0.5);
    weights.set(0, 2, 0.5);
    weights.set(1, 0, 1.0);
    weights.set(1, 1, 1.0);
    weights.set(1, 2, 1.0);

    let biases = column_vector![0.1, 0.1];

    let nn = SimpleNeuralNetwork {
        sizes: vec![num_neurons_layer_0, num_neurons_layer_1],
        weights: vec![weights],
        biases: vec![biases],
    };

    // let input_activations = column_vector![1.0, 1.0];

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
    use common::scalar_valued_multivariable_point::ScalarValuedMultivariablePoint;
    use float_cmp::approx_eq;

    #[test]
    fn sigmoid_works() {
        assert_eq!(sigmoid(-4.0), 0.01798620996209156);
        assert_eq!(sigmoid(-2.0), 0.11920292202211755);
        assert_eq!(sigmoid(-1.0), 0.2689414213699951);
        assert_eq!(sigmoid(0.0), 0.5);
        assert_eq!(sigmoid(1.0), 0.7310585786300049);
        assert_eq!(sigmoid(2.0), 0.8807970779778823);
        assert_eq!(sigmoid(4.0), 0.9820137900379085);
    }

    #[test]
    fn sigmoid_vector_works() {
        let m1 = column_vector![-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0];
        let m2 = sigmoid_vector(&m1);

        assert_eq!(m2.rows, 7);
        assert_eq!(m2.columns, 1);
        assert_eq!(m2.get(0, 0), sigmoid(-4.0));
        assert_eq!(m2.get(1, 0), sigmoid(-2.0));
        assert_eq!(m2.get(2, 0), sigmoid(-1.0));
        assert_eq!(m2.get(3, 0), 0.5);
        assert_eq!(m2.get(4, 0), sigmoid(1.0));
        assert_eq!(m2.get(5, 0), sigmoid(2.0));
        assert_eq!(m2.get(6, 0), sigmoid(4.0));
    }

    pub fn get_simple_two_layer_nn_for_test() -> SimpleNeuralNetwork {
        let num_neurons_layer_0 = 3;
        let num_neurons_layer_1 = 2;

        // Layer 0 (input): 3 neurons
        // Layer 1 (output): 2 neurons
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

        nn
    }

    pub fn get_simple_get_2_3_1_nn_for_test() -> SimpleNeuralNetwork {
        let num_neurons_layer_0 = 2;
        let num_neurons_layer_1 = 3;
        let num_neurons_layer_2 = 1;

        let inputs = column_vector![0.0, 0.5, 1.0];

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

        let inputs = column_vector![1.0, 2.0, 3.0];

        let weights = RowsMatrixBuilder::new()
            .with_row(&[1.0, 2.0, 3.0])
            .with_row(&[4.0, 5.0, 6.0])
            .build();

        let biases = column_vector![0.5, 0.5];

        let outputs = z(&weights, &inputs, &biases);

        assert_eq!(outputs.rows, 2);
        assert_eq!(outputs.columns, 1);
        assert_eq!(outputs.get(0, 0), 14.5);
        assert_eq!(outputs.get(1, 0), 32.5);
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
        assert_eq!(outputs.rows, 2);
        assert_eq!(outputs.columns, 1);
        assert_eq!(outputs.get(0, 0), 0.7005671424739729);
        assert_eq!(outputs.get(1, 0), 0.8320183851339245);
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
        let z0 = weights_0.multiply(&inputs).plus(&biases_0);
        let sz0 = sigmoid_vector(&z0);
        println!("sz0: {:?}", sz0);

        let z1 = weights_1.multiply(&sz0).plus(&biases_1);
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

        assert_eq!(outputs.rows, 2);
        assert_eq!(outputs.columns, 1);
        assert_eq!(outputs.get(0, 0), sz1.get(0, 0));
        assert_eq!(outputs.get(1, 0), sz1.get(1, 0));

        // the actual outputs, which should be the same as the manually computed outputs
        assert_eq!(outputs.get(0, 0), 0.8707823298624764);
        assert_eq!(outputs.get(1, 0), 0.9063170030285769);
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
        let z0 = weights_0.multiply(&inputs).plus(&biases_0);
        let sz0 = sigmoid_vector(&z0);
        println!("sz0: {:?}", sz0);

        let z1 = weights_1.multiply(&sz0).plus(&biases_1);
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

        assert_eq!(outputs.rows, 2);
        assert_eq!(outputs.columns, 1);
        assert_eq!(outputs.get(0, 0), sz1.get(0, 0));
        assert_eq!(outputs.get(1, 0), sz1.get(1, 0));

        // the actual outputs, which should be the same as the manually computed outputs
        assert_eq!(outputs.get(0, 0), 0.8707823298624764);
        assert_eq!(outputs.get(1, 0), 0.9063170030285769);
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
        let diff = 3.0 - a_output_value;
        let diff_squared = diff * diff;
        let over_two = diff_squared / 2.0;

        println!("a_output: \n{}", a_output_value);
        println!("over_two: \n{}", over_two);

        assert_eq!(c0, over_two);
    }

    #[test]
    pub fn test_cost_for_single_training_example_multiple_output_neurons() {
        let nn = get_three_layer_multiple_output_nn_for_test();

        let input_example = column_vector![0.0, 0.5, 1.0];
        let expected_output_vector = column_vector![2.0, 2.0];

        let c0 = nn.cost_for_single_training_example(&input_example, &expected_output_vector);

        // manually compute the expected cost for the single neuron in the last layer
        let actual_output_vector = nn.feed_forward(&input_example);
        let diff_vec = expected_output_vector.minus(&actual_output_vector);
        let length_of_diff_vector = diff_vec.vec_length();
        let length_of_diff_vector_squared = length_of_diff_vector * length_of_diff_vector;
        let over_two = length_of_diff_vector_squared / 2.0;

        println!("\nactual_output_vector: \n{:?}", actual_output_vector.data);
        println!("diff_vec: \n{:?}", diff_vec.data);
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

        for i_training_example in 0..input_examples.columns {
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

    #[test]
    fn test_nn() {
        let orange = 0.0;
        let blue = 1.0;

        // fake data roughly based on https://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=gauss&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=3,1&seed=0.22934&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false
        let training_data = vec![
            ScalarValuedMultivariablePoint::new_3d(-2.0, -2.0, orange),
            ScalarValuedMultivariablePoint::new_3d(-2.0, -2.0, orange),
            ScalarValuedMultivariablePoint::new_3d(-2.0, -2.0, orange),
            ScalarValuedMultivariablePoint::new_3d(-2.0, -2.0, orange),
            ScalarValuedMultivariablePoint::new_3d(-2.0, -2.0, orange),
            ScalarValuedMultivariablePoint::new_3d(-2.0, -2.0, orange),
            ScalarValuedMultivariablePoint::new_3d(-2.0, -2.0, orange),
            ScalarValuedMultivariablePoint::new_3d(-2.0, -2.0, orange),
            ScalarValuedMultivariablePoint::new_3d(-2.0, -2.0, orange),
            ScalarValuedMultivariablePoint::new_3d(-2.0, -2.0, orange),
            ScalarValuedMultivariablePoint::new_3d(2.0, 2.0, blue),
            ScalarValuedMultivariablePoint::new_3d(2.0, 2.0, blue),
            ScalarValuedMultivariablePoint::new_3d(2.0, 2.0, blue),
            ScalarValuedMultivariablePoint::new_3d(2.0, 2.0, blue),
            ScalarValuedMultivariablePoint::new_3d(2.0, 2.0, blue),
            ScalarValuedMultivariablePoint::new_3d(2.0, 2.0, blue),
            ScalarValuedMultivariablePoint::new_3d(2.0, 2.0, blue),
            ScalarValuedMultivariablePoint::new_3d(2.0, 2.0, blue),
            ScalarValuedMultivariablePoint::new_3d(2.0, 2.0, blue),
            ScalarValuedMultivariablePoint::new_3d(2.0, 2.0, blue),
        ];

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

        assert!(approx_eq!(f64, predicted_output_0, blue, epsilon = 0.01));
        assert!(approx_eq!(f64, predicted_output_1, orange, epsilon = 0.01));

        assert!(approx_eq!(f64, cost_of_predicted_0, 0.0, epsilon = 0.0001));
        assert!(approx_eq!(f64, cost_of_predicted_1, 0.0, epsilon = 0.0001));

        // assert_eq!(1, 0);
    }
}
