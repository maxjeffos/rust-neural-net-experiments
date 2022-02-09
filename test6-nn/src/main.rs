use std::collections::HashMap;
use std::mem::drop;

// use common::activation_functions::{elu, relu, sigmoid, ActivationFunction};
use activation::ActivationFunction;
use common::column_vec_of_random_values_from_distribution;
use common::datapoints::NDTrainingDataPoint;
use common::linalg::{euclidian_distance, euclidian_length, square};
use common::linalg::{ColumnVector, Matrix, MatrixShape};
use metrics::SimpleTimer;
use rand;
use rand::Rng;
use rayon::prelude::*;
use std::sync::{Arc, Mutex};

pub mod builder;
use builder::NeuralNetworkBuilder;

pub mod activation;
use activation::{activate_derivative_vector, activate_vector};

pub mod initializer;
use initializer::Initializer;

pub mod big_theta;
use big_theta::BigTheta;

pub mod training_log;
use training_log::TrainingSessionLogger;

type LayerIndex = usize;

// pub enum Optimizer {
//     SGD(f64),
//     Momentum(f64, f64),
//     Nesterov(f64, f64),
//     AdaGrad(f64),
//     Adam(f64, f64, f64, f64),
// }

#[derive(Debug)]
pub enum Optimizer {
    StanardGradientDescent(StandardGradientDescentConfig),
    Momentum(MomentumConfig),
    // Nesterov(f64, f64),
    // AdaGrad(f64),
    // Adam(f64, f64, f64, f64),
    Adam(AdamConfig),
}

impl Optimizer {
    pub fn standard_gradient_descent(learning_rate: f64) -> Self {
        Optimizer::StanardGradientDescent(StandardGradientDescentConfig {
            learning_rate: learning_rate,
        })
    }

    pub fn momentum(learning_rate: f64, momentum: f64) -> Self {
        Optimizer::Momentum(MomentumConfig {
            learning_rate: learning_rate,
            momentum: momentum,
        })
    }
}

#[derive(Debug)]
pub struct StandardGradientDescentConfig {
    pub learning_rate: f64,
}

#[derive(Debug)]
pub struct MomentumConfig {
    pub learning_rate: f64,
    pub momentum: f64,
}

#[derive(Debug)]
pub struct AdamConfig {
    pub learning_rate: f64,
    pub momentum_decay: f64, // beta_1 in HOML
    pub scaling_decay: f64,  // beta_1 in HOML
    pub epsilon: f64,
}

impl AdamConfig {
    pub fn default() -> Self {
        Self {
            learning_rate: 0.001,
            momentum_decay: 0.9,
            scaling_decay: 0.999,
            epsilon: 1e-7, // GH copilot's suggestion was 1e-8
        }
    }
    pub fn with_learning_rate(learning_rate: f64) -> Self {
        Self {
            learning_rate,
            momentum_decay: 0.9,
            scaling_decay: 0.999,
            epsilon: 1e-7, // GH copilot's suggestion was 1e-8
        }
    }
}

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

#[derive(Debug, Clone)]
struct LayerInfo {
    // Optional because the input layer doesn't have an activation function
    activation_function: Option<ActivationFunction>,
    // weights: Matrix,
    // biases: ColumnVector,
    initializer: Option<String>,
}

impl LayerInfo {
    fn new(activation_function: Option<ActivationFunction>) -> Self {
        Self {
            activation_function,
            initializer: None,
        }
    }

    fn new_with_initializer(
        activation_function: Option<ActivationFunction>,
        initializer: Option<String>,
    ) -> Self {
        Self {
            activation_function,
            initializer,
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

    layer_infos: HashMap<LayerIndex, LayerInfo>,
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

        let mut layer_infos = HashMap::new();
        layer_infos.insert(0, LayerInfo::new(None));
        for l in 1..sizes.len() {
            layer_infos.insert(l, LayerInfo::new(Some(ActivationFunction::Sigmoid)));
        }

        Self {
            sizes,
            weights,
            biases,
            layer_infos,
        }
    }

    pub fn num_layers(&self) -> usize {
        self.sizes.len()
    }

    pub fn is_output_layer(&self, layer_index: LayerIndex) -> bool {
        layer_index == self.num_layers() - 1
    }

    pub fn get_weight_matrix_shape(&self, layer_index: LayerIndex) -> MatrixShape {
        if layer_index == 0 {
            panic!("not valid for input layer (because it has no weights/biases");
        }
        MatrixShape::new(self.sizes[layer_index], self.sizes[layer_index - 1])
    }

    // THIS MAY NOT BE RIGHT!
    // Not sure if fan in is the total number of inbound connections at a layer, or just
    // the number of neurons in the previous layer.
    pub fn get_fan_in(&self, l: LayerIndex) -> usize {
        if l == 0 {
            panic!("not valid for input layer");
        }
        self.sizes[l - 1] * self.sizes[l]
    }

    // THIS MAY NOT BE RIGHT!
    // Not sure if fan out is the total number of outbound connections at a layer, or just
    // the number of neurons in the next layer.
    pub fn get_fan_out(&self, l: LayerIndex) -> usize {
        if self.is_output_layer(l) {
            panic!("not valid for output layer");
        }
        self.sizes[l] * self.sizes[l + 1]
    }

    pub fn feed_forward(&self, input_activations: &ColumnVector) -> ColumnVector {
        let mut activation_v = input_activations.clone();

        for l in 1..self.sizes.len() {
            let layer_info = self.layer_infos.get(&l).unwrap();
            let activation_function = layer_info.activation_function.as_ref().unwrap();

            let z_v = z(
                self.weights.get(&l).unwrap(),
                self.biases.get(&l).unwrap(),
                &activation_v,
            );
            activation_v = activate_vector(&z_v, activation_function);
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
                let layer_info = self.layer_infos.get(&l).unwrap();
                let activation_function = layer_info.activation_function.as_ref().unwrap();
                let z_v = z(
                    self.weights.get(&l).unwrap(),
                    self.biases.get(&l).unwrap(),
                    &activation_v,
                );
                activation_v = activate_vector(&z_v, activation_function);
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

        let output_v = self.feed_forward(&tr_ex.input_v);

        if output_v.num_elements() != tr_ex.desired_output_v.num_elements() {
            println!("output_v len: {}", output_v.num_elements());
            println!(
                "r_ex.desired_output_v len: {}",
                tr_ex.desired_output_v.num_elements()
            );

            panic!("output_v and desired_output_v must have the same length");
        }

        quadratic_cost(&tr_ex.desired_output_v, &output_v)
    }

    pub fn cost_training_set(&self, training_data: &[NDTrainingDataPoint]) -> f64 {
        training_data
            .par_iter()
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
            // TODO: see if there's a way to do this with ColumnVector::from_vec because that would be without cloning
            // and could significantly speed this up since big_theta_v can be really huge
            let b = ColumnVector::new(&big_theta_v[ptr..(ptr + self.sizes[l])]);
            ptr += self.sizes[l];

            weights.insert(l, w);
            biases.insert(l, b);
        }

        let layer_infos = self.layer_infos.clone();

        SimpleNeuralNetwork {
            sizes: self.sizes.clone(),
            weights,
            biases,
            layer_infos,
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
        let l_last_layer = self.num_layers() - 1;
        let layer_info = self.layer_infos.get(&l_last_layer).unwrap();
        let activation_function = layer_info.activation_function.as_ref().unwrap();

        output_activation_v
            .minus(desired_output_v)
            .hadamard_product_chaining(&activate_derivative_vector(
                output_layer_z_v,
                activation_function,
            ))
    }

    // Backprop Equation BP2 from the Neilson book
    fn err_non_last_layer(
        &self,
        layer: LayerIndex,
        plus_one_layer_error_v: &ColumnVector,
        this_layer_z_v: &ColumnVector,
    ) -> ColumnVector {
        let weight_matrix = self.weights.get(&(layer + 1)).unwrap();
        let layer_info = self.layer_infos.get(&layer).unwrap();
        let activation_function = layer_info.activation_function.as_ref().unwrap();

        weight_matrix
            .transpose()
            .mult_vector(plus_one_layer_error_v)
            .hadamard_product_chaining(&activate_derivative_vector(
                this_layer_z_v,
                activation_function,
            ))
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
                // println!("in backprop, l = {}", l);
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

    fn compute_gradients_par(
        &mut self,
        per_tr_ex_data: &Vec<(
            HashMap<usize, FeedForwardIntermediates>,
            HashMap<usize, ColumnVector>,
        )>, // outer Vec is per training example
    ) -> BigTheta {
        // let mut gradients = HashMap::new();
        let mut weights_matricies = HashMap::new();
        let mut bias_vectors = HashMap::new();

        let num_training_examples = per_tr_ex_data.len();

        for l in (1..self.num_layers()).rev() {
            // let mut weights_partials_matrix_avg =
            //     Matrix::new_zero_matrix(self.sizes[l], self.sizes[l - 1]);
            // let mut bias_partials_vector_avg = ColumnVector::new_zero_vector(self.sizes[l]);

            let mut weights_partials_matrix_avg = Arc::new(Mutex::new(Some(
                Matrix::new_zero_matrix(self.sizes[l], self.sizes[l - 1]),
            )));
            let mut bias_partials_vector_avg = Arc::new(Mutex::new(Some(
                ColumnVector::new_zero_vector(self.sizes[l]),
            )));

            // let mut magic_v = Arc::new(Mutex::new(Some(Vec::<usize>::new())));

            per_tr_ex_data
                .par_iter()
                .for_each(|(intermediates, error_vectors)| {
                    let prev_layer_activations_v =
                        &intermediates.get(&(l - 1)).unwrap().activation_v;

                    let this_layer_err_v = error_vectors.get(&l).unwrap();

                    // let prev_layer_activations_v_transpose = prev_layer_activations_v.transpose();
                    // let weights_grad =
                    //     this_layer_err_v.mult_matrix(&prev_layer_activations_v_transpose);

                    let weights_grad = this_layer_err_v.outer_product(&prev_layer_activations_v);

                    let mut weights_partials_matrix_avg =
                        weights_partials_matrix_avg.lock().unwrap();
                    let mut bias_partials_vector_avg = bias_partials_vector_avg.lock().unwrap();

                    weights_partials_matrix_avg
                        .as_mut()
                        .unwrap()
                        .add_in_place(&weights_grad);
                    bias_partials_vector_avg
                        .as_mut()
                        .unwrap()
                        .plus_in_place(this_layer_err_v);

                    // weights_partials_matrix_avg.add_in_place(&weights_grad);
                    // bias_partials_vector_avg.plus_in_place(this_layer_err_v);
                });

            let mut weights_partials_matrix_avg = weights_partials_matrix_avg.lock().unwrap();
            let mut bias_partials_vector_avg = bias_partials_vector_avg.lock().unwrap();

            weights_partials_matrix_avg
                .as_mut()
                .unwrap()
                .divide_by_scalar_in_place(num_training_examples as f64);
            bias_partials_vector_avg
                .as_mut()
                .unwrap()
                .divide_by_scalar_in_place(num_training_examples as f64);

            let w = weights_partials_matrix_avg.take().unwrap();
            let b = bias_partials_vector_avg.take().unwrap();

            weights_matricies.insert(l, w);
            bias_vectors.insert(l, b);
        }

        BigTheta {
            sizes: self.sizes.clone(),
            weights_matricies,
            bias_vectors,
        }
    }

    // try splitting the ||ism up by layers, not just by training examples (which might not be worth the overhead?)
    // another thing to try would be to send batches of tr examples to each worker.__rust_force_expr
    fn compute_gradients_par_4(
        &mut self,
        per_tr_ex_data: &Vec<(
            HashMap<usize, FeedForwardIntermediates>,
            HashMap<usize, ColumnVector>,
        )>, // outer Vec is per training example
    ) -> BigTheta {
        // let mut gradients = HashMap::new();
        // let mut weights_matricies = HashMap::<LayerIndex, Matrix>::new();
        // let mut bias_vectors = HashMap::<LayerIndex, ColumnVector>::new();

        // need to use a mutex to protect weights_matricies and bias_vectors
        let weights_matricies = Arc::new(Mutex::new(Some(HashMap::<LayerIndex, Matrix>::new())));
        let bias_vectors = Arc::new(Mutex::new(Some(HashMap::<LayerIndex, ColumnVector>::new())));

        let num_training_examples = per_tr_ex_data.len();
        let layers_in_from_last_to_1th: Vec<usize> = (1..self.num_layers()).rev().collect();

        layers_in_from_last_to_1th
            .par_iter()
            .for_each(|layer_index| {
                let weights_partials_matrix_avg = Arc::new(Mutex::new(Some(
                    Matrix::new_zero_matrix(self.sizes[*layer_index], self.sizes[*layer_index - 1]),
                )));
                let bias_partials_vector_avg = Arc::new(Mutex::new(Some(
                    ColumnVector::new_zero_vector(self.sizes[*layer_index]),
                )));

                // let mut t_comp_grads_par_4_all_tr_data =
                //     SimpleTimer::start_new("t_compute_gradients_par_4_all_tr_data");

                per_tr_ex_data
                    .par_iter()
                    .for_each(|(intermediates, error_vectors)| {
                        let prev_layer_activations_v =
                            &intermediates.get(&(*layer_index - 1)).unwrap().activation_v;

                        let this_layer_err_v = error_vectors.get(layer_index).unwrap();

                        let weights_grad =
                            this_layer_err_v.outer_product(&prev_layer_activations_v);

                        let mut weights_partials_matrix_avg =
                            weights_partials_matrix_avg.lock().unwrap();
                        weights_partials_matrix_avg
                            .as_mut()
                            .unwrap()
                            .add_in_place(&weights_grad);
                        drop(weights_partials_matrix_avg); // drop the MutexGuard early to reduce lock time

                        let mut bias_partials_vector_avg = bias_partials_vector_avg.lock().unwrap();
                        bias_partials_vector_avg
                            .as_mut()
                            .unwrap()
                            .plus_in_place(this_layer_err_v);
                    });

                // t_comp_grads_par_4_all_tr_data.stop();
                // println!("{}", t_comp_grads_par_4_all_tr_data);

                let mut weights_partials_matrix_avg = weights_partials_matrix_avg.lock().unwrap();
                weights_partials_matrix_avg
                    .as_mut()
                    .unwrap()
                    .divide_by_scalar_in_place(num_training_examples as f64);

                let mut bias_partials_vector_avg = bias_partials_vector_avg.lock().unwrap();
                bias_partials_vector_avg
                    .as_mut()
                    .unwrap()
                    .divide_by_scalar_in_place(num_training_examples as f64);

                let w = weights_partials_matrix_avg.take().unwrap();
                let b = bias_partials_vector_avg.take().unwrap();

                let mut weights_matricies_mutex_guard = weights_matricies.lock().unwrap();
                weights_matricies_mutex_guard
                    .as_mut()
                    .unwrap()
                    .insert(*layer_index, w);
                drop(weights_matricies_mutex_guard); // drop the MutexGuard early to reduce lock time

                let mut bias_vectors_mutex_guard = bias_vectors.lock().unwrap();
                bias_vectors_mutex_guard
                    .as_mut()
                    .unwrap()
                    .insert(*layer_index, b);
            });

        let mut weights_matricies_mutex_guard = weights_matricies.lock().unwrap();
        let weights_matricies = weights_matricies_mutex_guard.take().unwrap();
        drop(weights_matricies_mutex_guard);

        let mut bias_vectors_mutex_guard = bias_vectors.lock().unwrap();
        let bias_vectors = bias_vectors_mutex_guard.take().unwrap();
        drop(bias_vectors_mutex_guard);

        BigTheta {
            sizes: self.sizes.clone(),
            weights_matricies,
            bias_vectors,
        }
    }

    // using Rayon and returning stuff from the par iter
    fn compute_gradients_par_3(
        &mut self,
        per_tr_ex_data: &Vec<(
            HashMap<usize, FeedForwardIntermediates>,
            HashMap<usize, ColumnVector>,
        )>, // outer Vec is per training example
    ) -> BigTheta {
        // let mut gradients = HashMap::new();
        let mut weights_matricies = HashMap::new();
        let mut bias_vectors = HashMap::new();

        let num_training_examples = per_tr_ex_data.len();

        for l in (1..self.num_layers()).rev() {
            // let (tx, rx) = channel();

            let components = per_tr_ex_data
                .par_iter()
                .map(|(intermediates, error_vectors)| {
                    let prev_layer_activations_v =
                        &intermediates.get(&(l - 1)).unwrap().activation_v;

                    let this_layer_err_v = error_vectors.get(&l).unwrap();

                    let weights_grad = this_layer_err_v.outer_product(&prev_layer_activations_v);
                    let bias_grad = this_layer_err_v.clone(); // TODO: gross. Having trouble sending a reference through the channel.

                    (weights_grad, bias_grad)
                })
                // .for_each_with(tx, |s, (intermediates, error_vectors)| {
                //     let prev_layer_activations_v =
                //         &intermediates.get(&(l - 1)).unwrap().activation_v;
                //     let this_layer_err_v = error_vectors.get(&l).unwrap();
                //     let weights_grad = this_layer_err_v.outer_product(&prev_layer_activations_v);
                //     let bias_grad = this_layer_err_v.clone(); // TODO: gross. Having trouble sending a reference through the channel.
                //     // this_layer_err_v is the bias gradient
                //     s.send((weights_grad, bias_grad)).unwrap();
                // })
                .collect::<Vec<(Matrix, ColumnVector)>>();

            // the following data munging is a bit overly verbose and inefficient
            // could the components be a Vec<(Matrix, &ColumnVector)> in order to avoid the clone above?

            let mut w_items = Vec::<Matrix>::new();
            let mut b_items = Vec::<ColumnVector>::new();
            for x in components.into_iter() {
                w_items.push(x.0);
                b_items.push(x.1);
            }

            // now create the averages
            // let mut w = w_items.iter().fold(Matrix::new_zero_matrix(self.sizes[l], self.sizes[l - 1]), |acc, x| acc.add_in_place(x));
            let mut w = Matrix::new_zero_matrix(self.sizes[l], self.sizes[l - 1]);
            w_items
                .iter()
                .for_each(|next_w_item| w.add_in_place(next_w_item));
            w.divide_by_scalar_in_place(num_training_examples as f64);

            let mut b = ColumnVector::new_zero_vector(self.sizes[l]);
            b_items
                .iter()
                .for_each(|next_b_item| b.plus_in_place(next_b_item));
            b.divide_by_scalar_in_place(num_training_examples as f64);

            weights_matricies.insert(l, w);
            bias_vectors.insert(l, b);
        }

        BigTheta {
            sizes: self.sizes.clone(),
            weights_matricies,
            bias_vectors,
        }
    }

    pub fn train(
        &mut self,
        training_data: &Vec<NDTrainingDataPoint>,
        epocs: usize,
        learning_rate: f64,
        check_options: Option<&CheckOptions>,
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

        let default_check_options = CheckOptions::no_checks();
        let check_options = check_options.unwrap_or(&default_check_options);

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

            println!(
                "finished ff for all training points - epoch {}",
                epocs_count
            );

            // note: compute_gradients takes data for ALL training examples
            let mut gradients = self.compute_gradients(&per_tr_ex_data);

            // temp - print the gradients
            let g1 = gradients.get(&1).unwrap();
            println!("gradients for layer 1:");
            println!("weights grad: \n{}", g1.0);
            println!("biases grad: \n{}", g1.1);

            if check_options.gradient_checking {
                let approx_gradients_big_v = self.approximate_cost_gradient(training_data);
                // unroll the actual gradients
                let d_vec = self.unroll_gradients(&gradients);

                let ed = euclidian_distance(&approx_gradients_big_v, &d_vec);
                println!("ed: {}", ed);

                // if ed > GRADIENT_CHECK_EPSILON_SQUARED {
                //     panic!("failed gradient check");
                // }

                let normalized_distance = euclidian_distance(&approx_gradients_big_v, &d_vec)
                    / (euclidian_length(&approx_gradients_big_v) + euclidian_length(&d_vec));

                // if normalized_distance > GRADIENT_CHECK_EPSILON_SQUARED {
                //     panic!("failed gradient check");
                // }
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

            // Remove
            // Show weights and biases in l1
            let l1_weights = self.weights.get(&1).unwrap();
            println!("weights in layer 1: \n{}", l1_weights);
            let l1_biases = self.biases.get(&1).unwrap();
            println!("biases in layer 1: \n{}", l1_biases);

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
            epocs_count, final_cost,
        );
    }

    pub fn train_stochastic(
        &mut self,
        training_data: &Vec<NDTrainingDataPoint>,
        epocs: usize,
        optimizer: &Optimizer,
        mini_batch_size: usize,
        check_options: Option<&CheckOptions>,
        early_stop_config: Option<EarlyStopConfig>,
        full_cost_update_every: Option<usize>, // After how every epocs do you want to do a full cost update across the entire training set, if at all.
        session_logger: Option<TrainingSessionLogger>,
    ) {
        println!("computing initial cross accross entire training dataset...");
        let mut t_init_cost = SimpleTimer::start_new("t_init_cost");
        let initial_cost = self.cost_training_set(&training_data);
        t_init_cost.stop();
        println!("initial cost across entire training set: {}", initial_cost);
        println!("t_init_cost: {}", t_init_cost);

        if let Some(ref session_logger) = session_logger {
            let network_config = training_log::NetworkConfig::from_neural_network(&self);
            let optimizer_str = format!("{:?}", optimizer);
            session_logger.write_training_session_file(initial_cost, network_config, optimizer_str);
        }

        // here's what this does:
        // for epocs
        //     for each training example
        //         feed forward, capturing intermediates
        //         backpropagate to compute errors at each neuron of each layer
        //     compute gradients for w and b
        //     update weights and biases

        let mut epochs_count = 0;
        let mut prev_cost = initial_cost;

        let default_check_options = CheckOptions::no_checks();
        let check_options = check_options.unwrap_or(&default_check_options);
        let num_samples = training_data.len();

        // for the optimizers
        let mut momentum = BigTheta::zero_from_sizes(&self.sizes); // used by both Momentum and Adam optimizers
        let mut s = BigTheta::zero_from_sizes(&self.sizes); // used by Adam optimizer

        loop {
            if epochs_count >= epocs {
                println!("stopping after {} epocs", epochs_count);
                break;
            }

            let mut mini_batch_start = 0;
            let mut mini_batch_end = num_samples;

            if mini_batch_size < num_samples {
                // println!("num_samples: {}", num_samples);
                // let mini_batch_size = 200;
                let max_starting_point = num_samples - mini_batch_size;
                // println!("max_starting_point: {}", max_starting_point);
                mini_batch_start = rand::thread_rng().gen_range(0..max_starting_point); // note the upper limit is exclusive
                mini_batch_end = mini_batch_start + mini_batch_size; // this will be exclusive when used in the slice range
            }

            println!(
                "\nstarting mini batch from {} to {}",
                mini_batch_start, mini_batch_end
            );

            let tr_data_mini_batch = &training_data[mini_batch_start..mini_batch_end];

            // println!("mini batch size: {}", tr_data_mini_batch.len());

            let mut t_mini_batch_ff = SimpleTimer::start_new("t_mini_batch_ff");

            let per_tr_ex_data = tr_data_mini_batch
                .par_iter()
                .map(|tr_ex| {
                    let intermediates = self.feed_forward_capturing(&tr_ex.input_v);
                    let errors = self.backprop(&tr_ex.desired_output_v, &intermediates);

                    (intermediates, errors)
                })
                .collect::<Vec<(
                    HashMap<usize, FeedForwardIntermediates>,
                    HashMap<usize, ColumnVector>,
                )>>();

            // non ||: 3800ms
            t_mini_batch_ff.stop();
            println!("t_mini_batch_ff (all data points): {}", t_mini_batch_ff);

            println!(
                "finished ff for all training points - epoch {}",
                epochs_count
            );

            // note: compute_gradients takes data for ALL training examples
            let mut t_compute_gradients = SimpleTimer::start_new("t_compute_gradients");

            let mut gradients = self.compute_gradients_par_4(&per_tr_ex_data);

            t_compute_gradients.stop();
            println!(
                "t_compute_gradients epoch {}: {}",
                epochs_count, t_compute_gradients
            );

            // if check_options.gradient_checking {
            //     let approx_gradients_big_v = self.approximate_cost_gradient(training_data);
            //     // unroll the actual gradients
            //     let d_vec = self.unroll_gradients(&gradients);

            //     let ed = euclidian_distance(&approx_gradients_big_v, &d_vec);
            //     println!("ed: {}", ed);

            //     if ed > GRADIENT_CHECK_EPSILON_SQUARED {
            //         panic!("failed gradient check");
            //     }

            //     let normalized_distance = euclidian_distance(&approx_gradients_big_v, &d_vec)
            //         / (euclidian_length(&approx_gradients_big_v) + euclidian_length(&d_vec));

            //     if normalized_distance > GRADIENT_CHECK_EPSILON_SQUARED {
            //         panic!("failed gradient check");
            //     }
            // }

            // let mut momentum: HashMap<LayerIndex, (Matrix, ColumnVector)> = HashMap::new();
            // for l in 1..self.num_layers() {
            //     let w_empty = Matrix::new_zero_matrix_with_shape(&self.get_weight_matrix_shape(l));
            //     let b_empty = ColumnVector::new_zero_vector(self.sizes[l]);
            //     momentum.insert(l, (w_empty, b_empty));
            // }

            // update the weights and biases
            // TODO: extract to method for easy testing

            // for layer_index in 1..self.sizes.len() {
            //     match optimizer {
            //         Optimizer::StanardGradientDescent(optimizer_config) => {
            //             let weights_grad = gradients.get_weights_matrix_mut(&layer_index);
            //             weights_grad.multiply_by_scalar_in_place(optimizer_config.learning_rate);

            //             let weights = self.weights.get_mut(&layer_index).unwrap();
            //             weights.subtract_in_place(&weights_grad);

            //             let bias_grad = gradients.get_bias_vector_mut(&layer_index);
            //             bias_grad.multiply_by_scalar_in_place(optimizer_config.learning_rate);
            //             let biases = self.biases.get_mut(&layer_index).unwrap();
            //             biases.minus_in_place(&bias_grad);
            //         }
            //         Optimizer::Momentum(optimizer_config) => {
            //             let weights_grad = gradients.get_weights_matrix_mut(&layer_index);
            //             weights_grad.multiply_by_scalar_in_place(optimizer_config.learning_rate);

            //             let m_w = momentum.get_weights_matrix_mut(&layer_index);
            //             m_w.multiply_by_scalar_in_place(optimizer_config.momentum);
            //             m_w.subtract_in_place(&weights_grad);
            //             let weights = self.weights.get_mut(&layer_index).unwrap();
            //             weights.add_in_place(&m_w);

            //             let bias_grad = gradients.get_bias_vector_mut(&layer_index);
            //             bias_grad.multiply_by_scalar_in_place(optimizer_config.learning_rate);
            //             let m_b = momentum.get_bias_vector_mut(&layer_index);
            //             m_b.multiply_by_scalar_in_place(optimizer_config.momentum);
            //             m_b.minus_in_place(&bias_grad); // TODO: standardize the subtract_in_place / minus_in_place naming

            //             let biases = self.biases.get_mut(&layer_index).unwrap();
            //             biases.plus_in_place(&m_b);
            //         } // orig impl
            //           // Optimizer::Momentum(optimizer_config) => {
            //           //     let m = momentum.get_mut(&layer_index).unwrap();

            //           //     let weights_grad = gradients.get_weights_matrix_mut(&layer_index);
            //           //     weights_grad.multiply_by_scalar_in_place(optimizer_config.learning_rate);
            //           //     let m_w = &mut m.0;
            //           //     m_w.multiply_by_scalar_in_place(optimizer_config.momentum);
            //           //     m_w.subtract_in_place(&weights_grad);
            //           //     let weights = self.weights.get_mut(&layer_index).unwrap();
            //           //     weights.add_in_place(&m_w);

            //           //     let bias_grad = gradients.get_bias_vector_mut(&layer_index);
            //           //     bias_grad.multiply_by_scalar_in_place(optimizer_config.learning_rate);
            //           //     let m_b = &mut m.1;
            //           //     m_b.multiply_by_scalar_in_place(optimizer_config.momentum);
            //           //     m_b.minus_in_place(&bias_grad); // TODO: standardize the subtract_in_place / minus_in_place naming

            //           //     let biases = self.biases.get_mut(&layer_index).unwrap();
            //           //     biases.plus_in_place(&m_b);
            //           // }
            //     }
            // }

            match optimizer {
                Optimizer::StanardGradientDescent(optimizer_config) => {
                    for layer_index in 1..self.sizes.len() {
                        let weights_grad = gradients.get_weights_matrix_mut(&layer_index);
                        weights_grad.multiply_by_scalar_in_place(optimizer_config.learning_rate);

                        let weights = self.weights.get_mut(&layer_index).unwrap();
                        weights.subtract_in_place(&weights_grad);

                        let bias_grad = gradients.get_bias_vector_mut(&layer_index);
                        bias_grad.multiply_by_scalar_in_place(optimizer_config.learning_rate);
                        let biases = self.biases.get_mut(&layer_index).unwrap();
                        biases.minus_in_place(&bias_grad);
                    }
                }
                Optimizer::Momentum(optimizer_config) => {
                    for layer_index in 1..self.sizes.len() {
                        let weights_grad = gradients.get_weights_matrix_mut(&layer_index);
                        weights_grad.multiply_by_scalar_in_place(optimizer_config.learning_rate);

                        let m_w = momentum.get_weights_matrix_mut(&layer_index);
                        m_w.multiply_by_scalar_in_place(optimizer_config.momentum);
                        m_w.subtract_in_place(&weights_grad);
                        let weights = self.weights.get_mut(&layer_index).unwrap();
                        weights.add_in_place(&m_w);

                        let bias_grad = gradients.get_bias_vector_mut(&layer_index);
                        bias_grad.multiply_by_scalar_in_place(optimizer_config.learning_rate);
                        let m_b = momentum.get_bias_vector_mut(&layer_index);
                        m_b.multiply_by_scalar_in_place(optimizer_config.momentum);
                        m_b.minus_in_place(&bias_grad); // TODO: standardize the subtract_in_place / minus_in_place naming

                        let biases = self.biases.get_mut(&layer_index).unwrap();
                        biases.plus_in_place(&m_b);
                    }
                }
                Optimizer::Adam(adam_optimizer_config) => {
                    // 1. update momentum
                    momentum.mult_scalar_in_place(adam_optimizer_config.momentum_decay);
                    let x = gradients
                        .mult_scalar_return_new(1.0 - adam_optimizer_config.momentum_decay);
                    momentum.subtract_in_place(&x);

                    // 2. update s
                    s.mult_scalar_in_place(adam_optimizer_config.scaling_decay);
                    let mut x = gradients.clone(); // TODO: some chaining methods on BigTheta would be nice to clean this up
                    x.elementwise_mult_in_place(&gradients);
                    x.mult_scalar_in_place(1.0 - adam_optimizer_config.scaling_decay); // could make an elementwise_square
                    s.add_in_place(&x);

                    // compute momentum_decay_t and scaling_decay_t
                    // see https://machinelearningmastery.com/adam-optimization-from-scratch/
                    // and https://arxiv.org/pdf/1412.6980.pdf (the Adam paper)
                    // let momentum_decay_t = 1.0 - adam_optimizer_config.momentum_decay.powf(1.0 / adam_optimizer_config.epochs);
                    let momentum_decay_t = adam_optimizer_config
                        .momentum_decay
                        .powf(1.0 + epochs_count as f64);

                    let scaling_decay_t = adam_optimizer_config
                        .scaling_decay
                        .powf(1.0 + epochs_count as f64);

                    // 3. create m_hat (temp value)
                    let mut m_hat = momentum.divide_scalar_return_new(1.0 - momentum_decay_t);

                    // 4. create s_hat (temp value)
                    let mut s_hat = s.divide_scalar_return_new(1.0 - scaling_decay_t);

                    // 5. update weights and biases
                    // TODO: could prett this up with chaining methods
                    m_hat.mult_scalar_in_place(adam_optimizer_config.learning_rate);
                    s_hat.add_scalar_to_each_element_in_place(adam_optimizer_config.epsilon);
                    s_hat.elementwise_square_root_in_place();
                    m_hat.elementwise_divide_in_place(&s_hat);

                    // now do the layer by layer update (until I make BigTheta the main deal in the NN struct)
                    for layer_index in 1..self.sizes.len() {
                        let weights = self.weights.get_mut(&layer_index).unwrap();
                        let w = m_hat.get_weights_matrix_mut(&layer_index);
                        weights.add_in_place(&w);

                        let biases = self.biases.get_mut(&layer_index).unwrap();
                        let b = m_hat.get_bias_vector_mut(&layer_index);
                        biases.plus_in_place(&b);
                    }
                }
            }

            // gradients
            //     .iter_mut()
            //     .for_each(|(layer_index, (weights_grad, bias_grad))| {
            //         let layer_index = *layer_index;

            //         match optimizer {
            //             Optimizer::StanardGradientDescent(optimizer_config) => {
            //                 weights_grad
            //                     .multiply_by_scalar_in_place(optimizer_config.learning_rate);
            //                 bias_grad.multiply_by_scalar_in_place(optimizer_config.learning_rate);

            //                 let weights = self.weights.get_mut(&layer_index).unwrap();
            //                 let biases = self.biases.get_mut(&layer_index).unwrap();
            //                 weights.subtract_in_place(&weights_grad);
            //                 biases.minus_in_place(&bias_grad);
            //             }
            //             Optimizer::Momentum(optimizer_config) => {
            //                 weights_grad
            //                     .multiply_by_scalar_in_place(optimizer_config.learning_rate);
            //                 bias_grad.multiply_by_scalar_in_place(optimizer_config.learning_rate);

            //                 // update the momentum
            //                 let m = momentum.get_mut(&layer_index).unwrap();
            //                 let m_w = &mut m.0;
            //                 let m_b = &mut m.1;
            //                 m_w.multiply_by_scalar_in_place(optimizer_config.momentum);
            //                 m_b.multiply_by_scalar_in_place(optimizer_config.momentum);
            //                 m_w.subtract_in_place(&weights_grad);
            //                 m_b.minus_in_place(&bias_grad); // TODO: standardize the subtract_in_place / minus_in_place naming

            //                 let weights = self.weights.get_mut(&layer_index).unwrap();
            //                 let biases = self.biases.get_mut(&layer_index).unwrap();
            //                 weights.add_in_place(&m_w);
            //                 biases.plus_in_place(&m_b);
            //             }
            //         }
            //     });

            if check_options.cost_decreasing_check {
                let cost = self.cost_training_set(&training_data);
                println!(
                    "cost across training set after epoch {}: {}",
                    epochs_count, cost
                );
                if cost > prev_cost {
                    panic!(
                        "cost across training set increased from {} to {} on epoc {}",
                        prev_cost, cost, epochs_count
                    );
                }
                prev_cost = cost;
            }

            epochs_count += 1;

            let mut maybe_test_set_cost = None;

            if let Some(full_cost_update_every) = full_cost_update_every {
                if epochs_count % full_cost_update_every == 0 {
                    println!(
                        "\nCost Update\ncomputing cross across entire training dataset after {} epocs...",
                        epochs_count
                    );
                    let training_set_cost = self.cost_training_set(&training_data);
                    println!(
                        "  - cost across entire training set after {} epocs: {}",
                        epochs_count, training_set_cost,
                    );

                    println!(
                        "computing cross across entire test dataset after {} epocs...",
                        epochs_count
                    );
                    let test_set_cost = self.cost_training_set(&training_data);
                    println!(
                        "  - cost across test set after {} epocs: {}",
                        epochs_count, training_set_cost,
                    );
                    maybe_test_set_cost = Some(training_set_cost);

                    if let Some(ref session_logger) = session_logger {
                        let epoch = epochs_count - 1;
                        session_logger.write_update(
                            epoch,
                            epochs_count,
                            training_set_cost,
                            test_set_cost,
                        );
                    }
                }
            }

            if let Some(ref esc) = early_stop_config {
                if epochs_count % esc.check_every == 0 {
                    println!("TRAINING SET COST CHECK / EARLY STOP");

                    let cost = if let Some(test_set_cost) = maybe_test_set_cost {
                        // the test cost was previously compute in this epoch in the above full cost update
                        // so don't re-compute it
                        test_set_cost
                    } else {
                        let test_set_cost = self.cost_training_set(esc.test_data);
                        test_set_cost
                    };

                    println!("test dataset cost after {} epocs: {}", epochs_count, cost);
                    if cost <= esc.cost_threshold {
                        println!("stopping after {} epocs", epochs_count);
                        break;
                    }
                }
            }
        }

        println!("computing final cross accross entire training dataset...");
        let final_cost = self.cost_training_set(&training_data);
        println!(
            "\ncost across entire training set after {} epocs: {}",
            epochs_count, final_cost,
        );
    }
}

pub struct EarlyStopConfig<'a> {
    pub test_data: &'a [NDTrainingDataPoint],
    pub cost_threshold: f64,
    pub check_every: usize,
}

fn main() {
    println!("experimenting with training log stuff");

    let (training_data, test_data) = mnist_data::get_mnist_data(50000, 10000);
    println!("got the MNIST training data");

    // let jelu_instance = activation::jelu::JELU::new(-3.0);

    let mut nn = NeuralNetworkBuilder::new()
        .with_input_layer(784)
        .with_hidden_layer(
            100,
            Initializer::HeForReLUAndVariants,
            // ActivationFunction::ELU,
            // ActivationFunction::JELU(jelu_instance.clone()),
            ActivationFunction::LeakyReLU(activation::leaky_relu::LeakyReLU::new(0.1)),
        )
        .with_hidden_layer(
            100,
            Initializer::HeForReLUAndVariants,
            // ActivationFunction::ELU,
            // ActivationFunction::JELU(jelu_instance.clone()),
            ActivationFunction::LeakyReLU(activation::leaky_relu::LeakyReLU::new(0.1)),
        )
        .with_hidden_layer(
            100,
            Initializer::HeForReLUAndVariants,
            // ActivationFunction::ELU,
            // ActivationFunction::JELU(jelu_instance.clone()),
            ActivationFunction::LeakyReLU(activation::leaky_relu::LeakyReLU::new(0.1)),
        )
        // .with_hidden_layer(
        //     100,
        //     Initializer::XavierNormalHOMLForSigmoid,
        //     ActivationFunction::Sigmoid,
        // )
        .with_hidden_layer(
            50,
            Initializer::HeForReLUAndVariants,
            // ActivationFunction::Sigmoid,
            ActivationFunction::LeakyReLU(activation::leaky_relu::LeakyReLU::new(0.1)),
        )
        .with_output_layer(
            10,
            Initializer::XavierNormalHOMLForSigmoid,
            ActivationFunction::Sigmoid,
        )
        .build();

    let check_options = CheckOptions {
        gradient_checking: false,
        cost_decreasing_check: false,
    };

    println!("about to start training");

    let mut t_total = SimpleTimer::start_new("t_total");

    let early_stop_config = EarlyStopConfig {
        // test_data: &test_data,
        test_data: &test_data[0..10000],
        cost_threshold: 0.001,
        check_every: 10,
    };

    // TODO: should probably replace the ::new() + create_training_log_directory() thing with
    // an init() constructor that does both
    let mut session_logger = TrainingSessionLogger::new();
    session_logger
        .create_training_log_directory()
        .expect("failed creating traininig log directory");
    println!(
        "training session id: {:?}",
        &session_logger.training_session_id
    );
    println!(
        "created training log directory: {:?}",
        &session_logger.full_session_output_directory
    );

    nn.train_stochastic(
        &training_data,
        10_000,
        // &Optimizer::standard_gradient_descent(0.9),
        // &Optimizer::momentum(0.9, 0.9),
        &Optimizer::Adam(AdamConfig::default()),
        // &Optimizer::momentum(0.9, 0.9),
        5000,
        Some(&check_options),
        Some(early_stop_config),
        Some(10),
        Some(session_logger),
    );
    println!("done training");

    let t0 = test_data.get(0).unwrap();
    let p_output_v = nn.feed_forward(&t0.input_v);
    println!("p_output_v: {}", p_output_v);
    println!("desired_output_v: {}", &t0.desired_output_v);

    let t0_cost = nn.cost_single_tr_ex(t0);
    println!("t0_cost: {}", t0_cost);

    for i_test in 0..50 {
        let t = test_data.get(i_test).unwrap();
        let cost = nn.cost_single_tr_ex(t);
        println!("cost of {}th: {}", i_test, cost);
    }

    // let sub_tr_set = &test_data[0..1000];
    // println!("computing cost of a sub-set of the test data...");
    let test_set_cost = nn.cost_training_set(&test_data);
    println!("\ntest_set_cost: {}", test_set_cost);

    t_total.stop();
    println!("\nt_total: {}", t_total);
}

#[cfg(test)]
mod tests {
    use std::panic;

    use crate::activation::leaky_relu::LeakyReLU;

    use super::*;
    use common::column_vector;
    use common::linalg::RowsMatrixBuilder;
    use float_cmp::approx_eq;
    use time_test::time_test;

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

        let mut layer_infos = HashMap::new();
        layer_infos.insert(0, LayerInfo::new(None));
        layer_infos.insert(1, LayerInfo::new(Some(ActivationFunction::Sigmoid)));
        layer_infos.insert(2, LayerInfo::new(Some(ActivationFunction::Sigmoid)));

        let nn = SimpleNeuralNetwork {
            sizes: vec![
                num_neurons_layer_0,
                num_neurons_layer_1,
                num_neurons_layer_2,
            ],
            weights,
            biases,
            layer_infos,
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

        NeuralNetworkBuilder::new()
            .with_input_layer(num_neurons_layer_0)
            .with_hidden_layer(
                num_neurons_layer_1,
                Initializer::Manual(weights_l1, biases_l1),
                ActivationFunction::Sigmoid,
            )
            .with_output_layer(
                num_neurons_layer_2,
                Initializer::Manual(weights_l2, biases_l2),
                ActivationFunction::Sigmoid,
            )
            .build()
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
        // 3x2 NN; l1 weights matrix: 2x3
        let nn = NeuralNetworkBuilder::new()
            .with_input_layer(3)
            .with_output_layer(
                2,
                Initializer::Manual(
                    RowsMatrixBuilder::new()
                        .with_row(&[0.5, 0.5, 0.5])
                        .with_row(&[1.0, 1.0, 1.0])
                        .build(),
                    column_vector![0.1, 0.1],
                ),
                ActivationFunction::Sigmoid,
            )
            .build();

        let input_v = column_vector![0.0, 0.5, 1.0];
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
    fn test_fan_in_fan_out() {
        let nn = SimpleNeuralNetwork::new(vec![2, 3, 4, 2]);

        assert_eq!(
            panic::catch_unwind(|| {
                nn.get_fan_in(0);
            })
            .is_err(),
            true
        );
        assert_eq!(nn.get_fan_out(0), 6);

        assert_eq!(nn.get_fan_in(1), 6);
        assert_eq!(nn.get_fan_out(1), 12);

        assert_eq!(nn.get_fan_in(2), 12);
        assert_eq!(nn.get_fan_out(2), 8);

        assert_eq!(nn.get_fan_in(3), 8);
        assert_eq!(
            panic::catch_unwind(|| {
                nn.get_fan_out(3);
            })
            .is_err(),
            true
        );
    }

    #[test]
    fn feed_forward_works_simple_three_layer() {
        let num_neurons_layer_0 = 3;
        let num_neurons_layer_1 = 5;
        let num_neurons_layer_2 = 2;

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
        let sz0 = activate_vector(&z0, &ActivationFunction::Sigmoid);
        println!("sz0: {:?}", sz0);

        let z1 = weights_l2.mult_vector(&sz0).plus(&bias_v_l2);
        let sz1 = activate_vector(&z1, &ActivationFunction::Sigmoid);
        println!("sz1: {:?}", sz1);

        println!("\n now doing with feed_forward");

        let mut weights = HashMap::new();
        weights.insert(1, weights_l1);
        weights.insert(2, weights_l2);

        let mut biases = HashMap::new();
        biases.insert(1, bias_v_l1);
        biases.insert(2, bias_v_l2);

        let sizes = vec![
            num_neurons_layer_0,
            num_neurons_layer_1,
            num_neurons_layer_2,
        ];

        let mut layer_infos = HashMap::new();
        layer_infos.insert(0, LayerInfo::new(None));
        for l in 1..sizes.len() {
            layer_infos.insert(l, LayerInfo::new(Some(ActivationFunction::Sigmoid)));
        }

        let nn = SimpleNeuralNetwork {
            sizes,
            weights,
            biases,
            layer_infos,
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
        let sz0 = activate_vector(&z0, &ActivationFunction::Sigmoid);
        println!("sz0: {:?}", sz0);

        let z1 = weights_l2.mult_vector(&sz0).plus(&bias_v_l2);
        let sz1 = activate_vector(&z1, &ActivationFunction::Sigmoid);
        println!("sz1: {:?}", sz1);

        println!("\n now doing with feed_forward");

        let mut weights = HashMap::new();
        weights.insert(1, weights_l1);
        weights.insert(2, weights_l2);

        let mut biases = HashMap::new();
        biases.insert(1, bias_v_l1);
        biases.insert(2, bias_v_l2);

        let sizes = vec![
            num_neurons_layer_0,
            num_neurons_layer_1,
            num_neurons_layer_2,
        ];

        let mut layer_infos = HashMap::new();
        layer_infos.insert(0, LayerInfo::new(None));
        for l in 1..sizes.len() {
            layer_infos.insert(l, LayerInfo::new(Some(ActivationFunction::Sigmoid)));
        }

        let nn = SimpleNeuralNetwork {
            sizes,
            weights,
            biases,
            layer_infos,
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

    fn get_data_set_1() -> Vec<NDTrainingDataPoint> {
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

    // #[ignore] // because slow
    #[test]
    fn test_nn() {
        time_test!();
        let training_data = get_data_set_1();

        // 2 x 3 x 1
        let num_neurons_layer_0 = 2;
        let num_neurons_layer_1 = 3;
        let num_neurons_layer_2 = 1;

        // W0: 3x2 (num_neurons_layer_1 x num_neurons_layer_0)
        // W1: 1x3 (num_neurons_layer_2 x num_neurons_layer_1)

        let weights_l1 = RowsMatrixBuilder::new()
            .with_row(&[0.2, 0.2])
            .with_row(&[0.4, 0.4])
            .with_row(&[0.6, 0.6])
            .build();
        let bias_v_l1 = column_vector![0.1, 0.1, 0.1];

        let weights_l2 = RowsMatrixBuilder::new().with_row(&[0.5, 0.5, 0.5]).build();
        let bias_v_l2 = column_vector![0.1];

        let mut nn = NeuralNetworkBuilder::new()
            .with_input_layer(num_neurons_layer_0)
            .with_hidden_layer(
                num_neurons_layer_1,
                Initializer::Manual(weights_l1, bias_v_l1),
                ActivationFunction::Sigmoid,
            )
            .with_output_layer(
                num_neurons_layer_2,
                Initializer::Manual(weights_l2, bias_v_l2),
                ActivationFunction::Sigmoid,
            )
            .build();

        let epocs = 2000;
        let learning_rate = 0.9;

        nn.train(
            &training_data,
            epocs,
            learning_rate,
            Some(&CheckOptions::all_checks()),
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

    // #[ignore] // because slow
    #[test]
    fn test_nn_using_constructor_for_random_initial_weights_and_biases() {
        time_test!();
        // try the same data set as before but use the NN constructor to initialize with random weights/biases
        let training_data = get_data_set_1();

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

        nn.train(&training_data, epocs, learning_rate, Some(&check_options));

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
    fn simple_test_to_get_elu_sorted_out() {
        let w = RowsMatrixBuilder::new().with_row(&[0.5]).build();
        let b = column_vector![0.0];

        println!("initial weights:");
        println!("{}", w);

        println!("initial biases:");
        println!("{}", b);

        // 2x3x1 - i know from above that it converges

        let mut nn = NeuralNetworkBuilder::new()
            .with_input_layer(2)
            .with_hidden_layer(3, Initializer::RandomBasic, ActivationFunction::ELU)
            .with_output_layer(1, Initializer::RandomBasic, ActivationFunction::Sigmoid)
            .build();

        let training_data = get_data_set_1();

        // let training_data = vec![
        //     NDTrainingDataPoint::new(
        //         column_vector![0.0],
        //         column_vector![0.0],
        //     ),
        //     NDTrainingDataPoint::new(
        //         column_vector![1.0],
        //         column_vector![1.0],
        //     ),
        //     NDTrainingDataPoint::new(
        //         column_vector![4.0],
        //         column_vector![4.0],
        //     ),
        // ];

        let check_options = CheckOptions {
            gradient_checking: true,
            cost_decreasing_check: true,
        };

        nn.train(&training_data, 250, 0.9, Some(&check_options));
        nn.train_stochastic(
            &training_data,
            1000,
            &Optimizer::standard_gradient_descent(0.9),
            training_data.len(),
            Some(&check_options),
            None,
            None,
            None,
        );

        let final_weights = nn.weights.get(&1).unwrap();
        let final_biases = nn.biases.get(&1).unwrap();

        println!("final weights:");
        println!("{}", final_weights);

        println!("final biases:");
        println!("{}", final_biases);

        // predict
        let tr_ex = NDTrainingDataPoint {
            input_v: column_vector![2.0, 2.0],
            desired_output_v: column_vector![1.0],
        };
        let predicted_output_0 = nn.feed_forward(&tr_ex.input_v).into_value();
        println!(
            "predicted_output_0: {} (desired is {} (BLUE))",
            &predicted_output_0, BLUE
        );
        let cost_of_predicted_0 = nn.cost_single_tr_ex(&tr_ex);
        println!("cost_of_predicted_0: \n{}", &cost_of_predicted_0);

        // predict
        println!("second prediction");
        let tr_ex = NDTrainingDataPoint {
            input_v: column_vector![-2.0, -2.0],
            desired_output_v: column_vector![0.0],
        };
        let predicted_output_1 = nn.feed_forward(&tr_ex.input_v).into_value();
        println!(
            "predicted_output_1: {} (desired is {} (ORANGE))",
            &predicted_output_1, ORANGE
        );
        let cost_of_predicted_1 = nn.cost_single_tr_ex(&tr_ex);
        println!("cost_of_predicted_1: \n{}", &cost_of_predicted_1);

        assert!(approx_eq!(f64, predicted_output_0, BLUE, epsilon = 0.05));
        assert!(approx_eq!(f64, predicted_output_1, ORANGE, epsilon = 0.05));

        assert!(approx_eq!(f64, cost_of_predicted_0, 0.0, epsilon = 0.002));
        assert!(approx_eq!(f64, cost_of_predicted_1, 0.0, epsilon = 0.002));

        // assert_eq!(0, 1);
    }

    #[test]
    fn simple_jelu_test() {
        let w = RowsMatrixBuilder::new().with_row(&[0.5]).build();
        let b = column_vector![0.0];

        println!("initial weights:");
        println!("{}", w);

        println!("initial biases:");
        println!("{}", b);

        // 2x3x1 - i know from above that it converges

        let mut nn = NeuralNetworkBuilder::new()
            .with_input_layer(2)
            .with_hidden_layer(
                3,
                Initializer::HeForReLUAndVariants,
                ActivationFunction::JELU(activation::jelu::JELU::new(-3.0)),
            )
            .with_output_layer(
                1,
                Initializer::XavierNormalHOMLForSigmoid,
                ActivationFunction::Sigmoid,
            )
            .build();

        let training_data = get_data_set_1();

        let check_options = CheckOptions {
            gradient_checking: true,
            cost_decreasing_check: true,
        };

        nn.train(&training_data, 250, 0.9, Some(&check_options));
        nn.train_stochastic(
            &training_data,
            1000,
            &Optimizer::standard_gradient_descent(0.9),
            training_data.len(),
            Some(&check_options),
            None,
            None,
            None,
        );

        let final_weights = nn.weights.get(&1).unwrap();
        let final_biases = nn.biases.get(&1).unwrap();

        println!("final weights:");
        println!("{}", final_weights);

        println!("final biases:");
        println!("{}", final_biases);

        // predict
        let tr_ex = NDTrainingDataPoint {
            input_v: column_vector![2.0, 2.0],
            desired_output_v: column_vector![1.0],
        };
        let predicted_output_0 = nn.feed_forward(&tr_ex.input_v).into_value();
        println!(
            "predicted_output_0: {} (desired is {} (BLUE))",
            &predicted_output_0, BLUE
        );
        let cost_of_predicted_0 = nn.cost_single_tr_ex(&tr_ex);
        println!("cost_of_predicted_0: \n{}", &cost_of_predicted_0);

        // predict
        println!("second prediction");
        let tr_ex = NDTrainingDataPoint {
            input_v: column_vector![-2.0, -2.0],
            desired_output_v: column_vector![0.0],
        };
        let predicted_output_1 = nn.feed_forward(&tr_ex.input_v).into_value();
        println!(
            "predicted_output_1: {} (desired is {} (ORANGE))",
            &predicted_output_1, ORANGE
        );
        let cost_of_predicted_1 = nn.cost_single_tr_ex(&tr_ex);
        println!("cost_of_predicted_1: \n{}", &cost_of_predicted_1);

        assert!(approx_eq!(f64, predicted_output_0, BLUE, epsilon = 0.05));
        assert!(approx_eq!(f64, predicted_output_1, ORANGE, epsilon = 0.05));

        assert!(approx_eq!(f64, cost_of_predicted_0, 0.0, epsilon = 0.002));
        assert!(approx_eq!(f64, cost_of_predicted_1, 0.0, epsilon = 0.002));

        // assert_eq!(0, 1);
    }

    #[test]
    fn simple_leaky_relu_test() {
        let w = RowsMatrixBuilder::new().with_row(&[0.5]).build();
        let b = column_vector![0.0];

        println!("initial weights:");
        println!("{}", w);

        println!("initial biases:");
        println!("{}", b);

        // 2x3x1 - i know from above that it converges

        let mut nn = NeuralNetworkBuilder::new()
            .with_input_layer(2)
            .with_hidden_layer(
                3,
                Initializer::RandomBasic,
                ActivationFunction::LeakyReLU(activation::leaky_relu::LeakyReLU::new(0.1)),
            )
            .with_output_layer(1, Initializer::RandomBasic, ActivationFunction::Sigmoid)
            .build();

        let training_data = get_data_set_1();

        let check_options = CheckOptions {
            gradient_checking: true,
            cost_decreasing_check: true,
        };

        nn.train(&training_data, 250, 0.9, Some(&check_options));
        nn.train_stochastic(
            &training_data,
            1000,
            &Optimizer::standard_gradient_descent(0.9),
            training_data.len(),
            Some(&check_options),
            None,
            None,
            None,
        );

        let final_weights = nn.weights.get(&1).unwrap();
        let final_biases = nn.biases.get(&1).unwrap();

        println!("final weights:");
        println!("{}", final_weights);

        println!("final biases:");
        println!("{}", final_biases);

        // predict
        let tr_ex = NDTrainingDataPoint {
            input_v: column_vector![2.0, 2.0],
            desired_output_v: column_vector![1.0],
        };
        let predicted_output_0 = nn.feed_forward(&tr_ex.input_v).into_value();
        println!(
            "predicted_output_0: {} (desired is {} (BLUE))",
            &predicted_output_0, BLUE
        );
        let cost_of_predicted_0 = nn.cost_single_tr_ex(&tr_ex);
        println!("cost_of_predicted_0: \n{}", &cost_of_predicted_0);

        // predict
        println!("second prediction");
        let tr_ex = NDTrainingDataPoint {
            input_v: column_vector![-2.0, -2.0],
            desired_output_v: column_vector![0.0],
        };
        let predicted_output_1 = nn.feed_forward(&tr_ex.input_v).into_value();
        println!(
            "predicted_output_1: {} (desired is {} (ORANGE))",
            &predicted_output_1, ORANGE
        );
        let cost_of_predicted_1 = nn.cost_single_tr_ex(&tr_ex);
        println!("cost_of_predicted_1: \n{}", &cost_of_predicted_1);

        assert!(approx_eq!(f64, predicted_output_0, BLUE, epsilon = 0.05));
        assert!(approx_eq!(f64, predicted_output_1, ORANGE, epsilon = 0.05));

        assert!(approx_eq!(f64, cost_of_predicted_0, 0.0, epsilon = 0.002));
        assert!(approx_eq!(f64, cost_of_predicted_1, 0.0, epsilon = 0.002));

        // assert_eq!(0, 1);
    }

    // #[ignore] // because slow
    #[test]
    fn test_nn_using_more_hidden_layers_with_more_neurons() {
        time_test!();

        // try more layers and more neurons in the hidden layers to see if I can improve number of epocs
        let training_data = get_data_set_1();

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

        let epocs = 1000;
        let learning_rate = 0.9;

        let check_options = CheckOptions {
            gradient_checking: false,
            cost_decreasing_check: true,
        };

        nn.train(&training_data, epocs, learning_rate, Some(&check_options));

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

        assert!(approx_eq!(f64, cost_of_predicted_0, 0.0, epsilon = 0.001));
        assert!(approx_eq!(f64, cost_of_predicted_1, 0.0, epsilon = 0.001));
    }

    #[test]
    fn test_nn_using_more_hidden_layers_with_more_neurons_with_relu_hidden_layers() {
        // this is the same test as above except with ReLU (and different epoch and epsilons in the assertions, and using the newer train method)
        time_test!();

        // try more layers and more neurons in the hidden layers to see if I can improve number of epocs
        let training_data = get_data_set_1();

        // 2 x 16 x 16 x 1
        let mut nn = NeuralNetworkBuilder::new()
            .with_input_layer(2)
            .with_hidden_layer(
                16,
                Initializer::XavierNormalHOMLForSigmoid, // TODO: impl HOML for ReLU variants.
                ActivationFunction::ReLU,                // TODO: use ReLU but it's not working yet
            )
            .with_hidden_layer(
                16,
                Initializer::XavierNormalHOMLForSigmoid, // TODO: impl HOML for ReLU variants.
                ActivationFunction::ReLU,                // TODO: use ReLU but it's not working yet
            )
            .with_output_layer(1, Initializer::RandomBasic, ActivationFunction::Sigmoid)
            .build();

        println!("initial weights:");
        nn.weights.values().for_each(|w| {
            println!("{}", w);
        });

        println!("initial biases:");
        nn.biases.values().for_each(|b| {
            println!("{}", b);
        });

        let epocs = 250;

        let check_options = CheckOptions {
            gradient_checking: false,
            cost_decreasing_check: true,
        };

        let mini_batch_size = training_data.len();

        let test_data = vec![
            NDTrainingDataPoint::new(column_vector![2.0, 2.0], column_vector![1.0]),
            NDTrainingDataPoint::new(column_vector![-2.0, -2.0], column_vector![0.0]),
        ];

        let early_stop_config = EarlyStopConfig {
            test_data: &test_data,
            cost_threshold: 0.0001,
            check_every: 10,
        };

        nn.train_stochastic(
            &training_data,
            epocs,
            &Optimizer::standard_gradient_descent(0.9),
            mini_batch_size,
            Some(&check_options),
            Some(early_stop_config),
            None,
            None,
        );

        println!("\nin test, after training");

        // prediction 1
        println!("\nprediction 1");
        let tr_ex = NDTrainingDataPoint {
            input_v: column_vector![2.0, 2.0],
            desired_output_v: column_vector![1.0],
        };
        let predicted_output_1 = nn.feed_forward(&tr_ex.input_v).into_value();
        println!(
            "predicted_output_1: {} (desired is {} (BLUE))",
            &predicted_output_1, BLUE
        );
        let cost_of_predicted_1 = nn.cost_single_tr_ex(&tr_ex);
        println!("cost_of_predicted_1: \n{}", &cost_of_predicted_1);

        // prediction 1
        println!("\nprediction 2");
        let tr_ex = NDTrainingDataPoint {
            input_v: column_vector![-2.0, -2.0],
            desired_output_v: column_vector![0.0],
        };
        let predicted_output_2 = nn.feed_forward(&tr_ex.input_v).into_value();
        println!(
            "predicted_output_2: {} (desired is {} (ORANGE))",
            &predicted_output_2, ORANGE
        );
        let cost_of_predicted_2 = nn.cost_single_tr_ex(&tr_ex);
        println!("cost_of_predicted_2: \n{}", &cost_of_predicted_2);

        assert!(approx_eq!(f64, predicted_output_1, BLUE, epsilon = 0.05));
        assert!(approx_eq!(f64, predicted_output_2, ORANGE, epsilon = 0.05));
        assert!(approx_eq!(f64, cost_of_predicted_1, 0.0, epsilon = 0.001));
        assert!(approx_eq!(f64, cost_of_predicted_2, 0.0, epsilon = 0.001));
    }

    #[test]
    fn test_nn_using_more_hidden_layers_with_more_neurons_with_leaky_relu_and_momentum_opt() {
        time_test!();

        // try more layers and more neurons in the hidden layers to see if I can improve number of epocs
        let training_data = get_data_set_1();

        let lr = LeakyReLU::new(0.1);

        // 2 x 16 x 16 x 1
        let mut nn = NeuralNetworkBuilder::new()
            .with_input_layer(2)
            .with_hidden_layer(
                16,
                Initializer::XavierNormalHOMLForSigmoid, // TODO: impl HOML for ReLU variants.
                ActivationFunction::LeakyReLU(lr.clone()), // TODO: figure out how use just one instance
            )
            .with_hidden_layer(
                16,
                Initializer::XavierNormalHOMLForSigmoid,
                ActivationFunction::LeakyReLU(lr),
            )
            .with_output_layer(1, Initializer::RandomBasic, ActivationFunction::Sigmoid)
            .build();

        println!("initial weights:");
        nn.weights.values().for_each(|w| {
            println!("{}", w);
        });

        println!("initial biases:");
        nn.biases.values().for_each(|b| {
            println!("{}", b);
        });

        let epocs = 75;

        let check_options = CheckOptions {
            gradient_checking: false,
            cost_decreasing_check: false, // not valid for Momentum optimizer
        };

        let mini_batch_size = training_data.len();

        let test_data = vec![
            NDTrainingDataPoint::new(column_vector![2.0, 2.0], column_vector![1.0]),
            NDTrainingDataPoint::new(column_vector![-2.0, -2.0], column_vector![0.0]),
        ];

        let early_stop_config = EarlyStopConfig {
            test_data: &test_data,
            cost_threshold: 0.0001,
            check_every: 10,
        };

        nn.train_stochastic(
            &training_data,
            epocs,
            // &Optimizer::standard_gradient_descent(0.9),
            &Optimizer::momentum(0.9, 0.9),
            mini_batch_size,
            Some(&check_options),
            Some(early_stop_config),
            None,
            None,
        );

        println!("\nin test, after training");

        // prediction 1
        println!("\nprediction 1");
        let tr_ex = NDTrainingDataPoint {
            input_v: column_vector![2.0, 2.0],
            desired_output_v: column_vector![1.0],
        };
        let predicted_output_1 = nn.feed_forward(&tr_ex.input_v).into_value();
        println!(
            "predicted_output_1: {} (desired is {} (BLUE))",
            &predicted_output_1, BLUE
        );
        let cost_of_predicted_1 = nn.cost_single_tr_ex(&tr_ex);
        println!("cost_of_predicted_1: \n{}", &cost_of_predicted_1);

        // prediction 1
        println!("\nprediction 2");
        let tr_ex = NDTrainingDataPoint {
            input_v: column_vector![-2.0, -2.0],
            desired_output_v: column_vector![0.0],
        };
        let predicted_output_2 = nn.feed_forward(&tr_ex.input_v).into_value();
        println!(
            "predicted_output_2: {} (desired is {} (ORANGE))",
            &predicted_output_2, ORANGE
        );
        let cost_of_predicted_2 = nn.cost_single_tr_ex(&tr_ex);
        println!("cost_of_predicted_2: \n{}", &cost_of_predicted_2);

        assert!(approx_eq!(f64, predicted_output_1, BLUE, epsilon = 0.05));
        assert!(approx_eq!(f64, predicted_output_2, ORANGE, epsilon = 0.05));
        assert!(approx_eq!(f64, cost_of_predicted_1, 0.0, epsilon = 0.001));
        assert!(approx_eq!(f64, cost_of_predicted_2, 0.0, epsilon = 0.001));
    }

    #[test]
    fn test_nn_using_more_hidden_layers_with_more_neurons_with_leaky_relu_and_adam_opt() {
        time_test!();

        // try more layers and more neurons in the hidden layers to see if I can improve number of epocs
        let training_data = get_data_set_1();

        let lr = LeakyReLU::new(0.1);

        // 2 x 16 x 16 x 1
        let mut nn = NeuralNetworkBuilder::new()
            .with_input_layer(2)
            .with_hidden_layer(
                16,
                Initializer::XavierNormalHOMLForSigmoid, // TODO: impl HOML for ReLU variants.
                ActivationFunction::LeakyReLU(lr.clone()), // TODO: figure out how use just one instance
            )
            .with_hidden_layer(
                16,
                Initializer::XavierNormalHOMLForSigmoid,
                ActivationFunction::LeakyReLU(lr),
            )
            .with_output_layer(1, Initializer::RandomBasic, ActivationFunction::Sigmoid)
            .build();

        println!("initial weights:");
        nn.weights.values().for_each(|w| {
            println!("{}", w);
        });

        println!("initial biases:");
        nn.biases.values().for_each(|b| {
            println!("{}", b);
        });

        // with Adam, I can converge with even fewer epocs, but keeping it at 75
        // so I can compare the time to Momentum.
        let epocs = 75;

        let check_options = CheckOptions {
            gradient_checking: false,
            cost_decreasing_check: false,
        };

        let mini_batch_size = training_data.len();

        let test_data = vec![
            NDTrainingDataPoint::new(column_vector![2.0, 2.0], column_vector![1.0]),
            NDTrainingDataPoint::new(column_vector![-2.0, -2.0], column_vector![0.0]),
        ];

        let early_stop_config = EarlyStopConfig {
            test_data: &test_data,
            cost_threshold: 0.0001,
            check_every: 10,
        };

        nn.train_stochastic(
            &training_data,
            epocs,
            // &Optimizer::standard_gradient_descent(0.9),
            &Optimizer::Adam(AdamConfig::with_learning_rate(0.01)),
            mini_batch_size,
            Some(&check_options),
            Some(early_stop_config),
            None,
            None,
        );

        println!("\nin test, after training");

        // prediction 1
        println!("\nprediction 1");
        let tr_ex = NDTrainingDataPoint {
            input_v: column_vector![2.0, 2.0],
            desired_output_v: column_vector![1.0],
        };
        let predicted_output_1 = nn.feed_forward(&tr_ex.input_v).into_value();
        println!(
            "predicted_output_1: {} (desired is {} (BLUE))",
            &predicted_output_1, BLUE
        );
        let cost_of_predicted_1 = nn.cost_single_tr_ex(&tr_ex);
        println!("cost_of_predicted_1: \n{}", &cost_of_predicted_1);

        // prediction 1
        println!("\nprediction 2");
        let tr_ex = NDTrainingDataPoint {
            input_v: column_vector![-2.0, -2.0],
            desired_output_v: column_vector![0.0],
        };
        let predicted_output_2 = nn.feed_forward(&tr_ex.input_v).into_value();
        println!(
            "predicted_output_2: {} (desired is {} (ORANGE))",
            &predicted_output_2, ORANGE
        );
        let cost_of_predicted_2 = nn.cost_single_tr_ex(&tr_ex);
        println!("cost_of_predicted_2: \n{}", &cost_of_predicted_2);

        assert!(approx_eq!(f64, predicted_output_1, BLUE, epsilon = 0.05));
        assert!(approx_eq!(f64, predicted_output_2, ORANGE, epsilon = 0.05));
        assert!(approx_eq!(f64, cost_of_predicted_1, 0.0, epsilon = 0.001));
        assert!(approx_eq!(f64, cost_of_predicted_2, 0.0, epsilon = 0.001));

        // assert_eq!(1, 0);
    }

    // #[test]
    // fn test_with_mnist() {
    //     time_test!();

    //     let (training_data, test_data) = mnist_data::get_mnist_data();
    //     let mut nn = NeuralNetworkBuilder::new()
    //         .with_input_layer(784)
    //         .with_hidden_layer(100, BuilderWeightsAndBiasesConfig::RandomBasic)
    //         .with_hidden_layer(100, BuilderWeightsAndBiasesConfig::RandomBasic)
    //         .with_output_layer(10, BuilderWeightsAndBiasesConfig::RandomBasic)
    //         .build();

    //     let check_options = CheckOptions {
    //         gradient_checking: false,
    //         cost_decreasing_check: false,
    //     };

    //     // let n = training_data[0].input_v.num_elements();
    //     // assert_eq!(n, 0);
    //     nn.train(&training_data, 1, 0.9, Some(check_options));
    // }

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
        let nn = SimpleNeuralNetwork::new(vec![2, 3, 2]);
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
