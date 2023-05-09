use std::collections::HashMap;

// use common::activation_functions::{elu, relu, sigmoid, ActivationFunction};
use activation::ActivationFunction;
use common::column_vec_of_random_values_from_distribution;
use common::datapoints::NDTrainingDataPoint;
use metrics::SimpleTimer;
use rand;
use rand::Rng;
use rayon::prelude::*;

pub mod builder;

pub mod activation;
use activation::{activate_derivative_vector, activate_vector};

pub mod initializer;
use initializer::Initializer;

pub mod big_theta;
use big_theta::BigTheta;

pub mod training_log;
use training_log::TrainingSessionLogger;

pub mod layer_config;
use layer_config::LayerConfig;

pub mod cost;
use cost::quadratic_cost;

pub mod optimizer;
use optimizer::Optimizer;

type LayerIndex = usize;

use common::linalg::{euclidian_distance, euclidian_length, ColumnVector, Matrix, MatrixShape};

mod errors;
use errors::{InvalidLayerIndex, NeuralNetworkError, VectorDimensionMismatch};

/// FeedForwardIntermediates is used during the feed-forward of backpropagation.
/// It contains the intermediate values that are needed during the backward pass to compute gradients.
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

/// ForwardAndBackPassData encapsulates all the layer-specific data collected during the forward pass as well as the back pass.
struct ForwardAndBackPassData {
    /// intermediates is a map from layer index to the intermediates for that layer, computed during the forward pass
    intermediates: HashMap<LayerIndex, FeedForwardIntermediates>,
    /// error_vectors is a map from layer index to the error vector for that layer, computed during the back pass
    error_vectors: HashMap<LayerIndex, ColumnVector>,
}

// /// ForwardPassTrainingExampleData encapsulates the data collected during the forward pass for all training examples
// struct ForwardPassTrainingExampleData {
//     examples: Vec<ForwardPassLayerSpecificData>,
// }

/// LayerGradients encapsulates the gradients for a single layer, seperately for the weights and biases.
/// I think this is no longer used because where it was used is now using BigTheta instead.
struct LayerGradients {
    weight_gradients: Matrix,
    bias_gradients: ColumnVector,
}

impl LayerGradients {
    fn new(weight_gradients: Matrix, bias_gradients: ColumnVector) -> Self {
        Self {
            weight_gradients,
            bias_gradients,
        }
    }
}

pub struct CheckOptions {
    pub gradient_checking: bool,
    pub cost_decreasing_check: bool,
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

/// z computes the z vector, i.e. the weighted sum of the inputs and the bias.
fn z(weight_matrix: &Matrix, bias_v: &ColumnVector, input_v: &ColumnVector) -> ColumnVector {
    weight_matrix.mult_vector(input_v).add_chaining(bias_v)
}

pub struct EarlyStopConfig<'a> {
    pub test_data: &'a [NDTrainingDataPoint],
    pub cost_threshold: f64,
    pub check_every: usize,
}

pub struct SimpleNeuralNetwork {
    sizes: Vec<LayerIndex>,

    /// A HashMap of the weights keyed by the layer index.
    /// The dimensions will be [rows x columns] [# neurons in the previous layer x # neurons in the next layer].
    weights: HashMap<LayerIndex, Matrix>,

    /// A HashMap of the biases keyed by the layer index.
    /// The dimension of each ColumnVector will be the number neurons in the layer.
    biases: HashMap<LayerIndex, ColumnVector>,

    /// Meta data about each layer, such as the activation function and the initializer used.
    layer_infos: HashMap<LayerIndex, LayerConfig>,
}

impl SimpleNeuralNetwork {
    /// Creates a new SimpleNeuralNetwork with the given number of layers. This is old and you should really use the builder instead.
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
        layer_infos.insert(0, LayerConfig::new(None));
        for l in 1..sizes.len() {
            layer_infos.insert(l, LayerConfig::new(Some(ActivationFunction::Sigmoid)));
        }

        Self {
            sizes,
            weights,
            biases,
            layer_infos,
        }
    }

    /// Gets the number of layers in the network.
    pub fn num_layers(&self) -> usize {
        self.sizes.len()
    }

    /// Tells you if the given layer index corresponds to the output layer.
    pub fn is_output_layer(&self, layer_index: LayerIndex) -> bool {
        layer_index == self.num_layers() - 1
    }

    /// Gets the layer index of the output layer
    pub fn output_layer_index(&self) -> LayerIndex {
        self.num_layers() - 1
    }

    /// Gets the dimensions (shape) of the weight matrix for the given layer.
    /// Useful in gradient checking.
    pub fn weight_matrix_shape(&self, layer_index: LayerIndex) -> MatrixShape {
        if layer_index == 0 {
            panic!("not valid for input layer (because it has no weights/biases");
            // TODO: replace with an error like `NotValidForInputLayer`
        }
        MatrixShape::new(self.sizes[layer_index], self.sizes[layer_index - 1])
    }

    // TODO(dedupe): this also exists in BigTheta. Should I just use a BigTheta in SimpleNeuralNetwork?
    fn weights_at_layer_mut(
        &mut self,
        layer_index: LayerIndex,
    ) -> Result<&mut Matrix, NeuralNetworkError> {
        self.weights
            .get_mut(&layer_index)
            .ok_or(NeuralNetworkError::InvalidLayerIndex(InvalidLayerIndex(
                layer_index,
            )))
    }

    // TODO(dedupe): this also exists in BigTheta. Should I just use a BigTheta in SimpleNeuralNetwork?
    fn bias_at_layer_mut(
        &mut self,
        layer_index: LayerIndex,
    ) -> Result<&mut ColumnVector, NeuralNetworkError> {
        self.biases
            .get_mut(&layer_index)
            .ok_or(NeuralNetworkError::InvalidLayerIndex(InvalidLayerIndex(
                layer_index,
            )))
    }

    // TODO: use new methods here for getting layer weights / biases and return Result<ColumVector, NeuralNetworkError>
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

    /// Feed forward capturing the intermediate z vectors (weighted sum vectors) and activation vectors.
    /// # Arguments
    /// * `input_activations` - The input activations
    /// # Returns
    /// A HashMap<LayerIndex, FeedForwardIntermediateValues> with an entry for each layer. For the input layer, only the activation_v is populated since z_v doesn't make sense for it.
    /// You could also say that the input vector (or input_activations) *is* the activations vector for the input layer.
    fn feed_forward_capturing_intermediates(
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

    /// Given the single training example, feed the input forward through the network to compute the output.
    /// Then compute the cost based on that computed output and the desired output (found in the training example).
    pub fn cost_single_tr_ex(
        &self,
        tr_ex: &NDTrainingDataPoint,
    ) -> Result<f64, VectorDimensionMismatch> {
        if tr_ex.input_v.num_elements() != self.sizes[0] {
            return Err(VectorDimensionMismatch::new_with_msg(
                tr_ex.input_v.num_elements(),
                self.sizes[0],
                "input_v must have the same number of elements as the number of neurons in the input layer"
            ));
        }

        let output_v = self.feed_forward(&tr_ex.input_v);
        quadratic_cost(&tr_ex.desired_output_v, &output_v)
    }

    /// Computes the cost for a set of training points
    pub fn cost_training_set(
        &self,
        training_data: &[NDTrainingDataPoint],
    ) -> Result<f64, VectorDimensionMismatch> {
        let sum = training_data
            .par_iter()
            .map(|tr_ex| self.cost_single_tr_ex(tr_ex))
            .try_reduce(|| 0.0, |acc, cost| Ok(acc + cost))?;

        Ok(sum / training_data.len() as f64)
    }

    /// Computes the error in the output layer.
    /// Backprop Equation (the one that is unlabeled but follows after BP1a. I assume they meant to label it BP1b)
    /// from the Neilson book
    fn err_output_layer(
        &self,
        output_activation_v: &ColumnVector,
        desired_output_v: &ColumnVector,
        output_layer_z_v: &ColumnVector,
    ) -> ColumnVector {
        let layer_info = self.layer_infos.get(&self.output_layer_index()).unwrap();
        let activation_function = layer_info.activation_function.as_ref().unwrap();

        output_activation_v
            .subtract(desired_output_v)
            .hadamard_product_chaining(&activate_derivative_vector(
                output_layer_z_v,
                activation_function,
            ))
    }

    /// Computes the error in the non-output layers.
    /// Backprop Equation BP2 from the Neilson book
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
                self.err_output_layer(&activations_v, desired_output_v, &z_v)
            } else {
                let error_vector_for_plus_one_layer = error_vectors.get(&(l + 1)).unwrap();
                // println!("in backprop, l = {}", l);
                self.err_non_last_layer(l, error_vector_for_plus_one_layer, z_v)
            };

            error_vectors.insert(l, err_v);
        }

        error_vectors
    }

    /// `compute_gradients` computes the gradients for the neural network using backpropagation.
    /// This function takes a more structured input data type and returns a more structured output data type.
    fn compute_gradients(
        &mut self,
        // A slice of `ForwardAndBackPassData` instances, one for each training example
        forward_and_back_pass_data_for_all_training_examples: &[ForwardAndBackPassData],
    ) -> BigTheta {
        let num_training_examples = forward_and_back_pass_data_for_all_training_examples.len();

        let mut bt_weights = HashMap::new();
        let mut bt_biases = HashMap::new();

        // Iterate through the layers in reverse order (from output to input), skipping the input (0th) layer
        for l in (1..self.num_layers()).rev() {
            // Initialize an average weights gradient matrix with zeros for the current layer
            let mut avg_weight_gradients =
                Matrix::new_zero_matrix(self.sizes[l], self.sizes[l - 1]);

            // Initialize an average bias gradient vector with zeros for the current layer
            let mut avg_bias_gradients = ColumnVector::new_zero_vector(self.sizes[l]);

            // Iterate through the forward and back pass data for all training examples
            for d in forward_and_back_pass_data_for_all_training_examples.iter() {
                // get the activation vector for the previous layer and current training example
                let prev_layer_act_v = &d.intermediates.get(&(l - 1)).unwrap().activation_v;

                // get the error vector for the current layer and current training example
                let this_layer_err_v = d.error_vectors.get(&l).unwrap();

                // Calculate the weight gradients for the current layer by multiplying
                // the error vector of the current layer and the transposed activation vector
                // of the previous layer
                let w_grad = this_layer_err_v.mult_matrix(&prev_layer_act_v.transpose());

                // Add the calculated weight gradients to the average weight gradient matrix
                avg_weight_gradients.add_mut(&w_grad);

                // Add the calculated bias gradients (which are equal to the error vector)
                // to the average bias gradient vector
                avg_bias_gradients.add_mut(this_layer_err_v);
            }

            // Finish computing the average weight and bias gradients by dividing by the number of training examples
            avg_weight_gradients.div_scalar_mut(num_training_examples as f64);
            avg_bias_gradients.div_scalar_mut(num_training_examples as f64);

            bt_weights.insert(l, avg_weight_gradients);
            bt_biases.insert(l, avg_bias_gradients);
        }

        BigTheta {
            sizes: self.sizes.clone(),
            weights_matrices: bt_weights,
            bias_vectors: bt_biases,
        }
    }

    /// Compute the gradients using parallelism.
    /// This impl uses par_iter with map/reduce for vastly improved performance compared to using mutexes, as in previous versions.
    fn compute_gradients_par_6(
        &mut self,
        per_tr_ex_data: &[(
            HashMap<usize, FeedForwardIntermediates>,
            HashMap<usize, ColumnVector>,
        )],
    ) -> BigTheta {
        let num_training_examples = per_tr_ex_data.len();

        // TODO: consider extracting this to a method to improve readability. Note that I made a test for this functionality, though
        // it is just testing the same code, it isn't testing a shared method - see `test_rev_layer_indexs_computation`.
        let layers_in_from_last_to_1th: Vec<usize> = (1..self.num_layers()).rev().collect();

        let mut weights_matrices = HashMap::<LayerIndex, Matrix>::new();
        let mut bias_vectors = HashMap::<LayerIndex, ColumnVector>::new();

        for layer_index in layers_in_from_last_to_1th {
            let (mut weights_partials_matrix_avg, mut bias_partials_vector_avg) = per_tr_ex_data
                .par_iter()
                .map(|(intermediates, error_vectors)| {
                    let prev_layer_activations_v =
                        &intermediates.get(&(layer_index - 1)).unwrap().activation_v;

                    let this_layer_err_v = error_vectors.get(&layer_index).unwrap();

                    let weights_grad = this_layer_err_v.outer_product(&prev_layer_activations_v);

                    let bias_grad = this_layer_err_v.clone();

                    (weights_grad, bias_grad)
                })
                .reduce(
                    || {
                        (
                            Matrix::new_zero_matrix(
                                self.sizes[layer_index],
                                self.sizes[layer_index - 1],
                            ),
                            ColumnVector::new_zero_vector(self.sizes[layer_index]),
                        )
                    },
                    |(mut weights_acc, mut bias_acc), (weights_grad, bias_grad)| {
                        weights_acc.add_mut(&weights_grad);
                        bias_acc.add_mut(&bias_grad);
                        (weights_acc, bias_acc)
                    },
                );

            weights_partials_matrix_avg.div_scalar_mut(num_training_examples as f64);
            bias_partials_vector_avg.div_scalar_mut(num_training_examples as f64);

            weights_matrices.insert(layer_index, weights_partials_matrix_avg.clone());
            bias_vectors.insert(layer_index, bias_partials_vector_avg.clone());
        }

        BigTheta {
            sizes: self.sizes.clone(),
            weights_matrices,
            bias_vectors,
        }
    }

    pub fn train(
        &mut self,
        training_data: &Vec<NDTrainingDataPoint>,
        epocs: usize,
        learning_rate: f64,
        check_options: Option<&CheckOptions>,
    ) -> Result<(), NeuralNetworkError> {
        let initial_cost = self
            .cost_training_set(&training_data)
            .map_err(|e| NeuralNetworkError::VectorDimensionMismatch(e))?;
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

            // Here I'll try to use my new structures to capture the forward pass info in a more obvious manner
            // forward_pass_data contains the forward pass data for each training example
            let forward_pass_data: Vec<ForwardAndBackPassData> = training_data
                .iter()
                .map(|tr_ex| {
                    let intermediates = self.feed_forward_capturing_intermediates(&tr_ex.input_v);
                    let error_vectors = self.backprop(&tr_ex.desired_output_v, &intermediates);

                    ForwardAndBackPassData {
                        intermediates,
                        error_vectors,
                    }
                })
                .collect();

            println!(
                "finished ff for all training points - epoch {}",
                epocs_count
            );

            // note: compute_gradients takes data for ALL training examples
            let mut gradients = self.compute_gradients(&forward_pass_data);

            // temp - print the gradients
            // let g1 = gradients.get(&1).unwrap();
            // println!("gradients for layer 1:");
            // println!("weights grad: \n{}", g1.0);
            // println!("biases grad: \n{}", g1.1);

            if check_options.gradient_checking {
                let approx_gradients_big_v = self
                    .approximate_cost_gradient(training_data)
                    .map_err(|e| NeuralNetworkError::VectorDimensionMismatch(e))?;
                // unroll the actual gradients
                let d_vec = gradients.unroll();

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

            // update weights and biases based on gradients and learning rate
            // TODO: extract to method for easy testing?
            for layer_index in 1..self.sizes.len() {
                let weights_grad = gradients.weights_at_layer_mut(layer_index)?;
                self.weights_at_layer_mut(layer_index)?
                    .subtract_mut(weights_grad.mult_scalar_mut_chain(learning_rate));

                let bias_grad = gradients.bias_at_layer_mut(layer_index)?;
                self.bias_at_layer_mut(layer_index)?
                    .subtract_mut(bias_grad.mult_scalar_mut_chain(learning_rate));
            }

            // Remove
            // Show weights and biases in l1
            let l1_weights = self.weights.get(&1).unwrap();
            println!("weights in layer 1: \n{}", l1_weights);
            let l1_biases = self.biases.get(&1).unwrap();
            println!("biases in layer 1: \n{}", l1_biases);

            if check_options.cost_decreasing_check {
                let cost = self
                    .cost_training_set(&training_data)
                    .map_err(|e| NeuralNetworkError::VectorDimensionMismatch(e))?;

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

        let final_cost = self
            .cost_training_set(&training_data)
            .map_err(|e| NeuralNetworkError::VectorDimensionMismatch(e))?;
        println!(
            "\ncost across entire training set after {} epocs: {}",
            epocs_count, final_cost,
        );

        Ok(())
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
    ) -> Result<(), NeuralNetworkError> {
        println!("computing initial cross accross entire training dataset...");
        let mut t_init_cost = SimpleTimer::start_new("t_init_cost");
        let initial_cost = self
            .cost_training_set(&training_data)
            .map_err(|e| NeuralNetworkError::VectorDimensionMismatch(e))?;
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
                    let intermediates = self.feed_forward_capturing_intermediates(&tr_ex.input_v);
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
            // TODO: actually, I think this comment is wrong - I think it only takes in the data for all the training examples in the current mini batch
            println!("computing gradients...");
            let mut t_compute_gradients = SimpleTimer::start_new("t_compute_gradients");
            let mut gradients = self.compute_gradients_par_6(&per_tr_ex_data);

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
                        weights_grad.mult_scalar_mut(optimizer_config.learning_rate);

                        let weights = self.weights.get_mut(&layer_index).unwrap();
                        weights.subtract_mut(&weights_grad);

                        let bias_grad = gradients.get_bias_vector_mut(&layer_index);
                        bias_grad.mult_scalar_mut(optimizer_config.learning_rate);
                        let biases = self.biases.get_mut(&layer_index).unwrap();
                        biases.subtract_mut(&bias_grad);
                    }
                }
                Optimizer::Momentum(optimizer_config) => {
                    for layer_index in 1..self.sizes.len() {
                        let weights_grad = gradients.get_weights_matrix_mut(&layer_index);
                        weights_grad.mult_scalar_mut(optimizer_config.learning_rate);

                        let m_w = momentum.get_weights_matrix_mut(&layer_index);
                        m_w.mult_scalar_mut(optimizer_config.momentum);
                        m_w.subtract_mut(&weights_grad);
                        let weights = self.weights.get_mut(&layer_index).unwrap();
                        weights.add_mut(&m_w);

                        let bias_grad = gradients.get_bias_vector_mut(&layer_index);
                        bias_grad.mult_scalar_mut(optimizer_config.learning_rate);
                        let m_b = momentum.get_bias_vector_mut(&layer_index);
                        m_b.mult_scalar_mut(optimizer_config.momentum);
                        m_b.subtract_mut(&bias_grad); // TODO: standardize the subtract_in_place / minus_in_place naming

                        let biases = self.biases.get_mut(&layer_index).unwrap();
                        biases.add_mut(&m_b);
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
                        weights.add_mut(&w);

                        let biases = self.biases.get_mut(&layer_index).unwrap();
                        let b = m_hat.get_bias_vector_mut(&layer_index);
                        biases.add_mut(&b);
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
                let cost = self
                    .cost_training_set(&training_data)
                    .map_err(|e| NeuralNetworkError::VectorDimensionMismatch(e))?;
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
                        "\nCost Update\ncomputing cost across entire training dataset after {} epocs...",
                        epochs_count
                    );
                    let training_set_cost = self
                        .cost_training_set(&training_data)
                        .map_err(|e| NeuralNetworkError::VectorDimensionMismatch(e))?;
                    println!(
                        "  - cost across entire training set after {} epocs: {}",
                        epochs_count, training_set_cost,
                    );

                    println!(
                        "computing cost across entire test dataset after {} epocs...",
                        epochs_count
                    );
                    let test_set_cost = self
                        .cost_training_set(&training_data)
                        .map_err(|e| NeuralNetworkError::VectorDimensionMismatch(e))?;
                    println!(
                        "  - cost across test set after {} epocs: {}",
                        epochs_count, training_set_cost,
                    );
                    maybe_test_set_cost = Some(training_set_cost);

                    if let Some(ref session_logger) = session_logger {
                        let epoch = epochs_count - 1;
                        _ = session_logger.write_update(
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
                        let test_set_cost = self
                            .cost_training_set(esc.test_data)
                            .map_err(|e| NeuralNetworkError::VectorDimensionMismatch(e))?;
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
        let final_cost = self
            .cost_training_set(&training_data)
            .map_err(|e| NeuralNetworkError::VectorDimensionMismatch(e))?;
        println!(
            "\ncost across entire training set after {} epocs: {}",
            epochs_count, final_cost,
        );

        Ok(())
    }

    //////////////////////////////////////////////
    // methods used for gradient checking
    // TODO: can these be refactored out of here?
    //////////////////////////////////////////////

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
        // gradients: &HashMap<LayerIndex, (Matrix, ColumnVector)>,
        gradients: &HashMap<LayerIndex, LayerGradients>,
    ) -> Vec<f64> {
        let mut unrolled_vec = Vec::new();
        for l in 1..self.num_layers() {
            let this_layer_gradients = gradients.get(&l).unwrap();
            unrolled_vec.extend_from_slice(this_layer_gradients.weight_gradients.data.as_slice()); // .0 is the weights matrix
            unrolled_vec.extend_from_slice(this_layer_gradients.bias_gradients.get_data_as_slice());
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
            let w_shape = self.weight_matrix_shape(l);
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
    fn approximate_cost_gradient(
        &self,
        training_data: &Vec<NDTrainingDataPoint>,
    ) -> Result<Vec<f64>, VectorDimensionMismatch> {
        let mut big_theta_v = self.unroll_weights_and_biases();
        let mut gradient = Vec::new();

        for i in 0..big_theta_v.len() {
            let orig_i_value = big_theta_v[i];

            big_theta_v[i] = orig_i_value + GRADIENT_CHECK_EPSILON;
            let temp_nn = self.reshape_weights_and_biases(&big_theta_v);
            let cost_plus_epsilon = temp_nn.cost_training_set(&training_data)?;

            big_theta_v[i] = orig_i_value - GRADIENT_CHECK_EPSILON;
            let temp_nn = self.reshape_weights_and_biases(&big_theta_v);
            let cost_minus_epsilon = temp_nn.cost_training_set(training_data)?;

            big_theta_v[i] = orig_i_value; // important - restore the orig value

            let approx_cost_fn_partial_derivative =
                (cost_plus_epsilon - cost_minus_epsilon) / GRADIENT_CHECK_TWICE_EPSILON;
            gradient.push(approx_cost_fn_partial_derivative);
        }

        Ok(gradient)
    }

    //////////////////////////////////////////////
    // The following I think are leftover code.
    // They have to do with computing the initial w/b values but don't appear to be used.
    //////////////////////////////////////////////

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
}

#[cfg(test)]
mod tests {
    use std::panic;

    use crate::activation::leaky_relu::LeakyReLU;
    use crate::builder::NeuralNetworkBuilder;

    use super::optimizer;
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
        layer_infos.insert(0, LayerConfig::new(None));
        layer_infos.insert(1, LayerConfig::new(Some(ActivationFunction::Sigmoid)));
        layer_infos.insert(2, LayerConfig::new(Some(ActivationFunction::Sigmoid)));

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
    fn test_rev_layer_indexs_computation() {
        let nn = get_simple_get_2_3_1_nn_for_test();

        let layers_in_from_last_to_1th: Vec<usize> = (1..nn.num_layers()).rev().collect();

        assert_eq!(layers_in_from_last_to_1th.len(), 2);
        assert_eq!(layers_in_from_last_to_1th[0], 2);
        assert_eq!(layers_in_from_last_to_1th[1], 1);
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
        layer_infos.insert(0, LayerConfig::new(None));
        for l in 1..sizes.len() {
            layer_infos.insert(l, LayerConfig::new(Some(ActivationFunction::Sigmoid)));
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
        layer_infos.insert(0, LayerConfig::new(None));
        for l in 1..sizes.len() {
            layer_infos.insert(l, LayerConfig::new(Some(ActivationFunction::Sigmoid)));
        }

        let nn = SimpleNeuralNetwork {
            sizes,
            weights,
            biases,
            layer_infos,
        };

        let intermediates = nn.feed_forward_capturing_intermediates(&inputs);
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
    pub fn test_cost_single_tr_ex_single_output_neuron() {
        let nn = get_simple_get_2_3_1_nn_for_test();

        // let's go for y(x0, x1) = x0 + x1;
        let tr_ex = NDTrainingDataPoint {
            input_v: column_vector![1.0, 1.0],
            desired_output_v: column_vector![3.0],
        };

        let c0 = nn.cost_single_tr_ex(&tr_ex).unwrap();

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

        let c0 = nn.cost_single_tr_ex(&tr_ex).unwrap();

        // manually compute the expected cost for the single neuron in the last layer
        // this uses a different method to compute the cost, but it is mathematically equivalent to that used
        // in cost_single_tr_ex
        let actual_output_vector = nn.feed_forward(&tr_ex.input_v);
        let diff_vec = &tr_ex.desired_output_v.subtract(&actual_output_vector);
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
        let nn = get_three_layer_multiple_output_nn_for_test();

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

        let c = nn.cost_training_set(&training_data).unwrap();
        let c0 = nn.cost_single_tr_ex(&training_data[0]).unwrap();
        let c1 = nn.cost_single_tr_ex(&training_data[1]).unwrap();
        let c2 = nn.cost_single_tr_ex(&training_data[2]).unwrap();
        let c_avg = (c0 + c1 + c2) / 3.0;
        assert_eq!(c, c_avg);

        // now do the same cost test with big_theta_v
        let big_theta_v = nn.unroll_weights_and_biases();
        let reshaped = nn.reshape_weights_and_biases(&big_theta_v);
        let cost_big_theta_v = reshaped.cost_training_set(&training_data).unwrap();
        assert_eq!(cost_big_theta_v, c);

        // from here on, I'm testing the towards_backprop stuff
        println!("\nIn the test - doing the pre backprop stuff: {}", c0);
        let mut error_vectors_for_each_training_example = Vec::new();
        let mut intermediates_for_each_training_example = Vec::new();

        for i_tr_ex in 0..training_data.len() {
            println!("\nfor the {}th training example", i_tr_ex);
            let inputs_v = &training_data[i_tr_ex].input_v;
            let desired_output_v = &training_data[i_tr_ex].desired_output_v;
            let intermediates = nn.feed_forward_capturing_intermediates(&inputs_v);
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
        )
        .unwrap();

        // predict
        let tr_ex = NDTrainingDataPoint {
            input_v: column_vector![2.0, 2.0],
            desired_output_v: column_vector![1.0],
        };

        let predicted_output_0 = nn.feed_forward(&tr_ex.input_v).into_value();
        println!("predicted_output_0: {}", &predicted_output_0);
        let cost_of_predicted_0 = nn.cost_single_tr_ex(&tr_ex).unwrap();
        println!("cost_of_predicted_0: \n{}", &cost_of_predicted_0);

        // predict
        let tr_ex = NDTrainingDataPoint {
            input_v: column_vector![-2.0, -2.0],
            desired_output_v: column_vector![0.0],
        };

        let predicted_output_1 = nn.feed_forward(&tr_ex.input_v).into_value();
        println!("predicted_output_1: {}", &predicted_output_1);
        let cost_of_predicted_1 = nn.cost_single_tr_ex(&tr_ex).unwrap();
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
            println!("layer {} w:", l);
            println!("{}", w);
        }

        println!("initial biases:");
        for (l, b) in nn.biases.iter() {
            println!("layer {} b:", l);
            println!("{}", b);
        }

        let epocs = 7000;
        let learning_rate = 0.9;

        let check_options = CheckOptions {
            gradient_checking: false,
            cost_decreasing_check: true,
        };

        nn.train(&training_data, epocs, learning_rate, Some(&check_options))
            .unwrap();

        // predict
        let tr_ex = NDTrainingDataPoint {
            input_v: column_vector![2.0, 2.0],
            desired_output_v: column_vector![1.0],
        };

        let predicted_output_0 = nn.feed_forward(&tr_ex.input_v).into_value();
        println!("predicted_output_0: {}", &predicted_output_0);
        let cost_of_predicted_0 = nn.cost_single_tr_ex(&tr_ex).unwrap();
        println!("cost_of_predicted_0: \n{}", &cost_of_predicted_0);

        // predict
        println!("second prediction");
        let tr_ex = NDTrainingDataPoint {
            input_v: column_vector![-2.0, -2.0],
            desired_output_v: column_vector![0.0],
        };
        let predicted_output_1 = nn.feed_forward(&tr_ex.input_v).into_value();
        println!("predicted_output_1: {}", &predicted_output_1);
        let cost_of_predicted_1 = nn.cost_single_tr_ex(&tr_ex).unwrap();
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

        // TODO: why am I training twice here - should this first one be removed?
        nn.train(&training_data, 250, 0.9, Some(&check_options))
            .unwrap();

        nn.train_stochastic(
            &training_data,
            1000,
            &Optimizer::standard_gradient_descent(0.9),
            training_data.len(),
            Some(&check_options),
            None,
            None,
            None,
        )
        .unwrap();

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
        let cost_of_predicted_0 = nn.cost_single_tr_ex(&tr_ex).unwrap();
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
        let cost_of_predicted_1 = nn.cost_single_tr_ex(&tr_ex).unwrap();
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

        // TODO: why training twice? should one be removed?
        nn.train(&training_data, 250, 0.9, Some(&check_options))
            .unwrap();

        nn.train_stochastic(
            &training_data,
            1000,
            &Optimizer::standard_gradient_descent(0.9),
            training_data.len(),
            Some(&check_options),
            None,
            None,
            None,
        )
        .unwrap();

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
        let cost_of_predicted_0 = nn.cost_single_tr_ex(&tr_ex).unwrap();
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
        let cost_of_predicted_1 = nn.cost_single_tr_ex(&tr_ex).unwrap();
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

        // TODO: why training twice? should one be removed?
        nn.train(&training_data, 250, 0.9, Some(&check_options))
            .unwrap();

        nn.train_stochastic(
            &training_data,
            1000,
            &Optimizer::standard_gradient_descent(0.9),
            training_data.len(),
            Some(&check_options),
            None,
            None,
            None,
        )
        .unwrap();

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
        let cost_of_predicted_0 = nn.cost_single_tr_ex(&tr_ex).unwrap();
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
        let cost_of_predicted_1 = nn.cost_single_tr_ex(&tr_ex).unwrap();
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

        nn.train(&training_data, epocs, learning_rate, Some(&check_options))
            .unwrap();

        // predict
        let tr_ex = NDTrainingDataPoint {
            input_v: column_vector![2.0, 2.0],
            desired_output_v: column_vector![1.0],
        };
        let predicted_output_0 = nn.feed_forward(&tr_ex.input_v).into_value();
        println!("predicted_output_0: {}", &predicted_output_0);
        let cost_of_predicted_0 = nn.cost_single_tr_ex(&tr_ex).unwrap();
        println!("cost_of_predicted_0: \n{}", &cost_of_predicted_0);

        // predict
        println!("second prediction");
        let tr_ex = NDTrainingDataPoint {
            input_v: column_vector![-2.0, -2.0],
            desired_output_v: column_vector![0.0],
        };
        let predicted_output_1 = nn.feed_forward(&tr_ex.input_v).into_value();
        println!("predicted_output_1: {}", &predicted_output_1);
        let cost_of_predicted_1 = nn.cost_single_tr_ex(&tr_ex).unwrap();
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
        )
        .unwrap();

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
        let cost_of_predicted_1 = nn.cost_single_tr_ex(&tr_ex).unwrap();
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
        let cost_of_predicted_2 = nn.cost_single_tr_ex(&tr_ex).unwrap();
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
        )
        .unwrap();

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
        let cost_of_predicted_1 = nn.cost_single_tr_ex(&tr_ex).unwrap();
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
        let cost_of_predicted_2 = nn.cost_single_tr_ex(&tr_ex).unwrap();
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
            &Optimizer::Adam(optimizer::AdamConfig::with_learning_rate(0.01)),
            mini_batch_size,
            Some(&check_options),
            Some(early_stop_config),
            None,
            None,
        )
        .unwrap();

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
        let cost_of_predicted_1 = nn.cost_single_tr_ex(&tr_ex).unwrap();
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
        let cost_of_predicted_2 = nn.cost_single_tr_ex(&tr_ex).unwrap();
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
        let weight_matrix_l1_shape = nn.weight_matrix_shape(1);
        let weight_matrix_l2_shape = nn.weight_matrix_shape(2);
        assert_eq!(weight_matrix_l1_shape.rows, 3);
        assert_eq!(weight_matrix_l1_shape.columns, 2);
        assert_eq!(weight_matrix_l2_shape.rows, 2);
        assert_eq!(weight_matrix_l2_shape.columns, 3);
    }

    #[test]
    fn test_unroll_weights_and_biases() {
        let mut nn = SimpleNeuralNetwork::new(vec![2, 3, 2]);
        let weight_matrix_l1_shape = nn.weight_matrix_shape(1);
        let weight_matrix_l2_shape = nn.weight_matrix_shape(2);

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
        let weight_matrix_l1_shape = nn.weight_matrix_shape(1);
        let weight_matrix_l2_shape = nn.weight_matrix_shape(2);

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

        gradients.insert(1, LayerGradients::new(w1, b1));
        gradients.insert(2, LayerGradients::new(w2, b2));

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
