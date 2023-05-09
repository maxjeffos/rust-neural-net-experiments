use activation::ActivationFunction;
use metrics::SimpleTimer;

use test6_nn::{CheckOptions, EarlyStopConfig};

use test6_nn::activation;
use test6_nn::builder::NeuralNetworkBuilder;
use test6_nn::initializer::Initializer;
use test6_nn::optimizer::{AdamConfig, Optimizer};
use test6_nn::training_log::TrainingSessionLogger;

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

    let t0_cost = nn.cost_single_tr_ex(t0).unwrap();
    println!("t0_cost: {}", t0_cost);

    for i_test in 0..50 {
        let t = test_data.get(i_test).unwrap();
        let cost = nn.cost_single_tr_ex(t).unwrap();
        println!("cost of {}th: {}", i_test, cost);
    }

    // let sub_tr_set = &test_data[0..1000];
    // println!("computing cost of a sub-set of the test data...");
    let test_set_cost = nn.cost_training_set(&test_data).unwrap();
    println!("\ntest_set_cost: {}", test_set_cost);

    t_total.stop();
    println!("\nt_total: {}", t_total);
}
