use crate::LayerIndex;
use common::column_vec_of_random_values_from_distribution;
use common::linalg::{ColumnVector, Matrix};

pub enum Initializer {
    RandomBasic,
    Xavier,
    XavierNormalized,
    XavierNormalHOMLForSigmoid,
    HeForReLUAndVariants,
    // Random(f64, f64),
    Manual(Matrix, ColumnVector),
}

pub fn get_init_weights_and_biases(
    l: LayerIndex,
    sizes: &[usize],
    initializer: Initializer,
) -> (Matrix, ColumnVector) {
    if l == 0 {
        panic!("not valid for input layer");
    }

    match initializer {
        Initializer::RandomBasic => {
            let weights_m = Matrix::new_matrix_with_random_values_from_normal_distribution(
                sizes[l],
                sizes[l - 1],
                0.0,
                1.0,
            );

            let bias_v = column_vec_of_random_values_from_distribution(0.0, 1.0, sizes[l]);

            (weights_m, bias_v)
        }
        Initializer::Manual(weights_m, bias_v) => (weights_m, bias_v),
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

            (weights_m, bias_v)
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

            (weights_m, bias_v)
        }
        Initializer::XavierNormalHOMLForSigmoid => {
            let fan_in = sizes[l - 1];
            let fan_out = sizes[l];
            let fan_avg = (fan_in as f64 + fan_out as f64) / 2.0;
            let std_dev = (1.0_f64 / fan_avg).sqrt(); // See Table 11-1 in HOML

            println!("XavierNormalHOMLForSigmoid: fan_in: {}", fan_in);
            println!("XavierNormalHOMLForSigmoid: fan_out: {}", fan_out);
            println!("XavierNormalHOMLForSigmoid: fan_avg: {}", fan_avg);
            println!("XavierNormalHOMLForSigmoid: using std dev {}", std_dev);

            let weights_m = Matrix::new_matrix_with_random_values_from_normal_distribution(
                sizes[l],
                sizes[l - 1],
                0.0,
                std_dev,
            );

            let bias_v = ColumnVector::new_zero_vector(sizes[l]);

            (weights_m, bias_v)
        }
        Initializer::HeForReLUAndVariants => {
            let fan_in = sizes[l - 1];
            let std_dev = (2.0_f64 / fan_in as f64).sqrt(); // See Table 11-1 in HOML

            println!("HeForReLUAndVariants: fan_in: {}", fan_in);
            println!("HeForReLUAndVariants: using std dev {}", std_dev);

            let weights_m = Matrix::new_matrix_with_random_values_from_normal_distribution(
                sizes[l],
                sizes[l - 1],
                0.0,
                std_dev,
            );

            let bias_v = ColumnVector::new_zero_vector(sizes[l]);

            (weights_m, bias_v)
        }
    }
}
