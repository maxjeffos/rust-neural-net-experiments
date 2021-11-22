// TODO:
// add a stop on convergence mechanism
// at start of train(), check that the inputs are valid and throw errors as required

use common::dot_product;
use common::scalar_valued_multivariable_point::ScalarValuedMultivariablePoint;

fn main() {
    let data = vec![
        ScalarValuedMultivariablePoint::new_2d(0.0, 0.0),
        ScalarValuedMultivariablePoint::new_2d(2.0, 2.0),
        ScalarValuedMultivariablePoint::new_2d(4.0, 4.0),
        ScalarValuedMultivariablePoint::new_2d(6.0, 6.0),
        ScalarValuedMultivariablePoint::new_2d(3.0, 3.0),
        ScalarValuedMultivariablePoint::new_2d(5.0, 5.0),
    ];

    data.iter().for_each(|p| {
        println!("{:?}", p);
    });
    println!();

    let theta_init = vec![1.0, 0.0]; // ideal line for this data is (0.0, 1.0)
    let epocs = 200;
    let learning_rate = 0.1;

    let theta_final = train(&data, &theta_init, epocs, learning_rate);

    println!("theta0: {}", theta_final[0]);
    println!("theta1: {}\n", theta_final[1]);

    println!("final cost: {}", cost(&theta_final, &data));
}

#[derive(Debug, PartialEq)]
enum MultivariableRegressionError {
    InvalidData(InvalidDataError),
    InvalidEpocs,
    InvalidLearningRate,
}

#[derive(Debug, PartialEq)]
enum InvalidDataError {
    NotEnoughData,
    NonMatichgDimensions,
}

fn train(
    data: &[ScalarValuedMultivariablePoint],
    theta_init: &[f64],
    epocs: usize,
    learning_rate: f64,
) -> Vec<f64> {
    let mut theta = Vec::from(theta_init);

    // ensure data has some data points otherwise return error
    if data.len() < 2 {
        panic!("data must have at least 2 points");
    }

    // ensure that thata_init and data have the same dimensions otherwise return error
    let theta_dim = theta_init.len();
    let data_dim = data[0].dimension();
    if theta_dim != data_dim {
        panic!("theta_init and data must have the same dimensions");
    }

    let mut iteration = 0;

    loop {
        let cost_fn_partial_derivatives = compute_gradient_vector(&theta, data);

        for i in 0..theta_dim {
            theta[i] = theta[i] - learning_rate * cost_fn_partial_derivatives[i];
        }

        iteration += 1;
        if iteration >= epocs {
            println!("max iterations reached: {}", iteration);
            break;
        }
    }

    theta
}

fn predict(theta: &[f64], independant: &[f64]) -> f64 {
    theta[0] + dot_product(&theta[1..], independant).expect("dot product failed")
}

/// Computes the cost function for the given theta and data
/// # Arguments
/// * `theta` - the current theta values
/// * `data` - the data points
/// # Returns
/// the cost function value
/// # Errors
/// returns an error if the data is not valid
/// # Examples
/// ```
/// use common::scalar_valued_multivariable_point::ScalarValuedMultivariablePoint;
/// use common::multivariable_regression::cost;
/// let data = vec![
///     ScalarValuedMultivariablePoint::new_2d(0.0, 0.0),
///     ScalarValuedMultivariablePoint::new_2d(2.0, 2.0),
///     ScalarValuedMultivariablePoint::new_2d(4.0, 4.0),
///     ScalarValuedMultivariablePoint::new_2d(6.0, 6.0),
///     ScalarValuedMultivariablePoint::new_2d(3.0, 3.0),
///     ScalarValuedMultivariablePoint::new_2d(5.0, 5.0),
/// ];
/// let theta = vec![1.0, 0.0];
/// assert_eq!(cost(&theta, &data), 0.0);
/// ```
fn compute_gradient_vector(theta: &[f64], data: &[ScalarValuedMultivariablePoint]) -> Vec<f64> {
    if data.len() < 2 {
        panic!("data must have at least 2 points");
    }

    if theta.len() != data[0].independant.len() + 1 {
        println!("dimention of theta: {}", theta.len());
        println!("dimention of independant: {}", data[0].independant.len());
        panic!("length of theta must be one more than the length of the independant variable vector. dim theta: {}, dim independant: {}", 0, 0);
    }

    let mut sums = vec![0_f64; theta.len()];
    let m = data.len();

    data.iter().for_each(|p| {
        let independent = &p.independant;
        let dependant = p.dependant;

        let hyp = hypothesis(theta, independent);

        for i in 0..theta.len() {
            let sum_i = if i == 0 {
                hyp - dependant
            } else {
                (hyp - dependant) * independent[i - 1]
            };
            sums[i] += sum_i;
        }
    });

    for i in 0..sums.len() {
        sums[i] /= m as f64;
    }

    sums
}

fn hypothesis(theta: &[f64], independant: &[f64]) -> f64 {
    if theta.len() != independant.len() + 1 {
        println!("dimention of theta: {}", theta.len());
        println!("dimention of independant: {}", independant.len());
        panic!("length of theta must be one more than the length of the independant variable vector. dim theta: {}, dim independant: {}", 0, 0);
    }

    let mut hyp = theta[0];
    for i in 0..independant.len() {
        hyp += theta[i + 1] * independant[i];
    }

    hyp
}

fn cost(theta: &[f64], data: &[ScalarValuedMultivariablePoint]) -> f64 {
    let mut cost = 0.0;
    for point in data {
        let hyp = hypothesis(theta, &point.independant);
        let diff = hyp - point.dependant;
        let diff_squared = diff * diff;
        cost += diff_squared;
    }
    cost / (2.0 * data.len() as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use float_cmp::approx_eq;

    #[test]
    fn hypothesis_v_works() {
        let theta = vec![1_f64, 2_f64];
        let h = hypothesis(&theta, &[5_f64]);
        assert_eq!(h, 11_f64);
    }

    #[test]
    fn compute_partial_derivatives_v_works() {
        // fn compute_partial_derivatives(theta: &[f64], data: &[ScalarValuedMultivariablePoint]) -> (f64, f64) {

        let theta = vec![1.0, 2.0];
        let data = vec![
            ScalarValuedMultivariablePoint::new_2d(0.0, 0.0),
            ScalarValuedMultivariablePoint::new_2d(2.0, 2.0),
            ScalarValuedMultivariablePoint::new_2d(4.0, 4.0),
            ScalarValuedMultivariablePoint::new_2d(6.0, 6.0),
            ScalarValuedMultivariablePoint::new_2d(3.0, 3.0),
            ScalarValuedMultivariablePoint::new_2d(5.0, 5.0),
        ];

        let partials = compute_gradient_vector(&theta, &data);

        println!("partial[0]: {}", partials[0]);
        println!("partial[1]: {}", partials[1]);

        assert_eq!(partials[0], 4.333333333333333);
        assert_eq!(partials[1], 18.333333333333332);
    }

    #[test]
    fn cost_fn_works_for_zero_cost() {
        let theta = vec![0.0, 1.0];

        // all these points fall on the line y = 0.0 + 1.0x
        let data = vec![
            ScalarValuedMultivariablePoint::new_2d(0.0, 0.0),
            ScalarValuedMultivariablePoint::new_2d(1.0, 1.0),
            ScalarValuedMultivariablePoint::new_2d(2.0, 2.0),
            ScalarValuedMultivariablePoint::new_2d(3.0, 3.0),
            ScalarValuedMultivariablePoint::new_2d(4.0, 4.0),
        ];

        let cost = cost(&theta, &data);
        assert_eq!(cost, 0_f64);
    }

    #[test]
    fn cost_fn_works_for_non_zero_cost() {
        let theta = vec![1.0, 2.0];

        // all these points have some error wrt the line y = 1.0 + 2.0x
        let data = vec![
            ScalarValuedMultivariablePoint::new_2d(0.0, 0.0), // y(0.0) should be 1.0 ; error = 1.0
            ScalarValuedMultivariablePoint::new_2d(1.0, 1.0), // y(1.0) should be 3.0 ; error = 2.0
            ScalarValuedMultivariablePoint::new_2d(2.0, 2.0), // y(2.0) should be 5.0 ; error = 3.0
            ScalarValuedMultivariablePoint::new_2d(3.0, 3.0), // y(3.0) should be 7.0 ; error = 4.0
        ];

        let cost = cost(&theta, &data);
        assert_eq!(cost, 3.75);
    }

    #[test]
    fn it_yields_the_correct_result_for_2d() {
        let data = vec![
            ScalarValuedMultivariablePoint::new_2d(0.0, 0.0),
            ScalarValuedMultivariablePoint::new_2d(2.0, 2.0),
            ScalarValuedMultivariablePoint::new_2d(4.0, 4.0),
            ScalarValuedMultivariablePoint::new_2d(6.0, 6.0),
            ScalarValuedMultivariablePoint::new_2d(3.0, 3.0),
            ScalarValuedMultivariablePoint::new_2d(5.0, 5.0),
        ];

        let theta_init = [1.0, 0.0]; // ideal line for this data is (0.0, 1.0)

        let epocs = 200;
        let learning_rate = 0.1;

        let theta_final = train(&data, &theta_init, epocs, learning_rate);

        assert_eq!(theta_final[0], 0.007864997397684767);
        assert_eq!(theta_final[1], 0.9982229772190276);

        let h = predict(&theta_final, &[10.0]);
        assert!(approx_eq!(f64, h, 10.0, epsilon = 0.01));
    }

    #[test]
    fn it_yields_the_correct_result_for_3d_ex1() {
        // all these points appear on the line y = f(x0, x1) = x0 + x1
        let data = vec![
            ScalarValuedMultivariablePoint::new_3d(0.0, 0.0, 0.0),
            ScalarValuedMultivariablePoint::new_3d(1.0, 1.0, 2.0),
            ScalarValuedMultivariablePoint::new_3d(2.0, 2.0, 4.0),
            ScalarValuedMultivariablePoint::new_3d(3.0, 3.0, 6.0),
        ];

        let theta_init = [1.0, 0.0, 0.0]; // ideal line for this data is (0.0, 1.0, 1.0)

        let epocs = 1200;
        let learning_rate = 0.1;

        let theta_final = train(&data, &theta_init, epocs, learning_rate);
        println!("theta_final: {:?}", theta_final);

        assert_eq!(theta_final[0], 0.00000000000000002942779689818007);
        assert_eq!(theta_final[1], 1.0);
        assert_eq!(theta_final[2], 1.0);

        let h = predict(&theta_final, &[7.0, 11.0]);
        assert!(approx_eq!(f64, h, 18.0, epsilon = 0.000001));
    }

    #[test]
    fn it_yields_the_correct_result_for_3d_ex2() {
        // all these points appear on the line y = f(x0, x1) = 2x0 + 3x1
        let data = vec![
            ScalarValuedMultivariablePoint::new_3d(0.0, 0.0, 0.0),
            ScalarValuedMultivariablePoint::new_3d(1.0, 1.0, 5.0),
            ScalarValuedMultivariablePoint::new_3d(2.0, 2.0, 10.0),
            ScalarValuedMultivariablePoint::new_3d(3.0, 3.0, 15.0),
            ScalarValuedMultivariablePoint::new_3d(4.5, 7.0, 30.0),
        ];

        let theta_init = [1.0, 0.0, 0.0]; // ideal line for this data is (0.0, 1.0, 1.0)

        let epocs = 2000;
        let learning_rate = 0.1;

        let theta_final = train(&data, &theta_init, epocs, learning_rate);
        println!("theta_final: {:?}", theta_final);

        assert!(approx_eq!(f64, theta_final[0], 0.0, epsilon = 0.000001));
        assert!(approx_eq!(f64, theta_final[1], 2.0, epsilon = 0.000001));
        assert!(approx_eq!(f64, theta_final[2], 3.0, epsilon = 0.000001));

        let h = predict(&theta_final, &[7.0, 11.0]);
        println!("h: {}", h);
        assert!(approx_eq!(f64, h, 47.0, epsilon = 0.000001));
    }
}
