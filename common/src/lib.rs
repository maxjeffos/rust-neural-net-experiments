use rand;
use rand::Rng;

pub mod datapoints;
pub mod linalg;
pub mod old_matrix;
pub mod point;
pub mod scalar_valued_multivariable_point;
use rand::distributions::Distribution;
use rand_distr::Normal;

use linalg::{ColumnVector, Matrix};

// pub enum DistributionType {
//     Uniform,
//     Gaussian,
//     // Cauchy,
//     // Poisson,
// }

// pub enum Distribution {
//     Uniform(f64, f64),
//     Gaussian(f64, f64),
//     // Cauchy(f64, f64),
//     // Poisson(f64),
// }

// pub fn random_in_range(min: f64, max: f64) -> f64 {
//     rand::random::<f64>() * (max - min) + min
// }

// pub fn random_in_range_with_noise(min: f64, max: f64, noise_type: DistributionType) -> f64 {
//     match noise_type {
//         DistributionType::Linear => random_in_range(min, max),
//         DistributionType::Gaussian => unimplemented!(),
//     }
// }

#[derive(Debug, PartialEq)]
pub enum DotProductError {
    DimensionsDoNotMatch,
    DimensionIsZero,
}

pub fn dot_product(v1: &[f64], v2: &[f64]) -> Result<f64, DotProductError> {
    if v1.len() != v2.len() {
        println!("dimention of v1: {}", v1.len());
        println!("dimention of v2: {}", v2.len());
        println!("v1 and v2 must have the same dimension");
        return Err(DotProductError::DimensionsDoNotMatch);
    }

    if v1.len() == 0 {
        println!("v1 and v2 must have a non-zero dimension");
        return Err(DotProductError::DimensionIsZero);
    }

    let mut sum = 0_f64;

    for i in 0..v1.len() {
        sum += v1[i] * v2[i];
    }

    Ok(sum)
}

pub fn column_vec_of_random_values(min: f64, max: f64, size: usize) -> Matrix {
    let mut rng = rand::thread_rng();

    let mut values = Vec::new();
    for _ in 0..size {
        let x = rng.gen_range(min..max);
        values.push(x);
    }
    Matrix::new_column_vector(&values)
}

pub fn column_vec_of_random_values_from_distribution(
    mean: f64,
    std_dev: f64,
    size: usize,
) -> ColumnVector {
    let mut rng = rand::thread_rng();
    let normal = Normal::new(mean, std_dev).unwrap();

    let mut res = ColumnVector::empty();
    for _ in 0..size {
        let x = normal.sample(&mut rng);
        res.push(x);
    }
    res
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_product_works() {
        let v1 = vec![0.0, 1.0, 3.0];
        let v3 = vec![7.0, 11.0, 13.0];
        let dp = dot_product(&v1, &v3).unwrap();
        assert_eq!(dp, 50.0);
    }

    #[test]
    fn dot_returns_err_if_dimentions_are_zero() {
        let v1 = vec![];
        let v3 = vec![];
        let dp = dot_product(&v1, &v3);
        let dp_err = dp.err();
        assert_eq!(dp_err, Some(DotProductError::DimensionIsZero));
    }
}
