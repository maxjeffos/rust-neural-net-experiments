use rand;

pub mod point;

pub enum DistributionType {
    Uniform,
    Gaussian,
    // Cauchy,
    // Poisson,
}

pub enum Distribution {
    Uniform(f64, f64),
    Gaussian(f64, f64),
    // Cauchy(f64, f64),
    // Poisson(f64),
}

// pub fn random_in_range(min: f64, max: f64) -> f64 {
//     rand::random::<f64>() * (max - min) + min
// }

// pub fn random_in_range_with_noise(min: f64, max: f64, noise_type: DistributionType) -> f64 {
//     match noise_type {
//         DistributionType::Linear => random_in_range(min, max),
//         DistributionType::Gaussian => unimplemented!(),
//     }
// }
