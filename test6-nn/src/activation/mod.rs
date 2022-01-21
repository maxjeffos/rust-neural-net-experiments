use common::linalg::ColumnVector;

pub mod activator;
pub mod elu;
pub mod jelu;
pub mod leaky_relu;
pub mod relu;
pub mod sigmoid;

#[derive(Debug, Clone)]
pub enum ActivationFunction {
    Sigmoid,
    ReLU,
    ELU,
    JELU(jelu::JELU),
    LeakyReLU(leaky_relu::LeakyReLU),
}

pub fn activate_vector(
    z_v: &ColumnVector,
    activation_function: &ActivationFunction,
) -> ColumnVector {
    if z_v.num_elements() == 0 {
        panic!("this function is only valid for column vectors with at least one element (row)")
    }

    let data = match activation_function {
        ActivationFunction::Sigmoid => z_v
            .iter()
            .map(|z| sigmoid::activate(*z))
            .collect::<Vec<f64>>(),
        ActivationFunction::ReLU => z_v.iter().map(|z| relu::activate(*z)).collect::<Vec<f64>>(),
        ActivationFunction::ELU => z_v.iter().map(|z| elu::activate(*z)).collect::<Vec<f64>>(),
        ActivationFunction::JELU(j) => z_v.iter().map(|z| j.activate(*z)).collect::<Vec<f64>>(),
        ActivationFunction::LeakyReLU(leaky_relu) => z_v
            .iter()
            .map(|z| leaky_relu.activate(*z))
            .collect::<Vec<f64>>(),
    };

    ColumnVector::from_vec(data)
}

pub fn activate_derivative_vector(
    z_v: &ColumnVector,
    activation_function: &ActivationFunction,
) -> ColumnVector {
    if z_v.num_elements() == 0 {
        panic!("this function is only valid for column vectors with at least one element (row)")
    }

    let data = match activation_function {
        ActivationFunction::Sigmoid => z_v
            .iter()
            .map(|z| sigmoid::activate_derivative(*z))
            .collect::<Vec<f64>>(),
        ActivationFunction::ReLU => z_v
            .iter()
            .map(|z| relu::activate_derivative(*z))
            .collect::<Vec<f64>>(),
        ActivationFunction::ELU => z_v
            .iter()
            .map(|z| elu::activate_derivative(*z))
            .collect::<Vec<f64>>(),
        ActivationFunction::JELU(j) => z_v
            .iter()
            .map(|z| j.activate_derivative(*z))
            .collect::<Vec<f64>>(),
        ActivationFunction::LeakyReLU(leaky_relu) => z_v
            .iter()
            .map(|z| leaky_relu.activate_derivative(*z))
            .collect::<Vec<f64>>(),
    };

    ColumnVector::from_vec(data)
}

// pub trait Activator {
//     fn activate(&self, z: f64) -> f64;
//     fn activate_derivative(&self, z: f64) -> f64;
// }

// impl Debug for dyn Activator {
//     fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
//         write!(f, "Activator")
//     }
// }

// pub fn activate_vector<A>(z_v: &ColumnVector, activator: &A) -> ColumnVector
// where
//     A: Activator,
// {
//     if z_v.num_elements() == 0 {
//         panic!("this function is only valid for column vectors with at least one element (row)")
//     }
//     let data = z_v
//         .iter()
//         .map(|z| activator.activate(*z))
//         .collect::<Vec<f64>>();

//     ColumnVector::from_vec(data)
// }

// pub fn activate_derivative_vector<A>(z_v: &ColumnVector, activator: &A) -> ColumnVector
// where
//     A: Activator,
// {
//     if z_v.num_elements() == 0 {
//         panic!("this function is only valid for column vectors with at least one element (row)")
//     }
//     let data = z_v
//         .iter()
//         .map(|z| activator.activate_derivative(*z))
//         .collect::<Vec<f64>>();

//     ColumnVector::from_vec(data)
// }

#[cfg(test)]
mod tests {
    use super::*;
    use common::column_vector;

    // // Used for test only

    // #[derive(Debug, Clone)]
    // struct FakeActivator {}

    // impl Activator for FakeActivator {
    //     fn activate(&self, z: f64) -> f64 {
    //         2.0 * z
    //     }

    //     fn activate_derivative(&self, z: f64) -> f64 {
    //         3.0 * z // Note: this is not actually the derivative of this fake activation function - it's for testing
    //     }
    // }

    // #[test]
    // pub fn activate_vector_works() {
    //     let z_v = column_vector![1.0, 2.0, 3.0];
    //     let activator = FakeActivator {};

    //     let activate_res_v = activate_vector(&z_v, &activator);
    //     assert_eq!(activate_res_v, column_vector![2.0, 4.0, 6.0]);
    // }

    // #[test]
    // pub fn activate_derivative_vector_works() {
    //     let z_v = column_vector![1.0, 2.0, 3.0];
    //     let activator = FakeActivator {};

    //     let activate_derivative_res_v = activate_derivative_vector(&z_v, &activator);
    //     assert_eq!(activate_derivative_res_v, column_vector![3.0, 6.0, 9.0]);
    // }
}
