use common::linalg::ColumnVector;

pub mod leaky_relu;
pub mod relu;
pub mod sigmoid;
pub mod softmax;

#[derive(Debug, Clone, PartialEq)]
pub enum ActivationFunction {
    Sigmoid,
    ReLU,
    LeakyReLU(f64),
    Softmax,
}

impl ActivationFunction {
    // pub fn get_activator(&self) -> Box<dyn VectorActivator> {
    //     match self {
    //         ActivationFunction::Sigmoid(s) => Box::new(s.clone()),
    //         ActivationFunction::ReLU(r) => Box::new(r.clone()),
    //         ActivationFunction::LeakyReLU(l) => Box::new(l.clone()),
    //         ActivationFunction::Softmax(sm) => Box::new(sm.clone()),
    //     }
    // }
}

impl VectorActivator for ActivationFunction {
    fn activate_vector(&self, z: &ColumnVector) -> ColumnVector {
        match self {
            ActivationFunction::Sigmoid => {
                let data = z.iter()
                    .map(|z| sigmoid::activate(*z))
                    .collect::<Vec<f64>>();
                ColumnVector::from_vec(data)
            },
            ActivationFunction::ReLU => {
                let data = z.iter()
                    .map(|z| relu::activate(*z))
                    .collect::<Vec<f64>>();
                ColumnVector::from_vec(data)
            },
            ActivationFunction::LeakyReLU(tail_slope) => {
                let data = z.iter()
                    .map(|z| leaky_relu::activate(*z, *tail_slope))
                    .collect::<Vec<f64>>();
                ColumnVector::from_vec(data)
            },
            ActivationFunction::Softmax => {
                softmax::activate_vector(z)
            },
        }
    }

    fn activate_derivative_vector(&self, z: &ColumnVector) -> ColumnVector {
        match self {
            ActivationFunction::Sigmoid => {
                let data = z.iter()
                    .map(|z| sigmoid::activate_derivative(*z))
                    .collect::<Vec<f64>>();
                ColumnVector::from_vec(data)
            },
            ActivationFunction::ReLU => {
                let data = z.iter()
                    .map(|z| relu::activate_derivative(*z))
                    .collect::<Vec<f64>>();
                ColumnVector::from_vec(data)
            },
            ActivationFunction::LeakyReLU(tail_slope) => {
                let data = z.iter()
                    .map(|z| leaky_relu::activate_derivative(*z, *tail_slope))
                    .collect::<Vec<f64>>();
                ColumnVector::from_vec(data)
            },
            ActivationFunction::Softmax => {
                // the main reason being that it produces a Matrix (the Jacobian Matrix) not a Vector
                // so it doesn't fit with the design, but also, we don't need it as of now.
                unimplemented!("activate_derivative is not implemented for softmax.")
            },
        } 
    }
}


pub trait VectorActivator {
    fn activate_vector(&self, z: &ColumnVector) -> ColumnVector;
    fn activate_derivative_vector(&self, z: &ColumnVector) -> ColumnVector;
}


