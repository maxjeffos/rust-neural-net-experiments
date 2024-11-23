use crate::errors::VectorDimensionMismatch;
use common::linalg::{square, ColumnVector};

pub struct CrossEntropyCost {}

impl CrossEntropyCost {
    pub fn new() -> CrossEntropyCost {
        CrossEntropyCost {}
    }
}

impl Coster for CrossEntropyCost {
    fn cost(
        &self, 
        desired_v: &ColumnVector,
        actual_v: &ColumnVector,
    ) -> Result<f64, VectorDimensionMismatch> {
        if desired_v.num_elements() != actual_v.num_elements() {
            return Err(VectorDimensionMismatch::new_with_msg(
                desired_v.num_elements(),
                actual_v.num_elements(),
                "expected and actual outputs must have the same length",
            ));
        }

        // Using epsilon to avoid log(0) which is undefined
        let epsilon = 1e-10;

        let mut total_loss: f32 = 0.0;
        for (&act, &des) in actual_v.iter().zip(desired_v.iter()) {
            // Using epsilon to avoid log(0) which is undefined
            let pred = pred.max(epsilon).min(1.0 - epsilon);

            // Compute the loss for current element
            let loss = act * pred.log10() + (1.0 - act) * (1.0 - pred).log10();
            total_loss -= loss; // Sum up the negative log likelihood
        }

        Ok(total_loss / actual.len() as f32) // Return the average loss
    }
}
