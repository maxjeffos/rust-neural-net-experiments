use crate::errors::VectorDimensionMismatch;
use common::linalg::{square, ColumnVector};

#[derive(Debug, Clone, PartialEq)]
pub enum CostFunc {
    QuadraticCost,
    CrossEntropy,
}

pub trait Coster {
    fn cost(
        &self,
        desired_v: &ColumnVector,
        actual_v: &ColumnVector,
    ) -> Result<f64, VectorDimensionMismatch>;
}

/// Calculates the quadratic cost between the desired and actual output column vectors.
pub fn quadratic_cost(
    desired_v: &ColumnVector,
    actual_v: &ColumnVector,
) -> Result<f64, VectorDimensionMismatch> {
    // Note that 3B1B does not do the divide by 2 and he ends up with a 2 in the derivative function.
    // Neilson does the divide by 2
    // I'm doing the divide by 2

    if desired_v.num_elements() != actual_v.num_elements() {
        return Err(VectorDimensionMismatch::new_with_msg(
            desired_v.num_elements(),
            actual_v.num_elements(),
            "expected and actual outputs must have the same length",
        ));
    }

    Ok(desired_v
        .iter_with(actual_v)
        .map(|(exp, act)| exp - act)
        .map(square)
        .sum::<f64>()
        / 2.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use common::column_vector;

    #[test]
    pub fn test_quadratic_cost_fn() {
        let inputs = column_vector![0.0, 0.5, 1.0];
        let targets = column_vector![0.0, 0.5, 1.0];
        let cost = quadratic_cost(&inputs, &targets);
        assert_eq!(cost, Ok(0.0));

        let inputs = column_vector![4.0, 4.0];
        let targets = column_vector![2.0, 2.0];
        let cost = quadratic_cost(&inputs, &targets);
        assert_eq!(cost, Ok(4.0));
    }

    #[test]
    pub fn test_quadratic_cost_fn_dimension_mismatch() {
        let inputs = column_vector![0.0, 0.5, 1.0];
        let targets = column_vector![0.0, 0.5];

        let result = quadratic_cost(&inputs, &targets);
        assert!(result.is_err());

        let expected_error = VectorDimensionMismatch::new_with_msg(
            inputs.num_elements(),
            targets.num_elements(),
            "expected and actual outputs must have the same length",
        );
        assert_eq!(result.unwrap_err(), expected_error);
    }
}
