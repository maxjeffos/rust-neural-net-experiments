use crate::linalg::ColumnVector;

#[derive(Debug)]
pub struct NDTrainingDataPoint {
    pub input_v: ColumnVector,
    pub desired_output_v: ColumnVector,
}

impl NDTrainingDataPoint {
    pub fn new(input_v: ColumnVector, desired_output_v: ColumnVector) -> Self {
        NDTrainingDataPoint {
            input_v,
            desired_output_v,
        }
    }
}
