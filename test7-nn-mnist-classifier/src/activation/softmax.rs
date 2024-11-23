use common::linalg::ColumnVector;

pub fn softmax(logits: &[f64]) -> Vec<f64> {
    let max_logit = logits.iter().fold(f64::MIN, |max, &val| max.max(val));
    let exps: Vec<f64> = logits.iter().map(|&val| (val - max_logit).exp()).collect();
    let sum_exps: f64 = exps.iter().sum();
    exps.iter().map(|&exp| exp / sum_exps).collect()
}

pub fn activate_vector(z: &ColumnVector) -> ColumnVector {
    let data = softmax(z);
    ColumnVector::from_vec(data)
}
