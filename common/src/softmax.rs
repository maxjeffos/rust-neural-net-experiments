fn softmax(logits: &[f64]) -> Vec<f64> {
    let max_logit = logits
        .iter()
        .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let exps: Vec<f64> = logits
        .iter()
        .map(|&x| (x - max_logit).exp())
        .collect();
    let sum_exps: f64 = exps.iter().sum();
    exps.iter().map(|&x| x / sum_exps).collect()
}

// softmax_derivative computes the derivative of the softmax function.
// The output is the Jacobian Matrix which describes how a change in any input logit affects
// every output probability.
fn softmax_derivative(logits: &[f64]) -> Vec<Vec<f64>> {
    let probabilities = softmax(logits);
    let mut derivatives = vec![vec![0.0; logits.len()]; logits.len()];

    for i in 0..logits.len() {
        for j in 0..logits.len() {
            if i == j {
                derivatives[i][j] = probabilities[i] * (1.0 - probabilities[i]);
            } else {
                derivatives[i][j] = -probabilities[i] * probabilities[j];
            }
        }
    }
    derivatives
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_sum_to_one() {
        let logits = vec![0.5, 1.5, 3.0, 2.0];
        let probabilities = softmax(&logits);
        let sum: f64 = probabilities.iter().sum();
        assert!(
            (1.0 - sum).abs() < 1e-7,
            "Sum of probabilities should be close to 1.0, but is {}",
            sum
        );
    }

    #[test]
    fn test_softmax_output_range() {
        let logits = vec![1.0, 2.0, 3.0];
        let probabilities = softmax(&logits);
        for &prob in probabilities.iter() {
            assert!(
                prob >= 0.0 && prob <= 1.0,
                "Each probability should be within [0, 1], but got {}",
                prob
            );
        }
    }

    #[test]
    fn test_softmax_numerical_stability() {
        let logits = vec![1000.0, 1001.0, 1002.0];
        let probabilities = softmax(&logits);
        let sum: f64 = probabilities.iter().sum();
        assert!(
            (1.0 - sum).abs() < 1e-7,
            "Sum of probabilities should be close to 1.0, but is {}",
            sum
        );
        assert!(
            probabilities
                .iter()
                .all(|&p| p >= 0.0 && p <= 1.0),
            "Probabilities should be within [0, 1]"
        );
    }

    // This test checks that the diagonal elements of the derivative matrix, which represent the derivative of 
    // each softmax output with respect to its corresponding logit, are computed correctly.
    #[test]
    fn test_softmax_derivative_diagonal() {
        let logits = vec![1.0, 2.0, 3.0];
        let derivs = softmax_derivative(&logits);
        let probs = softmax(&logits);

        for i in 0..logits.len() {
            assert!(
                (derivs[i][i] - (probs[i] * (1.0 - probs[i]))).abs() < 1e-7,
                "Diagonal elements incorrect at index {}: expected {}, got {}",
                i,
                probs[i] * (1.0 - probs[i]),
                derivs[i][i]
            );
        }
    }

    #[test]
    fn test_softmax_derivative_off_diagonal() {
        let logits = vec![1.0, 2.0, 3.0];
        let derivs = softmax_derivative(&logits);
        let probs = softmax(&logits);

        for i in 0..logits.len() {
            for j in 0..logits.len() {
                if i != j {
                    assert!(
                        (derivs[i][j] - (-probs[i] * probs[j])).abs() < 1e-7,
                        "Off-diagonal elements incorrect at indices ({}, {}): expected {}, got {}",
                        i,
                        j,
                        -probs[i] * probs[j],
                        derivs[i][j]
                    );
                }
            }
        }
    }

    #[test]
    fn test_softmax_derivative_uniform_input() {
        let logits = vec![1.0, 1.0, 1.0]; // Uniform input
        let derivs = softmax_derivative(&logits);
        let probs = softmax(&logits);

        for i in 0..logits.len() {
            for j in 0..logits.len() {
                let expected = if i == j {
                    probs[i] * (1.0 - probs[i])
                } else {
                    -probs[i] * probs[j]
                };
                assert!(
                    (derivs[i][j] - expected).abs() < 1e-7,
                    "Uniform input elements incorrect at indices ({}, {}): expected {}, got {}",
                    i,
                    j,
                    expected,
                    derivs[i][j]
                );
            }
        }
    }

    #[test]
    fn test_softmax_derivative_matrix_size() {
        let logits = vec![0.5, -0.1, 2.0];
        let derivs = softmax_derivative(&logits);

        assert_eq!(
            derivs.len(),
            logits.len(),
            "Derivative matrix row count incorrect"
        );
        for row in derivs.iter() {
            assert_eq!(
                row.len(),
                logits.len(),
                "Derivative matrix column count incorrect for some rows"
            );
        }
    }
}
