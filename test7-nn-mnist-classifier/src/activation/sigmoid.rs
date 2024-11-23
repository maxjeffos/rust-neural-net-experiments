pub fn activate(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}

/// Compute the derivative of the sigmoid function at the given z
pub fn activate_derivative(z: f64) -> f64 {
    let az = activate(z);
    az * (1.0 - az)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn activate_works() {
        assert_eq!(activate(-4.0), 0.01798620996209156);
        assert_eq!(activate(-2.0), 0.11920292202211755);
        assert_eq!(activate(-1.0), 0.2689414213699951);
        assert_eq!(activate(0.0), 0.5);
        assert_eq!(activate(1.0), 0.7310585786300049);
        assert_eq!(activate(2.0), 0.8807970779778823);
        assert_eq!(activate(4.0), 0.9820137900379085);
    }

    #[test]
    fn activate_derivative_works() {
        assert_eq!(activate_derivative(-1.0), 0.19661193324148185);
        assert_eq!(activate_derivative(-0.5), 0.2350037122015945);
        assert_eq!(activate_derivative(0.0), 0.25);
        assert_eq!(activate_derivative(0.5), 0.2350037122015945);
        assert_eq!(activate_derivative(1.0), 0.19661193324148185);
    }
}
