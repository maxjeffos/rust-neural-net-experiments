pub fn activate(z: f64) -> f64 {
    if z < 0.0 {
        0.0
    } else {
        z
    }
}

pub fn activate_derivative(z: f64) -> f64 {
    if z > 0.0 {
        1.0
    } else {
        0.0
    }
}

// tests
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    pub fn activate_works() {
        assert_eq!(activate(-1.0), 0.0);
        assert_eq!(activate(0.0), 0.0);
        assert_eq!(activate(1.0), 1.0);
        assert_eq!(activate(2.0), 2.0);
    }

    #[test]
    pub fn activate_prime_works() {
        assert_eq!(activate_derivative(-1.0), 0.0);
        assert_eq!(activate_derivative(0.0), 0.0);
        assert_eq!(activate_derivative(1.0), 1.0);
        assert_eq!(activate_derivative(2.0), 1.0);
    }
}
