use crate::activation::activator::Activator;

#[derive(Debug)]
pub struct ReLU {}
impl ReLU {
    pub fn new() -> ReLU {
        ReLU {}
    }
}

impl Activator for ReLU {
    fn activate(&self, z: f64) -> f64 {
        if z < 0.0 {
            0.0
        } else {
            z
        }
    }

    fn activate_derivative(&self, z: f64) -> f64 {
        if z > 0.0 {
            1.0
        } else {
            0.0
        }
    }
}

// tests
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    pub fn activate_works() {
        let a = ReLU::new();
        assert_eq!(a.activate(-1.0), 0.0);
        assert_eq!(a.activate(0.0), 0.0);
        assert_eq!(a.activate(1.0), 1.0);
        assert_eq!(a.activate(2.0), 2.0);
    }

    #[test]
    pub fn activate_prime_works() {
        let a = ReLU::new();
        assert_eq!(a.activate_derivative(-1.0), 0.0);
        assert_eq!(a.activate_derivative(0.0), 0.0);
        assert_eq!(a.activate_derivative(1.0), 1.0);
        assert_eq!(a.activate_derivative(2.0), 1.0);
    }
}
