#[derive(Debug, Clone, PartialEq)]
pub struct LeakyReLU {
    tail_slope: f64,
}

impl LeakyReLU {
    pub fn new(tail_slope: f64) -> Self {
        Self { tail_slope }
    }
}

impl LeakyReLU {
    pub fn activate(&self, z: f64) -> f64 {
        if z < 0.0 {
            self.tail_slope * z
        } else {
            z
        }
    }

    pub fn activate_derivative(&self, z: f64) -> f64 {
        if z <= 0.0 {
            self.tail_slope
        } else {
            1.0
        }
    }
}

pub fn activate(z: f64, tail_slope: f64) -> f64 {
    if z < 0.0 {
        tail_slope * z
    } else {
        z
    }
}

pub fn activate_derivative(z: f64, tail_slope: f64) -> f64 {
    if z <= 0.0 {
        tail_slope
    } else {
        1.0
    }
}

// tests
#[cfg(test)]
mod tests {
    use super::*;

    use float_cmp::approx_eq;

    #[test]
    pub fn activate_works() {
        let leaky_relu = LeakyReLU {
            tail_slope: 0.0, // this makes this non-leaky, like regular ReLU
        };
        assert_eq!(leaky_relu.activate(-1.0), 0.0);
        assert_eq!(leaky_relu.activate(0.0), 0.0);
        assert_eq!(leaky_relu.activate(1.0), 1.0);
        assert_eq!(leaky_relu.activate(2.0), 2.0);

        let leaky_relu = LeakyReLU { tail_slope: 0.1 };

        // using approx_eq because of floating point errors
        assert!(approx_eq!(
            f64,
            leaky_relu.activate(-3.0),
            -0.3,
            epsilon = 0.0001
        ));
        assert!(approx_eq!(
            f64,
            leaky_relu.activate(-2.0),
            -0.2,
            epsilon = 0.0001
        ));
        assert!(approx_eq!(
            f64,
            leaky_relu.activate(-1.0),
            -0.1,
            epsilon = 0.0001
        ));

        assert_eq!(leaky_relu.activate(0.0), 0.0);
        assert_eq!(leaky_relu.activate(1.0), 1.0);
        assert_eq!(leaky_relu.activate(2.0), 2.0);
    }

    #[test]
    pub fn activate_prime_works() {
        let leaky_relu = LeakyReLU {
            tail_slope: 0.0, // this makes this non-leaky, like regular ReLU
        };
        assert_eq!(leaky_relu.activate_derivative(-1.0), 0.0);
        assert_eq!(leaky_relu.activate_derivative(0.0), 0.0);
        assert_eq!(leaky_relu.activate_derivative(1.0), 1.0);
        assert_eq!(leaky_relu.activate_derivative(2.0), 1.0);

        let leaky_relu = LeakyReLU { tail_slope: 0.1 };
        assert_eq!(leaky_relu.activate_derivative(-3.0), 0.1);
        assert_eq!(leaky_relu.activate_derivative(-2.0), 0.1);
        assert_eq!(leaky_relu.activate_derivative(-1.0), 0.1);
        assert_eq!(leaky_relu.activate_derivative(0.0), 0.1); // not mathematically exact since the derivative is not defined at 0
        assert_eq!(leaky_relu.activate_derivative(1.0), 1.0);
        assert_eq!(leaky_relu.activate_derivative(2.0), 1.0);
    }
}
