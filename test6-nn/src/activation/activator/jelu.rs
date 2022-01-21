use crate::activation::activator::Activator;

#[derive(Debug)]
struct JELU {
    /// crossover_point is the crossover point between the tail and the exponential component and must be negative.
    pub crossover_point: f64,
    tail_slope: f64,
    tail_intercept: f64,
}

impl JELU {
    pub fn new(crossover_point: f64) -> JELU {
        if crossover_point >= 0.0 {
            panic!("crossover_point must be negative");
        }

        let e_to_the_crossover_point = crossover_point.exp();
        let tail_slope = e_to_the_crossover_point;
        let tail_intercept = e_to_the_crossover_point * (1.0 - crossover_point) - 1.0;

        JELU {
            crossover_point,
            tail_slope,
            tail_intercept,
        }
    }
}

impl Activator for JELU {
    fn activate(&self, z: f64) -> f64 {
        if z <= self.crossover_point {
            self.tail_slope * z + self.tail_intercept
        } else if z > self.crossover_point && z < 0.0 {
            // the exponential part - same as ELU
            z.exp() - 1.0
        } else {
            // z >= 0.0
            z
        }
    }

    fn activate_derivative(&self, z: f64) -> f64 {
        if z <= self.crossover_point {
            self.tail_slope
        } else if z > self.crossover_point && z < 0.0 {
            // the exponential part - same as ELU
            z.exp()
        } else {
            // z >= 0.0
            1.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_activate() {
        let crossover_point = -3.0;
        let lelu = JELU::new(crossover_point);
        let crossover_y = crossover_point.exp() - 1.0;

        let tail_slope = crossover_point.exp();
        let tail_intercept = crossover_point.exp() * (1.0 - crossover_point) - 1.0;

        let at_minus_5 = (-5.0) * tail_slope + tail_intercept;

        assert_eq!(lelu.activate(-5.0), at_minus_5);

        // transition point from tail to exponential range
        assert_eq!(lelu.activate(crossover_point), crossover_y);
        assert_eq!(
            lelu.activate(crossover_point),
            (crossover_point).exp() - 1.0
        );

        assert_eq!(lelu.activate(-1.0), (-1.0_f64).exp() - 1.0);

        // transition point from exponential range to head
        assert_eq!(lelu.activate(0.0), (0.0_f64).exp() - 1.0);
        assert_eq!(lelu.activate(0.0), 0.0);

        assert_eq!(lelu.activate(1.0), 1.0);
        assert_eq!(lelu.activate(2.0), 2.0);
        assert_eq!(lelu.activate(3.0), 3.0);
    }

    #[test]
    fn test_activate_derivative() {
        let crossover_point = -3.0;
        let lelu = JELU::new(crossover_point);
        let tail_slope = crossover_point.exp();

        assert_eq!(lelu.activate_derivative(-5.0), tail_slope);

        // transition point from tail to exponential range
        assert_eq!(lelu.activate_derivative(crossover_point), tail_slope);
        assert_eq!(
            lelu.activate_derivative(crossover_point),
            crossover_point.exp()
        );

        assert_eq!(lelu.activate_derivative(-1.0), (-1.0_f64).exp()); // within the exponential range

        // transition point from exponential range to head
        assert_eq!(lelu.activate_derivative(0.0), 0.0_f64.exp());
        assert_eq!(lelu.activate_derivative(0.0), 1.0);

        assert_eq!(lelu.activate_derivative(1.0), 1.0);
        assert_eq!(lelu.activate_derivative(2.0), 1.0);
        assert_eq!(lelu.activate_derivative(3.0), 1.0);
    }
}
