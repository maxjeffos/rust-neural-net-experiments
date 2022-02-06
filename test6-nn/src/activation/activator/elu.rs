// for info on ELU, see HOML P 336 and https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html#elu

// const ELU_SCALE: f64 = 1.0; // this is usually 1.0 but it can also be a hyperparameter - see HOML p 336

use crate::activation::activator::Activator;

#[derive(Debug)]
pub struct ELU {}

impl ELU {
    pub fn new() -> ELU {
        ELU {}
    }
}

impl Activator for ELU {
    fn activate(&self, z: f64) -> f64 {
        if z < 0.0 {
            // ELU_SCALE * (z.exp() - 1.0)
            z.exp() - 1.0
        } else {
            z
        }
    }

    fn activate_derivative(&self, z: f64) -> f64 {
        if z < 0.0 {
            // ELU_SCALE * z.exp()
            z.exp()
        } else {
            1.0
        }
    }
}

// tests
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    pub fn activate_works() {
        let a = ELU::new();
        assert_eq!(a.activate(-5.0), (-5.0_f64).exp() - 1.0);
        assert_eq!(a.activate(-1.0), (-1.0_f64).exp() - 1.0);
        assert_eq!(a.activate(0.0), 0.0);
        assert_eq!(a.activate(1.0), 1.0);
        assert_eq!(a.activate(2.0), 2.0);
    }

    #[test]
    pub fn activate_prime_works() {
        let a = ELU::new();
        assert_eq!(a.activate_derivative(-5.0), (-5.0_f64).exp());
        assert_eq!(a.activate_derivative(-1.0), (-1.0_f64).exp());
        assert_eq!(a.activate_derivative(0.0), 1.0);
        assert_eq!(a.activate_derivative(1.0), 1.0);
        assert_eq!(a.activate_derivative(2.0), 1.0);
    }
}
