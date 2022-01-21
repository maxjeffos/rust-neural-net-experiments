// for info on ELU, see HOML P 336 and https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html#elu

// const ELU_SCALE: f64 = 1.0; // this is usually 1.0 but it can also be a hyperparameter - see HOML p 336

pub fn activate(z: f64) -> f64 {
    if z < 0.0 {
        // ELU_SCALE * (z.exp() - 1.0)
        z.exp() - 1.0
    } else {
        z
    }
}

pub fn activate_derivative(z: f64) -> f64 {
    if z < 0.0 {
        // ELU_SCALE * z.exp()
        z.exp()
    } else {
        1.0
    }
}

// tests
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    pub fn activate_works() {
        assert_eq!(activate(-5.0), (-5.0_f64).exp() - 1.0);
        assert_eq!(activate(-1.0), (-1.0_f64).exp() - 1.0);
        assert_eq!(activate(0.0), 0.0);
        assert_eq!(activate(1.0), 1.0);
        assert_eq!(activate(2.0), 2.0);
    }

    #[test]
    pub fn activate_prime_works() {
        assert_eq!(activate_derivative(-5.0), (-5.0_f64).exp());
        assert_eq!(activate_derivative(-1.0), (-1.0_f64).exp());
        assert_eq!(activate_derivative(0.0), 1.0);
        assert_eq!(activate_derivative(1.0), 1.0);
        assert_eq!(activate_derivative(2.0), 1.0);
    }
}
