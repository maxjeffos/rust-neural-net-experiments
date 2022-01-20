use crate::linalg::ColumnVector;

pub fn activate(z: f64) -> f64 {
    if z < 0.0 {
        0.0
    } else {
        z
    }
}

pub fn activate_vector(z_v: &ColumnVector) -> ColumnVector {
    if z_v.num_elements() == 0 {
        panic!("this function is only valid for column vectors with at least one element (row)")
    }

    let data = z_v
        .iter()
        .map(|z| activate(*z))
        .collect::<Vec<f64>>();

    ColumnVector::from_vec(data)
}

pub fn activate_prime(z: f64) -> f64 {
    if z > 0.0 {
        1.0
    } else {
        0.0
    }
}

pub fn activate_prime_vector(z_v: &ColumnVector) -> ColumnVector {
    if z_v.num_elements() == 0 {
        panic!("this function is only valid for column vectors with at least one element (row)")
    }

    let data = z_v
        .iter()
        .map(|z| activate_prime(*z))
        .collect::<Vec<f64>>();

    ColumnVector::from_vec(data)
}

// tests
#[cfg(test)]
mod tests {
    use super::*;
    use crate::column_vector;

    #[test]
    pub fn activate_works() {
        assert_eq!(activate(-1.0), 0.0);
        assert_eq!(activate(0.0), 0.0);
        assert_eq!(activate(1.0), 1.0);
        assert_eq!(activate(2.0), 2.0);
    }

    #[test]
    pub fn activate_vector_works() {
        let v = column_vector![-1.0, 0.0, 1.0, 2.0];
        let v_prime = activate_vector(&v);
        assert_eq!(v_prime, column_vector![0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    pub fn activate_prime_works() {
        assert_eq!(activate_prime(-1.0), 0.0);
        assert_eq!(activate_prime(0.0), 0.0);
        assert_eq!(activate_prime(1.0), 1.0);
        assert_eq!(activate_prime(2.0), 1.0);
    }

    #[test]
    pub fn activate_prime_vector_works() {
        let v = column_vector![-1.0, 0.0, 1.0, 2.0];
        let v_prime = activate_prime_vector(&v);
        assert_eq!(v_prime, column_vector![0.0, 0.0, 1.0, 1.0]);
    }
}
