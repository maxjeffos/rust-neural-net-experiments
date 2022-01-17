use crate::linalg::ColumnVector;

pub fn activate(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
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

pub fn activate_vector_in_place(z_v: &mut ColumnVector) {
    if z_v.num_elements() == 0 {
        panic!("this function is only valid for column vectors with at least one element (row)")
    }
    for i in 0..z_v.num_elements() {
        z_v.set(i, activate(z_v.get(i)));
    }
}

/// Compute the derivative of the sigmoid function at the given z
pub fn activate_prime(z: f64) -> f64 {
    let az = activate(z);
    az * (1.0 - az)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::column_vector;

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
    fn activate_vector_works() {
        let m1 = column_vector![-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0];
        let m2 = activate_vector(&m1);
        assert_eq!(m2.num_elements(), 7);
        assert_eq!(m2.get(0), activate(-4.0));
        assert_eq!(m2.get(1), activate(-2.0));
        assert_eq!(m2.get(2), activate(-1.0));
        assert_eq!(m2.get(3), 0.5);
        assert_eq!(m2.get(4), activate(1.0));
        assert_eq!(m2.get(5), activate(2.0));
        assert_eq!(m2.get(6), activate(4.0));
    }

    #[test]
    fn activate_vector_in_place_works() {
        let mut z = column_vector![-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0];
        activate_vector_in_place(&mut z);
        assert_eq!(z.num_elements(), 7);
        assert_eq!(z.get(0), activate(-4.0));
        assert_eq!(z.get(1), activate(-2.0));
        assert_eq!(z.get(2), activate(-1.0));
        assert_eq!(z.get(3), 0.5);
        assert_eq!(z.get(4), activate(1.0));
        assert_eq!(z.get(5), activate(2.0));
        assert_eq!(z.get(6), activate(4.0));
    }

    #[test]
    fn activate_prime_works() {
        assert_eq!(activate_prime(-1.0), 0.19661193324148185);
        assert_eq!(activate_prime(-0.5), 0.2350037122015945);
        assert_eq!(activate_prime(0.0), 0.25);
        assert_eq!(activate_prime(0.5), 0.2350037122015945);
        assert_eq!(activate_prime(1.0), 0.19661193324148185);
    }

    #[test]
    fn activate_prime_vec_works() {
        let v = ColumnVector::new(&[-1.0, -0.5, 0.0, 0.5, 1.0]);
        let sig_vec = activate_prime_vector(&v);
        assert_eq!(sig_vec.num_elements(), 5);
        assert_eq!(sig_vec.get(0), 0.19661193324148185);
        assert_eq!(sig_vec.get(1), 0.2350037122015945);
        assert_eq!(sig_vec.get(2), 0.25);
        assert_eq!(sig_vec.get(3), 0.2350037122015945);
        assert_eq!(sig_vec.get(4), 0.19661193324148185);
    }
}
