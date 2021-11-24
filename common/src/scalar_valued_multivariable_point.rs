#[derive(Debug, PartialEq)]
pub struct ScalarValuedMultivariablePoint {
    pub independant: Vec<f64>,
    pub dependant: f64,
}

impl ScalarValuedMultivariablePoint {
    pub fn new(independant: Vec<f64>, dependant: f64) -> ScalarValuedMultivariablePoint {
        ScalarValuedMultivariablePoint {
            independant,
            dependant,
        }
    }

    pub fn new_2d(x: f64, y: f64) -> ScalarValuedMultivariablePoint {
        ScalarValuedMultivariablePoint {
            independant: vec![x],
            dependant: y,
        }
    }

    pub fn new_3d(x0: f64, x1: f64, y: f64) -> ScalarValuedMultivariablePoint {
        ScalarValuedMultivariablePoint {
            independant: vec![x0, x1],
            dependant: y,
        }
    }

    pub fn dimension(&self) -> usize {
        self.independant.len() + 1 // +1 for the dependant
    }
}
