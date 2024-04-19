#[derive(Debug)]
pub enum Optimizer {
    StanardGradientDescent(StandardGradientDescentConfig),
    Momentum(MomentumConfig),
    // Nesterov(f64, f64),
    // AdaGrad(f64),
    // Adam(f64, f64, f64, f64),
    Adam(AdamConfig),
}

impl Optimizer {
    pub fn standard_gradient_descent(learning_rate: f64) -> Self {
        Optimizer::StanardGradientDescent(StandardGradientDescentConfig {
            learning_rate: learning_rate,
        })
    }

    pub fn momentum(learning_rate: f64, momentum: f64) -> Self {
        Optimizer::Momentum(MomentumConfig {
            learning_rate: learning_rate,
            momentum: momentum,
        })
    }
}

#[derive(Debug)]
pub struct StandardGradientDescentConfig {
    pub learning_rate: f64,
}

#[derive(Debug)]
pub struct MomentumConfig {
    pub learning_rate: f64,
    pub momentum: f64,
}

#[derive(Debug)]
pub struct AdamConfig {
    pub learning_rate: f64,
    pub momentum_decay: f64, // beta_1 in HOML
    pub scaling_decay: f64,  // beta_1 in HOML
    pub epsilon: f64,
}

impl AdamConfig {
    pub fn default() -> Self {
        Self {
            learning_rate: 0.001,
            momentum_decay: 0.9,
            scaling_decay: 0.999,
            epsilon: 1e-7, // GH copilot's suggestion was 1e-8
        }
    }
    pub fn with_learning_rate(learning_rate: f64) -> Self {
        Self {
            learning_rate,
            momentum_decay: 0.9,
            scaling_decay: 0.999,
            epsilon: 1e-7, // GH copilot's suggestion was 1e-8
        }
    }
}
