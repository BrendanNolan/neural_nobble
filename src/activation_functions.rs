use crate::derivative::ActivationFunction;

#[derive(Debug, Clone, Copy, Default)]
pub struct IdFunc {}

impl ActivationFunction for IdFunc {
    fn apply(&self, input: f64) -> f64 {
        input
    }
    fn derivative(&self, at: f64) -> f64 {
        1.0
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct SigmoidFunc {}

impl ActivationFunction for SigmoidFunc {
    fn apply(&self, input: f64) -> f64 {
        1.0 / (1.0 + (-input).exp())
    }
    fn derivative(&self, at: f64) -> f64 {
        let sig = self.apply(at);
        sig * (1.0 - sig)
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ReluFunc {}

impl ActivationFunction for ReluFunc {
    fn apply(&self, input: f64) -> f64 {
        input.max(0.0)
    }
    fn derivative(&self, at: f64) -> f64 {
        if at > 0.0 {
            1.0
        } else {
            0.0
        }
    }
}
