use crate::derivative::DifferentiableFunction;

#[derive(Debug, Clone, Copy, Default)]
pub struct IdFunc {}

impl DifferentiableFunction for IdFunc {
    fn apply(&self, input: f64) -> f64 {
        input
    }
    fn derivative(&self, at: f64) -> f64 {
        1.0
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct SigmoidFunc {}

impl DifferentiableFunction for SigmoidFunc {
    fn apply(&self, input: f64) -> f64 {
        1.0 / (1.0 + (-input).exp())
    }
    fn derivative(&self, at: f64) -> f64 {
        let sig = self.apply(at);
        sig * (1.0 - sig)
    }
}
