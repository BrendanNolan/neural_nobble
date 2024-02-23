use super::NeuralNetwork;
use crate::common::*;

#[derive(Debug)]
struct MiniBatch {
    pub inputs: Array2<f64>,
    pub targets: Array2<f64>,
}

enum MiniBatchSizeError {
    InputTargetCountMismatch,
    InputsOfDifferingSizes,
    TargetsOfDifferingSizes,
}

impl MiniBatch {
    pub fn new(
        inputs: Vec<Array1<f64>>,
        targets: Vec<Array1<f64>>,
    ) -> Result<Self, MiniBatchSizeError> {
        let batch_size = inputs.len();
        let mut mini_batch = MiniBatch::new_zeroed(inputs[0].len(), targets[0].len(), batch_size);
        mini_batch.populate(inputs, targets)?;
        Ok(mini_batch)
    }

    fn new_zeroed(input_size: usize, target_size: usize, batch_size: usize) -> Self {
        MiniBatch {
            inputs: Array2::default((input_size, batch_size)),
            targets: Array2::zeros((target_size, batch_size)),
        }
    }

    fn populate(
        &mut self,
        inputs: Vec<Array1<f64>>,
        targets: Vec<Array1<f64>>,
    ) -> Result<(), MiniBatchSizeError> {
        if inputs.len() != targets.len() {
            return Err(MiniBatchSizeError::InputTargetCountMismatch);
        } else if inputs
            .iter()
            .any(|input| input.len() != row_count(&self.inputs))
        {
            return Err(MiniBatchSizeError::InputsOfDifferingSizes);
        } else if targets
            .iter()
            .any(|target| target.len() != row_count(&self.targets))
        {
            return Err(MiniBatchSizeError::TargetsOfDifferingSizes);
        }
        for (batch_index, (input, target)) in inputs.iter().zip(targets.iter()).enumerate() {
            self.add_data_point(input, target, batch_index);
        }
        Ok(())
    }

    fn add_data_point(&mut self, input: &Array1<f64>, target: &Array1<f64>, batch_index: usize) {
        self.inputs.column_mut(batch_index).assign(input);
        self.targets.column_mut(batch_index).assign(target);
    }
}
