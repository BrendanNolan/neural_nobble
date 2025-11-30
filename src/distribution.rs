pub enum Distribution {
    Normal { mean: f32, standard_deviation: f32 },
    Uniform { lower_bound: f32, upper_bound: f32 },
}
