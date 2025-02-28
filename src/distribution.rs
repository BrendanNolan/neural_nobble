pub enum Distribution {
    Normal { mean: f64, standard_deviation: f64 },
    Uniform { lower_bound: f64, upper_bound: f64 },
}
