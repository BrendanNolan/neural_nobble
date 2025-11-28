pub fn log(message: &str) {
    #[cfg(feature = "neural_nobble_log")]
    {
        println!("{}", message);
    }
}
