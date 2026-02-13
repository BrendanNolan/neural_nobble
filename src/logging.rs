use std::io::{self, Write};

pub fn log(message: &str) {
    #[cfg(feature = "neural_nobble_log")]
    {
        eprintln!("{}", message);
        io::stdout().flush().unwrap();
    }
}
