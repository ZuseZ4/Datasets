mod helper;

pub mod mnist_builder;

pub use mnist_builder::{mnist_fashion, mnist};

#[cfg(feature = "download")]
mod download;

#[cfg(test)]
mod test;
