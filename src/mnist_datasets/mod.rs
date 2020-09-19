mod helper;

pub mod mnist_builder;

pub use mnist_builder::{mnist, mnist_fashion};

#[cfg(feature = "download")]
mod download;

//#[cfg(test)]
//mod test;
