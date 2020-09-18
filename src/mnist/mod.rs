mod mnist_builder;
pub use mnist_builder::MnistFashion;
pub use mnist_builder::Mnist;

#[cfg(feature = "download")]
mod download;

#[cfg(test)]
mod test;
