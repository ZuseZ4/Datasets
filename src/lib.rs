mod cifar_datasets;
mod mnist_datasets;
pub use cifar_datasets::cifar_builder::cifar10;
pub use mnist_datasets::mnist_builder::{mnist, mnist_fashion};
