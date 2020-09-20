pub mod cifar_builder;
pub use cifar_builder::cifar10;
pub use cifar_builder::cifar100;

#[cfg(feature = "download")]
mod download;
