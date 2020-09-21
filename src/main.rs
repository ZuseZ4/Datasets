#[allow(unused_imports)]
use datasets::{cifar10, cifar100, mnist, mnist_fashion};
use ndarray::Axis;

pub fn main() {
    #[cfg(feature = "download")]
    cifar10::download_and_extract();
    #[cfg(feature = "download")]
    mnist::download_and_extract();

    let cifar10::Data {
        trn_img,
        trn_lbl: _,
        tst_img,
        tst_lbl: _,
        ..
    } = cifar10::new_normalized();
    let (train_size, test_size, depth, rows, cols) = (50_000, 10_000, 3, 32, 32); //cifar
    assert_eq!(trn_img.shape(), &[train_size, depth, rows, cols]);
    assert_eq!(tst_img.shape(), &[test_size, depth, rows, cols]);
    let first_image = trn_img.index_axis(Axis(0), 0);
    assert_eq!(first_image.shape(), &[3, 32, 32]);

    let mnist::Data {
        trn_img,
        trn_lbl: _,
        tst_img,
        tst_lbl: _,
        ..
    } = mnist::new_normalized();
    let (train_size, test_size, rows, cols) = (60_000, 10_000, 28, 28); //mnist
    assert_eq!(trn_img.shape(), &[train_size, rows, cols]);
    assert_eq!(tst_img.shape(), &[test_size, rows, cols]);
    // Get the image of the first digit.
    assert_eq!(first_image.shape(), &[28, 28]);
}
