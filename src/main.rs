use ndarray::Axis;
#[allow(unused_imports)]
use datasets::{mnist, mnist_fashion};

pub fn main() {
    let (train_size, test_size, rows, cols) = (60_000, 10_000, 28, 28);

    //mnist_fashion::download_and_extract();
    let mnist_fashion::Data {
        trn_img,
        trn_lbl: _,
        tst_img,
        tst_lbl: _,
        ..
    } = mnist_fashion::new();
    assert_eq!(trn_img.shape(), &[train_size, rows, cols]);
    assert_eq!(tst_img.shape(), &[test_size, rows, cols]);

    // Get the image of the first digit.
    let first_image = trn_img.index_axis(Axis(0), 0);
    assert_eq!(first_image.shape(), &[28, 28]);
}
