use ndarray::{Array2, Array3};
use std::path::Path;

#[cfg(feature = "download")]
use super::download;
use super::helper;

static TRN_IMG_FILENAME: &str = "train-images-idx3-ubyte";
static TRN_LBL_FILENAME: &str = "train-labels-idx1-ubyte";
static TST_IMG_FILENAME: &str = "t10k-images-idx3-ubyte";
static TST_LBL_FILENAME: &str = "t10k-labels-idx1-ubyte";
static TRN_LEN: u32 = 60000;
static TST_LEN: u32 = 10000;
static CLASSES: usize = 10;
static ROWS: usize = 28;
static COLS: usize = 28;

pub struct Data {
    pub trn_img: Array3<f32>,
    pub trn_lbl: Array2<f32>,
    pub val_img: Array3<f32>,
    pub val_lbl: Array2<f32>,
    pub tst_img: Array3<f32>,
    pub tst_lbl: Array2<f32>,
}

pub fn get_data(base_path: &str) -> Data {
    let trn_img_filename = TRN_IMG_FILENAME;
    let trn_lbl_filename = TRN_LBL_FILENAME;
    let tst_img_filename = TST_IMG_FILENAME;
    let tst_lbl_filename = TST_LBL_FILENAME;
    let one_hot = true;

    let (trn_len, val_len, tst_len) = (TRN_LEN as usize, 0 as usize, TST_LEN as usize);
    let total_length = trn_len + val_len + tst_len;
    let available_length = (TRN_LEN + TST_LEN) as usize;
    assert!(
        total_length <= available_length,
        format!(
            "Total data set length ({}) greater than maximum possible length ({}).",
            total_length, available_length
        )
    );
    let mut trn_img = helper::images(&Path::new(base_path).join(trn_img_filename), TRN_LEN);
    let mut trn_lbl = helper::labels(&Path::new(base_path).join(trn_lbl_filename), TRN_LEN);
    let mut tst_img = helper::images(&Path::new(base_path).join(tst_img_filename), TST_LEN);
    let mut tst_lbl = helper::labels(&Path::new(base_path).join(tst_lbl_filename), TST_LEN);
    trn_img.append(&mut tst_img);
    trn_lbl.append(&mut tst_lbl);
    let mut val_img = trn_img.split_off(trn_len * ROWS * COLS);
    let mut val_lbl = trn_lbl.split_off(trn_len);
    let mut tst_img = val_img.split_off(val_len * ROWS * COLS);
    let mut tst_lbl = val_lbl.split_off(val_len);
    tst_img.split_off(tst_len * ROWS * COLS);
    tst_lbl.split_off(tst_len);
    if one_hot {
        fn digit2one_hot(v: Vec<u8>) -> Vec<u8> {
            v.iter()
                .map(|&i| {
                    let mut v = vec![0; CLASSES as usize];
                    v[i as usize] = 1;
                    v
                })
                .flatten()
                .collect()
        }
        trn_lbl = digit2one_hot(trn_lbl);
        val_lbl = digit2one_hot(val_lbl);
        tst_lbl = digit2one_hot(tst_lbl);
    }
    let trn_img = Array3::from_shape_vec((trn_len, ROWS, COLS), trn_img)
        .unwrap()
        .mapv(|x| x as f32);
    let tst_img = Array3::from_shape_vec((tst_len, ROWS, COLS), tst_img)
        .unwrap()
        .mapv(|x| x as f32);
    let val_img = Array3::from_shape_vec((val_len, ROWS, COLS), val_img)
        .unwrap()
        .mapv(|x| x as f32);
    let trn_lbl = Array2::from_shape_vec((trn_len, CLASSES), trn_lbl)
        .unwrap()
        .mapv(|x| x as f32);
    let tst_lbl = Array2::from_shape_vec((tst_len, CLASSES), tst_lbl)
        .unwrap()
        .mapv(|x| x as f32);
    let val_lbl = Array2::from_shape_vec((val_len, CLASSES), val_lbl)
        .unwrap()
        .mapv(|x| x as f32);
    Data {
        trn_img,
        trn_lbl,
        val_img,
        val_lbl,
        tst_img,
        tst_lbl,
    }
}

pub fn get_normalized_data(base_path: &str) -> Data {
    let Data {
        mut trn_img,
        trn_lbl,
        mut val_img,
        val_lbl,
        mut tst_img,
        tst_lbl,
    } = get_data(base_path);
    trn_img.mapv_inplace(|x| x / 256.);
    tst_img.mapv_inplace(|x| x / 256.);
    val_img.mapv_inplace(|x| x / 256.);
    Data {
        trn_img,
        trn_lbl,
        val_img,
        val_lbl,
        tst_img,
        tst_lbl,
    }
}

pub mod mnist {
    pub use super::Data;
    static BASE_PATH: &str = "data/mnist";
    pub fn new() -> Data {
        super::get_data(BASE_PATH)
    }
    pub fn new_normalized() -> Data {
        super::get_normalized_data(BASE_PATH)
    }

    #[cfg(feature = "download")]
    pub fn download_and_extract() {
        super::download::download_and_extract(BASE_PATH, false).unwrap();
    }
}

pub mod mnist_fashion {
    pub use super::Data;
    static BASE_PATH_FASHION: &str = "data/mnist_fashion";
    pub fn new() -> Data {
        super::get_data(BASE_PATH_FASHION)
    }
    pub fn new_normalized() -> Data {
        super::get_normalized_data(BASE_PATH_FASHION)
    }
    #[cfg(feature = "download")]
    pub fn download_and_extract() {
        super::download::download_and_extract(BASE_PATH_FASHION, true).unwrap();
    }
}
