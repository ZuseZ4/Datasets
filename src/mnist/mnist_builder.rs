use ndarray::{Array2, Array3};
use byteorder::{BigEndian, ReadBytesExt};
use std::fs::File;
use std::io::prelude::*;
use std::path::Path;

#[cfg(feature = "download")]
use super::download;

static BASE_PATH: &str = "data/mnist";
static BASE_PATH_FASHION: &str = "data/mnist_fashion";
static TRN_IMG_FILENAME: &str = "train-images-idx3-ubyte";
static TRN_LBL_FILENAME: &str = "train-labels-idx1-ubyte";
static TST_IMG_FILENAME: &str = "t10k-images-idx3-ubyte";
static TST_LBL_FILENAME: &str = "t10k-labels-idx1-ubyte";
static IMG_MAGIC_NUMBER: u32 = 0x0000_0803;
static LBL_MAGIC_NUMBER: u32 = 0x0000_0801;
static TRN_LEN: u32 = 60000;
static TST_LEN: u32 = 10000;
static CLASSES: usize = 10;
static ROWS: usize = 28;
static COLS: usize = 28;

pub struct MNIST {
    pub trn_img: Array3<f32>,
    pub trn_lbl: Array2<f32>,
    pub val_img: Array3<f32>,
    pub val_lbl: Array2<f32>,
    pub tst_img: Array3<f32>,
    pub tst_lbl: Array2<f32>,
}

pub struct Helper {}

impl Helper {
    fn labels(path: &Path, expected_length: u32) -> Vec<u8> {
        let mut file =
            File::open(path).unwrap_or_else(|_| panic!("Unable to find path to labels at {:?}.", path));
        let magic_number = file
            .read_u32::<BigEndian>()
            .unwrap_or_else(|_| panic!("Unable to read magic number from {:?}.", path));
        assert!(
            LBL_MAGIC_NUMBER == magic_number,
            format!(
                "Expected magic number {} got {}.",
                LBL_MAGIC_NUMBER, magic_number
            )
        );
        let length = file
            .read_u32::<BigEndian>()
            .unwrap_or_else(|_| panic!("Unable to length from {:?}.", path));
        assert!(
            expected_length == length,
            format!(
                "Expected data set length of {} got {}.",
                expected_length, length
            )
        );
        file.bytes().map(|b| b.unwrap()).collect()
    }

    fn images(path: &Path, expected_length: u32) -> Vec<u8> {
        // Read whole file in memory
        let mut content: Vec<u8> = Vec::new();
        let mut file = {
            let mut fh = File::open(path)
                .unwrap_or_else(|_| panic!("Unable to find path to images at {:?}.", path));
            let _ = fh
                .read_to_end(&mut content)
                .unwrap_or_else(|_| panic!("Unable to read whole file in memory ({})", path.display()));
            // The read_u32() method, coming from the byteorder crate's ReadBytesExt trait, cannot be
            // used with a `Vec` directly, it requires a slice.
            &content[..]
        };

        let magic_number = file
            .read_u32::<BigEndian>()
            .unwrap_or_else(|_| panic!("Unable to read magic number from {:?}.", path));
        assert!(
            IMG_MAGIC_NUMBER == magic_number,
            format!(
                "Expected magic number {} got {}.",
                IMG_MAGIC_NUMBER, magic_number
            )
        );
        let length = file
            .read_u32::<BigEndian>()
            .unwrap_or_else(|_| panic!("Unable to length from {:?}.", path));
        assert!(
            expected_length == length,
            format!(
                "Expected data set length of {} got {}.",
                expected_length, length
            )
        );
        let rows = file
            .read_u32::<BigEndian>()
            .unwrap_or_else(|_| panic!("Unable to number of rows from {:?}.", path))
            as usize;
        assert!(
            ROWS == rows,
            format!("Expected rows length of {} got {}.", ROWS, rows)
        );
        let cols = file
            .read_u32::<BigEndian>()
            .unwrap_or_else(|_| panic!("Unable to number of columns from {:?}.", path))
            as usize;
        assert!(
            COLS == cols,
            format!("Expected cols length of {} got {}.", COLS, cols)
        );
    // Convert `file` from a Vec to a slice.
    file.to_vec()
    }

    pub fn new(base_path: &str) -> MNIST {
        //let base_path = BASE_PATH;
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
        let mut trn_img = Helper::images(
            &Path::new(base_path).join(trn_img_filename),
            TRN_LEN,
        );
        let mut trn_lbl = Helper::labels(
            &Path::new(base_path).join(trn_lbl_filename),
            TRN_LEN,
        );
        let mut tst_img = Helper::images(
            &Path::new(base_path).join(tst_img_filename),
            TST_LEN,
        );
        let mut tst_lbl = Helper::labels(
            &Path::new(base_path).join(tst_lbl_filename),
            TST_LEN,
        );
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

        let trn_img = Array3::from_shape_vec((trn_len, ROWS, COLS), trn_img).unwrap().mapv(|x| x as f32);
        let tst_img = Array3::from_shape_vec((tst_len, ROWS, COLS), tst_img).unwrap().mapv(|x| x as f32);
        let val_img = Array3::from_shape_vec((val_len, ROWS, COLS), val_img).unwrap().mapv(|x| x as f32);
        let trn_lbl = Array2::from_shape_vec((trn_len, CLASSES), trn_lbl).unwrap().mapv(|x| x as f32);
        let tst_lbl = Array2::from_shape_vec((tst_len, CLASSES), tst_lbl).unwrap().mapv(|x| x as f32);
        let val_lbl = Array2::from_shape_vec((val_len, CLASSES), val_lbl).unwrap().mapv(|x| x as f32);



        MNIST {
            trn_img,
            trn_lbl,
            val_img,
            val_lbl,
            tst_img,
            tst_lbl,
        }
    }
}
pub struct Mnist {}
impl Mnist {
  pub fn new() -> MNIST {
    Helper::new(BASE_PATH)
  }

  #[cfg(feature = "download")]
  pub fn download_and_extract() {
      download::download_and_extract(BASE_PATH, false).unwrap();
  }

}

pub struct MnistFashion {}
impl MnistFashion {
  pub fn new() -> MNIST {
    Helper::new(BASE_PATH_FASHION)
  }
  #[cfg(feature = "download")]
  pub fn download_and_extract() {
      download::download_and_extract(BASE_PATH_FASHION, true).unwrap();
  }

}
