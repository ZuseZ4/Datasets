use ndarray::prelude::*;
use ndarray::{Array2, Array4};

use std::error::Error;
use std::fs::File;
use std::io::prelude::*;

#[cfg(feature = "download")]
use super::download;

pub struct Data {
    pub trn_img: Array4<f32>,
    pub trn_lbl: Array2<f32>,
    pub tst_img: Array4<f32>,
    pub tst_lbl: Array2<f32>,
}

fn read_into_buffer(bin_paths: Vec<&str>, base_path: &str) -> Result<Vec<u8>, Box<dyn Error>> {
    let mut buffer: Vec<u8> = Vec::new();
    for bin in &bin_paths {
        let full_cifar_path = [base_path, bin].join("");
        println!("{}", full_cifar_path);

        let mut f = File::open(full_cifar_path)?;

        // read the whole file
        let mut temp_buffer: Vec<u8> = Vec::new();
        f.read_to_end(&mut temp_buffer)?;
        buffer.extend(&temp_buffer);
    }
    Ok(buffer)
}

fn buffer2data(
    buffer: Vec<u8>,
    num_records: usize,
) -> Result<(Array4<u8>, Array2<f32>), Box<dyn Error>> {
    let mut labels: Array2<f32> = Array2::zeros((num_records, 10));
    labels[[0, buffer[0] as usize]] = 1.;
    let mut data: Vec<u8> = Vec::with_capacity(num_records * 3072);

    for num in 0..num_records {
        // println!("Through image #{}/{}", num, num_records);
        let base = num * (3073);
        let label = buffer[base];
        if label > 9 {
            panic!(format!(
                "Label is {}, which is inconsistent with the CIFAR-10 scheme",
                label
            ));
        }
        labels[[num, label as usize]] = 1.;
        data.extend(&buffer[base + 1..=base + 3072]);
    }
    let data: Array4<u8> = Array::from_shape_vec((num_records, 3, 32, 32), data)?;

    Ok((data, labels))
}

fn get_dataset(base_path: &str, normalized: bool) -> Data {
    let bin_paths_trn = vec![
        "data_batch_1.bin",
        "data_batch_2.bin",
        "data_batch_3.bin",
        "data_batch_4.bin",
        "data_batch_5.bin",
    ];
    let bin_paths_tst = vec!["test_batch.bin"];
    let num_records_trn = 50_000;
    let num_records_tst = 10_000;

    println!("{}", base_path);
    let buffer_trn = read_into_buffer(bin_paths_trn, base_path).unwrap();
    let buffer_tst = read_into_buffer(bin_paths_tst, base_path).unwrap();
    //println!("- Done parsing binary files to Vec<u8>");

    let (trn_img, trn_lbl) = buffer2data(buffer_trn, num_records_trn).unwrap();
    let (tst_img, tst_lbl) = buffer2data(buffer_tst, num_records_tst).unwrap();

    let mut trn_img = trn_img.mapv(|x| x as f32);
    let mut tst_img = tst_img.mapv(|x| x as f32);
    if normalized {
        trn_img.mapv_inplace(|x| x / 256.);
        tst_img.mapv_inplace(|x| x / 256.);
    }
    Data {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
    }
}

pub mod cifar10 {
    pub use super::Data;
    static BASE_PATH: &str = "data/cifar-10-batches-bin/";
    pub fn new() -> Data {
        super::get_dataset(BASE_PATH, false)
    }
    pub fn new_normalized() -> Data {
        super::get_dataset(BASE_PATH, true)
    }

    #[cfg(feature = "download")]
    pub fn download_and_extract() {
        super::download::download_and_extract(BASE_PATH, false).unwrap();
    }
}
pub mod cifar100 {
    pub use super::Data;
    static BASE_PATH: &str = "data/cifar-100-binary/";
    pub fn new() -> Data {
        super::get_dataset(BASE_PATH, false)
    }
    pub fn new_normalized() -> Data {
        super::get_dataset(BASE_PATH, true)
    }

    #[cfg(feature = "download")]
    pub fn download_and_extract() {
        super::download::download_and_extract(BASE_PATH, true).unwrap();
    }
}
