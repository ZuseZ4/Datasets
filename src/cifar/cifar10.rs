
//mod test;


use ndarray::prelude::*;

use std::error::Error;
use std::fs::File;
use std::io::prelude::*;

#[derive(Debug)]
pub struct Cifar10<'a> {
    base_path: &'a str,
    show_images: bool,
    encode_one_hot: bool,
    training_bin_paths: Vec<&'a str>,
    testing_bin_paths: Vec<&'a str>,
    num_records_train: usize,
    num_records_test: usize,
}

impl<'a> Cifar10<'a> {
    pub fn default() -> Self {
        Cifar10 {
            base_path: "data/cifar-10-batches-bin/",
            show_images: false,
            encode_one_hot: true,
            training_bin_paths: vec![
                "data_batch_1.bin",
                "data_batch_2.bin",
                "data_batch_3.bin",
                "data_batch_4.bin",
                "data_batch_5.bin",
            ],
            testing_bin_paths: vec!["test_batch.bin"],
            num_records_train: 50_000,
            num_records_test: 10_000,
        }
    }

    pub fn base_path(mut self, base_path: &'a str) -> Self {
        self.base_path = base_path;
        self
    }

    pub fn encode_one_hot(mut self, encode_one_hot: bool) -> Self {
        self.encode_one_hot = encode_one_hot;
        self
    }

    pub fn training_bin_paths(mut self, training_bin_paths: Vec<&'a str>) -> Self {
        self.training_bin_paths = training_bin_paths;
        self
    }

    pub fn testing_bin_paths(mut self, testing_bin_paths: Vec<&'a str>) -> Self {
        self.testing_bin_paths = testing_bin_paths;
        self
    }

    pub fn num_records_train(mut self, num_records_train: usize) -> Self {
        self.num_records_train = num_records_train;
        self
    }

    pub fn num_records_test(mut self, num_records_test: usize) -> Self {
        self.num_records_test = num_records_test;
        self
    }


    pub fn build(self) -> Result<(Array4<u8>, Array2<u8>, Array4<u8>, Array2<u8>), Box<dyn Error>> {
        let (train_data, train_labels) = get_data(&self, "train")?;
        let (test_data, test_labels) = get_data(&self, "test")?;

        Ok((train_data, train_labels, test_data, test_labels))

    }

    pub fn build_as_flat_f32(self) -> Result<(Array2<f32>, Array2<f32>, Array2<f32>, Array2<f32>), Box<dyn Error>> {
        
        let (train_data, train_labels) = get_data(&self, "train")?;
        let (test_data, test_labels) = get_data(&self, "test")?;
        
        let train_labels = train_labels.mapv(|x| x as f32);
        let train_data = train_data
            .into_shape((self.num_records_train, 32 * 32 * 3))?
            .mapv(|x| x as f32 / 256.);
        let test_labels = test_labels.mapv(|x| x as f32);
        let test_data = test_data
            .into_shape((self.num_records_test, 32 * 32 * 3))?
            .mapv(|x| x as f32 / 256.);

        Ok((train_data, train_labels, test_data, test_labels))
    }

    pub fn new() -> (Array4<f32>, Array2<f32>, Array4<f32>, Array2<f32>) {
        let (trn_img, trn_lbl, tst_img, tst_lbl) = Cifar10::default().build().expect("Failed to build CIFAR-10 data");
        let trn_img = trn_img.mapv(|x| x as f32);
        let trn_lbl = trn_lbl.mapv(|x| x as f32);
        let tst_img = tst_img.mapv(|x| x as f32);
        let tst_lbl = tst_lbl.mapv(|x| x as f32);
        (trn_img, trn_lbl, tst_img, tst_lbl)
    }
    
}

fn get_data(config: &Cifar10, dataset: &str) -> Result<(Array4<u8>, Array2<u8>), Box<dyn Error>> {
    let mut buffer: Vec<u8> = Vec::new();

    let (bin_paths, num_records) = match dataset {
        "train" => (config.training_bin_paths.clone(), config.num_records_train),
        "test" => (config.testing_bin_paths.clone(), config.num_records_test),
        _ => panic!("An unexpected value was passed for which dataset should be parsed"),
    };

    for bin in &bin_paths {
        let full_cifar_path = [config.base_path, bin].join("");
        // println!("{}", full_cifar_path);

        let mut f = File::open(full_cifar_path)?;

        // read the whole file
        let mut temp_buffer: Vec<u8> = Vec::new();
        f.read_to_end(&mut temp_buffer)?;
        buffer.extend(&temp_buffer);
    }

    //println!("- Done parsing binary files to Vec<u8>");
    let mut labels: Array2<u8> = Array2::zeros((num_records, 10));
    labels[[0, buffer[0] as usize]] = 1;
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
        labels[[num, label as usize]] = 1;
        data.extend(&buffer[base + 1..=base + 3072]);
    }
    let data: Array4<u8> = Array::from_shape_vec((num_records, 3, 32, 32), data)?;


    Ok((data, labels))
}
