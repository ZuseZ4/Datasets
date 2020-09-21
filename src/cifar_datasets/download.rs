use tar::Archive;

use std::path::{Path, PathBuf};
use std::{fs, io};

use crate::download_helper::downloader;

const BASE_URL: &str = "https://www.cs.toronto.edu/~kriz";
const ARCHIVE: &str = "cifar-10-binary.tar.gz";
const ARCHIVE_LARGE: &str = "cifar-100-binary.tar.gz";

pub fn download_and_extract(base_path: &str, use_large_dataset: bool) -> Result<(), String> {
    let archive = if use_large_dataset {
        ARCHIVE_LARGE
    } else {
        ARCHIVE
    };

    println!("Attempting to download and extract {}...", archive);
    downloader::download(&base_path, BASE_URL.to_string(), vec![archive]).unwrap();
    let download_dir = PathBuf::from(base_path);
    println!("done downloading!");

    extract_gz(&archive, &download_dir)?;
    println!("done unpacking .tar.gz");

    Ok(())
}

fn extract_gz(archive_name: &str, download_dir: &Path) -> Result<(), String> {
    let archive = download_dir.join(&archive_name);
    let extract_to = download_dir.join(&archive_name.replace(".gz", ""));

    println!("Extracting archive {:?} to {:?}...", archive, extract_to);
    let file_in = fs::File::open(&archive)
        .or_else(|e| Err(format!("Failed to open archive {:?}: {:?}", archive, e)))?;
    let file_in = io::BufReader::new(file_in);
    let gz = flate2::bufread::GzDecoder::new(file_in);
    let mut archive = Archive::new(gz);
    let dst: String = "data".to_string();
    archive.unpack(dst).unwrap();

    Ok(())
}
