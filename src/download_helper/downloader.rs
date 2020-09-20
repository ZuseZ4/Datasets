extern crate flate2;
extern crate reqwest;

use std::path::{Path, PathBuf};
use std::{fs, io};

pub fn download(base_path: &str, base_url: String, online_files: Vec<&str>) -> Result<(), String> {
    let download_dir = PathBuf::from(base_path);
    if !download_dir.exists() {
        println!(
            "Download directory {} does not exists. Creating....",
            download_dir.display()
        );
        fs::create_dir_all(&download_dir).or_else(|e| {
            Err(format!(
                "Failed to create directory {:?}: {:?}",
                download_dir, e
            ))
        })?;
    }

  //parallelize?
  for file in online_files.iter() {
    let res = single_download(&download_dir, base_url.clone(), file);
    match res {
      Ok(()) => continue,
      Err(e) => eprintln!("{}",e),
    }
  }
  Ok(())
}


fn single_download(download_dir: &Path, base_url: String, archive: &str) -> Result<(), String> {

    let url = format!("{}/{}", base_url, archive);

    let file_name = download_dir.join(&archive);
    
    if file_name.exists() {
        println!(
            "  File {:?} already exists, skipping downloading.",
            file_name
        );
    } else {
        println!("  Downloading {} to {:?}...", url, download_dir);
        let f = fs::File::create(&file_name)
            .or_else(|e| Err(format!("Failed to create file {:?}: {:?}", file_name, e)))?;
        let mut writer = io::BufWriter::new(f);
        let mut response =
            reqwest::blocking::get(&url).expect(format!("Failed to download {:?}", url).as_str());

        let _ = io::copy(&mut response, &mut writer).or_else(|e| {
            Err(format!(
                "Failed to to write to file {:?}: {:?}",
                file_name, e
            ))
        })?;
        println!("Downloading or {} to {:?} done!", archive, download_dir);
    }
    Ok(())
}
