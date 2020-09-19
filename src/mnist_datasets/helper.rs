use byteorder::{BigEndian, ReadBytesExt};
use std::fs::File;
use std::io::prelude::*;
use std::path::Path;

static IMG_MAGIC_NUMBER: u32 = 0x0000_0803;
static LBL_MAGIC_NUMBER: u32 = 0x0000_0801;
static ROWS: usize = 28;
static COLS: usize = 28;

pub fn labels(path: &Path, expected_length: u32) -> Vec<u8> {
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

pub fn images(path: &Path, expected_length: u32) -> Vec<u8> {
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
