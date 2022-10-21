#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_doctest_main)]
//! # NFT image and metadata generator for Ethereum and Solana
//!
//! Inspired by hashlips 'nft_image_and_metada_generator' provides functionality to generate NFT images as well as their respective metadata
//! for both the Ethereum and Solana network. Note that this is the initial release, subsequent releases will contain performance improvements
//! and general code optimizations.
//!
//! Github: ```https://github.com/bwintertkb/nft_image_and_metadata_generator```

//! # Example of how to generate NFT images and metadata.
//! The following code demonstrates how to generate the NFT images along with metadata.
//! ```
//! use nft_image_and_metadata_generator::{
//!     metadata::{Creator, MetadataHeader},
//!     ImageGenerator, Network,
//! };
//!
//! fn main() {
//!     let collection_name = String::from("Test collection");
//!     let symbol = "TestSymbol";
//!     let description = "A test for generating NFT images along with the metadata.";
//!     let seller_fee_basis_points: u32 = 1000; // Define how much % you want from secondary market sales 1000 = 10%
//!     let external_url = "https://www.rust-lang.org/";
//!     let base_uri = "ipfs://NewUriToReplace"; // Not important for sol
//!     let address1 = "7fXNuer5sbZtaTEPhtJ5g5gNtuyRoKkvxdjEjEnPN4mC";
//!     let share1: u8 = 50;
//!     let creator1: Creator = Creator::new(address1, share1);
//!     let address2 = "99XNuer5sbZtaTEPhtJ5g5gNtuyRoKkvxdjEjEnPN4mC";
//!     let share2: u8 = 50;
//!     let creator2: Creator = Creator::new(address2, share2);
//!     let creators: Vec<Creator> = vec![creator1, creator2];
//!     let metadata: MetadataHeader = MetadataHeader::new(
//!         collection_name,
//!         symbol,
//!         description,
//!         seller_fee_basis_points,
//!         external_url,
//!         creators,
//!     );
//!
//!     let path = "../layers";
//!     let output_path = "./output";
//!     let network = Network::Sol;
//!     let layer_order = vec![
//!         "Background",
//!         "Eyeball",
//!         "Eye color",
//!         "Goo",
//!         "Iris",
//!         "Shine",
//!         "Bottom lid",
//!         "Top lid",
//!     ];
//!
//!     let layer_exclusion_probability = vec![0.0, 0.8, 0.0, 0.8, 0.5, 0.0, 0.0, 0.0];
//!
//!     let delimeter = '#';
//!     let num_assets: u64 = 5;
//!
//!     let mut img = ImageGenerator::new(
//!         path,
//!         output_path,
//!         network,
//!         base_uri,
//!         layer_order,
//!         Option::Some(layer_exclusion_probability),
//!         num_assets,
//!         delimeter,
//!         metadata,
//!     );
//!     img.generate();
//! }
//! ```

/// A collection of structs which will be used to build the metadata for the NFTs.
/// The only structs that the user needs to care about is ```MetadataHeader``` and ```Creator```.
pub mod metadata;
mod util;

use crate::metadata::{Attributes, Files, Json, MetadataHeader, Properties};
use crate::rand::Rng;
use rayon::prelude::*;
use std::{collections::HashMap, fs, io::Write, sync::Arc, thread, thread::JoinHandle};

extern crate image;
extern crate rand;
extern crate sha256;

const DEFAULT_ASSET_DIR_NAME: &str = "assets";
const DEFAULT_ASSET_SUBDIR_IMG_NAME: &str = "images";
const DEFAULT_ASSET_SUBDIR_METADATA_NAME: &str = "metadata";
const DEFAULT_METADATA_DIR_NAME: &str = "metadata";

/// Enum to distinguish between the Ethereum and Solana networks
#[derive(Debug, Clone)]
pub enum Network {
    /// Ethereum network
    Eth,
    /// Solana network
    Sol,
}

impl Network {
    fn start_index(&self) -> usize {
        match self {
            Network::Eth => 1,
            Network::Sol => 0,
        }
    }
}

/// ```ImageGenerator``` struct is used to generate the NFTs and respective metadata.
/// Attributes for the ImageGenerator struct:
/// - ```root_asset_path```: Path to the assets directory
/// - ```output_path```: Path to where the generated NFTs and metadata will be stored. The assets with individual
/// asset metadata will be stored in the created path /output_path/assets and combined metadata /output__path/metadata
/// - ```network```: Expects the Network enum, defining which network the NFTs are created for
/// - ```base_uri```: Base URI for the NFTs, important for Ethereum, ignore for Solana.
/// - ```layer_order```: Order in which the NFTs will be layered, e.g. ["background", "layer1", "layer2"] will use a "background" image as
///  the first layer, "layer1" image as the second layer and "layer2" image as the third layer
/// - ```layer_exclusion_prob```: Defines the probability in which a particular layer will be excluded, e.g. [0.0, 0.0, 0.1] will mean the third layer
/// has a 10% chance of being excluded from the final NFT image.
/// - ```num_assets```: Number of unique assets to be generated
/// - ```delimeter```: Delimeter used to specify the relative weighting between assets, e.g. if an asset folder contained assets named Blue#1, Red#50, the
/// delimeter would be # and with Red being 49 times more likely to be chosen than Blue.
/// - ```metadata_header```: Expected a MetadataHeader struct to define the high level information for the NFT collection
/// - ```idx```: Variable used to define the starting index of the metadata, users need not worry about this.
#[derive(Debug, Clone)]
pub struct ImageGenerator<'a> {
    root_asset_path: &'a str,
    output_path: &'a str,
    network: Network,
    base_uri: &'a str,
    layer_order: Vec<&'a str>,
    layer_exclusion_prob: Vec<f32>,
    num_assets: u64,
    delimeter: char,
    metadata_header: MetadataHeader,
    idx: usize,
}

impl<'a> ImageGenerator<'a> {
    /// Returns an initialized ```ImageGenerator``` instance
    pub fn new(
        root_asset_path: &'a str,
        output_path: &'a str,
        network: Network,
        base_uri: &'a str,
        layer_order: Vec<&'a str>,
        layer_exclusion_prob: Option<Vec<f32>>,
        num_assets: u64,
        delimeter: char,
        metadata_header: MetadataHeader,
    ) -> ImageGenerator<'a> {
        let _network = network.clone();
        let layer_exclusion_prob = ImageGenerator::verify_layer_exclusion_probability(
            layer_order.len(),
            layer_exclusion_prob,
        );
        if layer_exclusion_prob.len() != layer_order.len() {
            panic!(
                "Layer order length ({}) not equal to layer exclusion probability length ({})",
                layer_order.len(),
                layer_exclusion_prob.len()
            );
        };

        let idx = network.start_index();
        ImageGenerator {
            root_asset_path,
            output_path,
            network,
            base_uri,
            layer_order,
            layer_exclusion_prob,
            num_assets,
            delimeter,
            metadata_header,
            idx,
        }
    }

    fn verify_layer_exclusion_probability(
        num_layers: usize,
        layer_exclusion_probability: Option<Vec<f32>>,
    ) -> Vec<f32> {
        if let Some(val) = layer_exclusion_probability {
            return val;
        }
        vec![0.0; num_layers]
    }

    fn set_up_output_directory(&self) -> std::io::Result<()> {
        let path = self.output_path.to_owned();
        util::create_dir(&util::generate_path(vec![&path]))?;
        util::create_dir(&util::generate_path(vec![
            &path,
            &DEFAULT_ASSET_DIR_NAME.to_owned(),
        ]))?;
        util::create_dir(&util::generate_path(vec![
            &path,
            &DEFAULT_METADATA_DIR_NAME.to_owned(),
        ]))?;
        util::create_dir(&util::generate_path(vec![
            &path,
            &DEFAULT_ASSET_DIR_NAME.to_owned(),
            &DEFAULT_ASSET_SUBDIR_IMG_NAME.to_owned(),
        ]))?;
        util::create_dir(&util::generate_path(vec![
            &path,
            &DEFAULT_ASSET_DIR_NAME.to_owned(),
            &DEFAULT_ASSET_SUBDIR_METADATA_NAME.to_owned(),
        ]))?;

        Ok(())
    }

    fn get_output_paths(&self, num_paths: usize) -> Vec<String> {
        let mut output_paths = vec![String::new(); num_paths];
        let mut idx = self.idx;
        for i in 0..num_paths {
            let output_path = util::generate_path(vec![
                &String::from(self.output_path),
                &String::from(DEFAULT_ASSET_DIR_NAME),
                &String::from(DEFAULT_ASSET_SUBDIR_IMG_NAME),
                &format!("{}.png", idx),
            ]);
            output_paths[i] = output_path;
            idx += 1;
        }
        output_paths
    }

    /// Function used to generate the NFTs and metadata. Returns a ```Result``` enum.
    /// The ```Ok``` result will contain an empty tuple. The error result will most likely be
    /// and ```std::io::Error```, generally because paths cannot be read.
    pub fn generate(&self) -> Result<(), std::io::Error> {
        util::remove_dir_all(self.output_path)?;

        let mixed_asset_paths = self.gen_mix_assets_paths();
        let output_paths: Vec<String> = self.get_output_paths(mixed_asset_paths.len());
        self.set_up_output_directory()?;

        let num = self.num_assets as usize;
        let img_handler = Arc::new(ImageHandler::new(
            num,
            mixed_asset_paths,
            output_paths,
            self.network.clone(),
            self.output_path.to_owned(),
            self.delimeter,
            self.idx,
            self.metadata_header.clone(),
            self.base_uri.to_owned(),
        ));
        let img_handler2 = img_handler.clone();
        let img_handler3 = img_handler.clone();
        let handle = thread::spawn(move || {
            img_handler.display_progress();
        });
        let handle2 = thread::spawn(move || {
            img_handler2.image_to_disk();
        });
        let handle3 = thread::spawn(move || {
            img_handler3.gen_img_metadata().unwrap();
        });
        let threads: [JoinHandle<()>; 3] = [handle, handle2, handle3];
        for h in threads.into_iter() {
            h.join().unwrap();
        }
        Ok(())
    }

    fn get_weight_from_file_name(&self, file_name: &str) -> f32 {
        let reg_exp = &format!(r"{}([0-9\\.]*)[^\\.a-z]", self.delimeter)[..];
        let re = regex::Regex::new(reg_exp).unwrap();
        if !file_name.contains(self.delimeter) {
            return 1.0;
        }
        let mut w: f32 = 0.0;
        for cap in re.captures_iter(file_name) {
            w = String::from(&cap[0])
                .replace(self.delimeter, "")
                .parse::<f32>()
                .unwrap();
            if w == 0.0 {
                w = 1.0;
            }
        }
        w
    }

    fn get_asset_weights(&self, _layer: &Vec<String>) -> Vec<f32> {
        let mut weights: Vec<f32> = Vec::with_capacity(_layer.len());
        _layer
            .iter()
            .for_each(|l| weights.push(self.get_weight_from_file_name(l)));
        weights
    }

    fn get_prob_from_weights(&self, weights: &Vec<f32>) -> Vec<f32> {
        let mut prob: Vec<f32> = vec![0.0; weights.len()];
        let weights_sum: f32 = weights.iter().sum();
        for i in 0..weights.len() {
            prob[i] = weights[i] / weights_sum;
        }
        prob
    }

    fn create_prob_range_vec(&self, prob: Vec<f32>) -> Vec<(f32, f32)> {
        let mut prob_range: Vec<(f32, f32)> = vec![(0.0, 0.0); prob.len()];
        let mut current_range: f32 = 0.0;
        for (idx, val) in prob.into_iter().enumerate() {
            prob_range[idx] = (current_range, current_range + val);
            current_range += val;
        }
        prob_range
    }

    fn get_weighted_index(&self, prob_range: &[(f32, f32)]) -> usize {
        // Generate uniform dist. num and get the index from prob_range
        let mut rng = rand::thread_rng();
        let rand = rng.gen::<f32>();
        let mut _idx: usize = 0;
        for (idx, range) in prob_range.iter().enumerate() {
            if rand >= range.0 && rand < range.1 {
                return idx;
            }
        }
        _idx
    }

    fn skip_layer(&self, layer_name: &str) -> bool {
        let idx = self
            .layer_order
            .iter()
            .position(|&l| l == layer_name)
            .unwrap();
        let mut rng = rand::thread_rng();
        let rand = rng.gen::<f32>();
        if rand < self.layer_exclusion_prob[idx] {
            return true;
        }
        false
    }

    fn get_weighted_asset(&self, prob_range: &[(f32, f32)]) -> usize {
        self.get_weighted_index(prob_range)
    }

    fn create_asset_path(
        &self,
        assets: &HashMap<String, Vec<String>>,
        layer_prob_ranges: &HashMap<String, Vec<(f32, f32)>>,
    ) -> (Vec<String>, String) {
        let mut asset_path: Vec<String> = Vec::new();
        let mut concat_path = String::new();

        let mut _idx: usize = 0;
        for odr in &self.layer_order {
            if !assets.contains_key(*odr) {
                panic!(
                    "Layer {} does not exist in assets {:?}",
                    *odr,
                    assets.keys()
                );
            }
            if self.skip_layer(*odr) {
                continue;
            }
            let _layer = assets.get(*odr).unwrap();
            let prob_range = layer_prob_ranges.get(&odr.to_string()).unwrap();
            let _asset_name = _layer[self.get_weighted_asset(prob_range)].clone();
            let full_asset_path = util::generate_path(vec![
                &String::from(self.root_asset_path),
                &String::from(*odr),
                &_asset_name,
            ]);
            concat_path.push_str(&full_asset_path);
            asset_path.push(full_asset_path.clone());
            _idx += 1;
        }
        (asset_path, sha256::digest(concat_path))
    }

    fn create_layer_prob_ranges(
        &self,
        assets: &HashMap<String, Vec<String>>,
    ) -> HashMap<String, Vec<(f32, f32)>> {
        let mut layer_prob_ranges: HashMap<String, Vec<(f32, f32)>> = HashMap::new();
        for (key, val) in assets.iter() {
            let weights = self.get_asset_weights(val);
            let prob = self.get_prob_from_weights(&weights);
            let prob_range = self.create_prob_range_vec(prob);
            layer_prob_ranges.insert(key.to_string(), prob_range);
        }
        layer_prob_ranges
    }

    fn gen_mix_assets_paths(&self) -> Vec<Vec<String>> {
        // Generate the assets that will be mixed
        let assets = self.get_assets();
        let mut _assets_path_store: Vec<Vec<String>> = vec![Vec::new(); self.num_assets as usize];

        let layer_prob_ranges = self.create_layer_prob_ranges(&assets);
        let mut ctr: usize = 0;
        let mut dna_store: Vec<String> = vec![String::new(); self.num_assets as usize];
        while ctr < _assets_path_store.len() {
            print!("\rFound: {}/{} unique assets", ctr + 1, self.num_assets);
            std::io::stdout().flush().unwrap();
            let (path, hash) = self.create_asset_path(&assets, &layer_prob_ranges);
            if dna_store.contains(&hash) {
                continue;
            }
            dna_store[ctr] = hash;
            _assets_path_store[ctr] = path;
            ctr += 1;
        }
        print!("\n");

        _assets_path_store
    }

    /// Returns a hashmap of the asset paths for each asset layer.
    pub fn get_assets(&self) -> HashMap<String, Vec<String>> {
        let mut asset_store: HashMap<String, Vec<String>> = HashMap::new();
        let paths = fs::read_dir(&self.root_asset_path).unwrap();
        for path in paths {
            let p = path.unwrap().path().into_os_string().into_string().unwrap();
            let key = util::get_subdirectory(&p[..], 1);
            let inner_paths = fs::read_dir(&p[..]).unwrap();
            let mut img_name_store: Vec<String> = Vec::new();
            for inner_path in inner_paths {
                let p = inner_path
                    .unwrap()
                    .path()
                    .into_os_string()
                    .into_string()
                    .unwrap();
                let img_name = util::get_subdirectory(&p[..], 1);
                img_name_store.push(img_name);
            }
            asset_store.insert(key, img_name_store);
        }
        asset_store
    }

    /// Ouputs to the console the paths to the assets that the ```ImageGenerator``` has found.
    pub fn print_assets(&self) {
        let paths = fs::read_dir(&self.root_asset_path).unwrap();
        for path in paths {
            let p = path.unwrap().path().into_os_string().into_string().unwrap();
            let inner_paths = fs::read_dir(&p[..]).unwrap();
            for inner_path in inner_paths {
                println!("{}", inner_path.unwrap().path().display());
            }
        }
    }
}

// ImageHandler is a struct used to cache the metadata and images to disk. This struct was designed
// to perform those tasks via multiple threads.
#[derive(Debug, Clone)]
struct ImageHandler {
    num_assets: usize,
    mixed_asset_paths: Vec<Vec<String>>,
    output_paths: Vec<String>,
    network: Network,
    output_path: String,
    delimeter: char,
    idx: usize,
    metadata_header: MetadataHeader,
    base_uri: String,
}

impl ImageHandler {
    fn new(
        num_assets: usize,
        mixed_asset_paths: Vec<Vec<String>>,
        output_paths: Vec<String>,
        network: Network,
        output_path: String,
        delimeter: char,
        idx: usize,
        metadata_header: MetadataHeader,
        base_uri: String,
    ) -> Self {
        Self {
            num_assets,
            mixed_asset_paths,
            output_paths,
            network,
            output_path,
            delimeter,
            idx,
            metadata_header,
            base_uri,
        }
    }

    fn image_to_disk(&self) {
        self.mixed_asset_paths
            .clone()
            .into_par_iter()
            .enumerate()
            .for_each(|(idx, paths)| {
                let img_store = self.layer_img_store(&paths);
                let mut base_img = img_store[0].clone();
                for img in img_store.into_iter() {
                    image::imageops::overlay(&mut base_img, &img, 0, 0);
                }
                base_img.save(self.output_paths[idx].clone()).unwrap();
            });
    }

    fn layer_img_store(&self, paths: &Vec<String>) -> Vec<image::DynamicImage> {
        paths
            .into_par_iter()
            .map(|path| image::open(path).unwrap())
            .collect()
    }

    fn sol_metadata(&self, asset_paths: &[String], idx: usize) -> json::JsonValue {
        let mut metadata = self.metadata_header.clone();
        let append_to_name = format!(" {}{}", self.delimeter, idx);
        metadata.collection_name.push_str(&append_to_name);
        let creators = metadata.creators.clone();
        let mut img_metadata = metadata.to_json();
        img_metadata["image"] = format!("{}.png", idx).into();
        img_metadata["edition"] = idx.into();
        img_metadata.remove("creators");
        let mut att_store: Vec<json::JsonValue> = Vec::new();
        for path in asset_paths {
            let trait_type = util::get_subdirectory(&path[..], 2);
            let value = self.get_value_from_asset(&util::get_subdirectory(&path[..], 1));
            let att: Attributes = Attributes::new(trait_type, value);
            att_store.push(att.to_json());
        }
        let files = Files::new(format!("{}.png", idx), "image/png".to_owned());
        let category = "image".to_owned();
        let prop = Properties::new(vec![files], category, creators);
        let prop_json = prop.to_json();
        img_metadata["attributes"] = att_store.into();
        img_metadata["properties"] = prop_json;
        img_metadata
    }
    fn eth_metadata(&self, asset_paths: &[String], idx: usize) -> json::JsonValue {
        let mut metadata = self.metadata_header.clone();
        let append_to_name = format!(" {}{}", self.delimeter, idx);
        metadata.collection_name.push_str(&append_to_name);
        let mut img_metadata = metadata.to_json();
        img_metadata["image"] = format!("{}{}{}.png", &self.base_uri, '/', idx).into();
        img_metadata["edition"] = idx.into();
        img_metadata.remove("creators");
        let mut att_store: Vec<json::JsonValue> = Vec::new();
        for path in asset_paths {
            let trait_type = util::get_subdirectory(&path[..], 2);
            let value = self.get_value_from_asset(&util::get_subdirectory(&path[..], 1));
            let att: Attributes = Attributes::new(trait_type, value);
            att_store.push(att.to_json());
        }
        img_metadata["attributes"] = att_store.into();
        let compiler: String = "Rust NFT image and metadata generator".to_owned();
        img_metadata["compiler"] = compiler.into();
        img_metadata.remove("symbol");
        img_metadata.remove("seller_fee_basis_points");
        img_metadata
    }

    fn gen_img_metadata(&self) -> std::io::Result<()> {
        let full_metadata: Vec<String> = self
            .mixed_asset_paths
            .clone()
            .into_par_iter()
            .enumerate()
            .map(|(idx, asset_path)| {
                let metadata = match self.network {
                    Network::Eth => self.eth_metadata(&asset_path, self.idx + idx),
                    Network::Sol => self.sol_metadata(&asset_path, self.idx + idx),
                };
                let data = metadata.to_string();
                let path = util::generate_path(vec![
                    &self.output_path.clone(),
                    &String::from(DEFAULT_ASSET_DIR_NAME),
                    &String::from(DEFAULT_ASSET_SUBDIR_METADATA_NAME),
                    &format!("{}.json", self.idx + idx),
                ]);
                self.output_metadata_to_disk(path, &data).unwrap();
                data
            })
            .collect();
        let full_metadata = self.format_full_metadata(full_metadata);
        let path = util::generate_path(vec![
            &self.output_path.clone(),
            &String::from(DEFAULT_METADATA_DIR_NAME),
            &String::from("_metadata.json"),
        ]);
        self.output_metadata_to_disk(path, &full_metadata)?;

        Ok(())
    }

    fn get_value_from_asset(&self, asset_name: &str) -> String {
        let split_asset_name: Vec<&str> = asset_name.split(self.delimeter).collect();
        String::from(split_asset_name[0])
    }

    fn output_metadata_to_disk(&self, path: String, metadata: &String) -> std::io::Result<()> {
        std::fs::write(path, metadata)?;

        Ok(())
    }

    fn format_full_metadata(&self, data: Vec<String>) -> String {
        let data_len = data.len();
        let mut data_store: Vec<String> = vec![String::new(); data_len];
        for (idx, mut val) in data.into_iter().enumerate() {
            if idx < data_len - 1 {
                val.push(',');
            }
            data_store[idx] = val
        }
        let str_data_store = String::from_iter(data_store);
        let mut s_bracket = String::from("[");
        let e_bracket = String::from("]");
        s_bracket.push_str(&str_data_store);
        s_bracket.push_str(&e_bracket);
        s_bracket
    }

    fn display_progress(&self) {
        let path_imgs = util::generate_path(vec![
            &self.output_path,
            &DEFAULT_ASSET_DIR_NAME.to_owned(),
            &DEFAULT_ASSET_SUBDIR_IMG_NAME.to_owned(),
        ]);

        let path_mtd = util::generate_path(vec![
            &self.output_path,
            &DEFAULT_ASSET_DIR_NAME.to_owned(),
            &DEFAULT_ASSET_SUBDIR_METADATA_NAME.to_owned(),
        ]);
        let mut count = 0;
        while count < self.num_assets as u64 {
            let path_imgs = fs::read_dir(&path_imgs).unwrap();
            let path_mtd = fs::read_dir(&path_mtd).unwrap();
            let mut total_imgs = 0;
            let mut total_mtd = 0;
            for _ in path_imgs {
                total_imgs += 1;
            }
            for _ in path_mtd {
                total_mtd += 1;
            }
            print!(
                "\rGenerating images and metadata... Images: {}/{}, Metadata: {}/{}",
                total_imgs, self.num_assets, total_mtd, self.num_assets
            );
            std::io::stdout().flush().unwrap();
            count = total_imgs;
        }
        print!("\n");
        println!("Saving concatenated metadata to disk...");
    }
}
