//! # NFT image and metadata generator for Etherium and Solana
//!
//! Inspired by hashlips 'nft_image_and_metada_generator' provides functionality to generate NFT images as well as their respective metadata
//! for both the Etherium and Solana network. Note that this is the initial release, subsequent releases will contain performance improvements
//! and general code optimizations.

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
use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::Path;

extern crate image;
extern crate rand;
extern crate sha256;

const DEFAULT_ASSET_DIR_NAME: &str = "assets";
const DEFAULT_METADATA_DIR_NAME: &str = "metadata";

/// Enum to distinguish between the Etherium and Solana networks
#[derive(Debug, Clone)]
pub enum Network {
    /// Etherium network
    Eth,
    /// Solana network
    Sol,
}

/// ```ImageGenerator``` struct is used to generate the NFTs and respective metadata.
/// Attributes for the ImageGenerator struct:
/// - ```root_asset_path```: Path to the assets directory
/// - ```output_path```: Path to where the generated NFTs and metadata will be stored. The assets with individual
/// asset metadata will be stored in the created path /output_path/assets and combined metadata /output__path/metadata
/// - ```network```: Expects the Network enum, defining which network the NFTs are created for
/// - ```base_uri```: Base URI for the NFTs, important for Etherium, ignore for Solana.
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
    metadata_header: MetadataHeader<'a>,
    idx: u64,
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
        metadata_header: MetadataHeader<'a>,
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

        ImageGenerator {
            root_asset_path: root_asset_path,
            output_path: output_path,
            network: network,
            base_uri: base_uri,
            layer_order: layer_order,
            layer_exclusion_prob: layer_exclusion_prob,
            num_assets: num_assets,
            delimeter: delimeter,
            metadata_header: metadata_header,
            idx: ImageGenerator::get_start_index(_network),
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

    fn get_start_index(network: Network) -> u64 {
        match network {
            Network::Eth => {
                return 1;
            }
            Network::Sol => {
                return 0;
            }
        }
    }

    fn layer_img_store(
        &self,
        paths: &Vec<String>,
        cache: &mut cache::ImageCache,
    ) -> Vec<image::DynamicImage> {
        let mut img_store: Vec<image::DynamicImage> = Vec::new();
        for path in paths.into_iter() {
            if let Ok(image) = cache.query(path) {
                img_store.push(image.clone());
            } else {
                // Not in cache, load image from disk
                cache.update(path.to_string(), image::open(path).unwrap());
                img_store.push(cache.query(&path[..]).unwrap().clone());
            }
        }
        img_store
    }
    fn get_value_from_asset(&self, asset_name: &String) -> String {
        let split_asset_name: Vec<&str> = asset_name.split(self.delimeter).map(|x| x).collect();
        String::from(split_asset_name[0])
    }

    fn set_up_output_directory(&self, data_type: String) -> std::io::Result<()> {
        let mut path = String::from(self.output_path);
        if Path::new(&path).exists() == false {
            std::fs::create_dir(&path)?;
        }
        path.push(std::path::MAIN_SEPARATOR);
        path.push_str(&data_type);
        if Path::new(&path).exists() == false {
            std::fs::create_dir(&path)?;
        }

        Ok(())
    }

    fn output_metadata_to_disk(
        &self,
        data_type: String,
        file_name: String,
        metadata: &String,
    ) -> std::io::Result<()> {
        self.set_up_output_directory(data_type.clone())?;
        let mut path = String::from(self.output_path);
        path.push(std::path::MAIN_SEPARATOR);
        path.push_str(&data_type);
        path.push(std::path::MAIN_SEPARATOR);
        path.push_str(&file_name);
        std::fs::write(path, metadata)?;

        Ok(())
    }

    fn format_full_metadata(&self, data: String) -> String {
        let mut s_bracket = String::from("[");
        let e_bracket = String::from("]");
        s_bracket.push_str(&data);
        s_bracket.push_str(&e_bracket);
        s_bracket
    }

    fn sol_metadata(&mut self, asset_paths: &Vec<String>, hash: String) -> json::JsonValue {
        let mut metadata = self.metadata_header.clone();
        let append_to_name = format!(" {}{}", self.delimeter, self.idx);
        metadata.collection_name.push_str(&append_to_name);
        let creators = metadata.creators.clone();
        let mut img_metadata = metadata.to_json();
        img_metadata["image"] = format!("{}.png", self.idx).into();
        img_metadata["edition"] = self.idx.into();
        img_metadata["image_sha256"] = hash.into();
        img_metadata.remove("creators");
        let mut att_store: Vec<json::JsonValue> = Vec::new();
        for path in asset_paths {
            let trait_type = util::get_subdirectory(&path[..], 2 as usize);
            let value = self.get_value_from_asset(&util::get_subdirectory(&path[..], 1 as usize));
            let att: Attributes = Attributes::new(trait_type, value);
            att_store.push(att.to_json());
        }
        let files = Files::new(format!("{}.png", self.idx), String::from("image/png"));
        let category = String::from("image");
        let prop = Properties::new(vec![files], category, creators);
        let prop_json = prop.to_json();
        img_metadata["attributes"] = att_store.into();
        img_metadata["properties"] = prop_json.into();
        img_metadata
    }
    fn eth_metadata(&mut self, asset_paths: &Vec<String>, hash: String) -> json::JsonValue {
        let mut metadata = self.metadata_header.clone();
        let append_to_name = format!(" {}{}", self.delimeter, self.idx);
        metadata.collection_name.push_str(&append_to_name);
        let mut img_metadata = metadata.to_json();
        img_metadata["image"] = format!("{}{}{}.png", self.base_uri, '/', self.idx).into();
        img_metadata["edition"] = self.idx.into();
        img_metadata["image_sha256"] = hash.into();
        img_metadata["name"] = format!("{}{}", self.delimeter, self.idx).into();
        img_metadata.remove("creators");
        let mut att_store: Vec<json::JsonValue> = Vec::new();
        for path in asset_paths {
            let trait_type = util::get_subdirectory(&path[..], 2 as usize);
            let value = self.get_value_from_asset(&util::get_subdirectory(&path[..], 1 as usize));
            let att: Attributes = Attributes::new(trait_type, value);
            att_store.push(att.to_json());
        }
        img_metadata["attributes"] = att_store.into();
        img_metadata.remove("symbol");
        img_metadata.remove("description");
        img_metadata.remove("seller_fee_basis_points");
        img_metadata
    }

    fn gen_img_metadata(
        &mut self,
        asset_paths: &Vec<Vec<String>>,
        output_paths: &Vec<String>,
    ) -> std::io::Result<()> {
        print!("\n");
        let mut full_metadata: Vec<String> = Vec::new();
        for (idx, asset_path) in asset_paths.into_iter().enumerate() {
            print!("\rSaving metadata to disk: {}/{}", idx + 1, self.num_assets);
            std::io::stdout().flush().unwrap();
            let p = &output_paths[idx];
            let hash = sha256::digest_file(p).unwrap();
            let metadata = match self.network {
                Network::Eth => self.eth_metadata(asset_path, hash),
                Network::Sol => self.sol_metadata(asset_path, hash),
            };
            let data = metadata.to_string();
            self.output_metadata_to_disk(
                String::from(DEFAULT_ASSET_DIR_NAME),
                format!("{}.json", self.idx),
                &data,
            )?;
            self.idx += 1;
            full_metadata.push(data);
        }
        let full_metadata = self.format_full_metadata(String::from_iter(full_metadata));

        self.output_metadata_to_disk(
            String::from(DEFAULT_METADATA_DIR_NAME),
            String::from("_metadata.json"),
            &full_metadata,
        )?;

        Ok(())
    }

    /// Function used to generate the NFTs and metadata. Returns a ```Result``` enum.
    /// The ```Ok``` result will contain an empty tuple. The error result will most likely be
    /// and ```std::io::Error```, generally because paths cannot be read.
    pub fn generate(&mut self) -> Result<(), std::io::Error> {
        let mixed_asset_paths = self.gen_mix_assets_paths();
        let mix_assets_paths_img = mixed_asset_paths.clone();
        let mut output_paths: Vec<String> = vec![String::new(); mix_assets_paths_img.len()];
        let mut cache = cache::ImageCache::new();
        self.set_up_output_directory(DEFAULT_ASSET_DIR_NAME.to_string())
            .unwrap();
        for (idx, asset_paths) in mix_assets_paths_img.into_iter().enumerate() {
            let img_store = self.layer_img_store(&asset_paths, &mut cache);
            let mut base_img = img_store[0].clone();
            for img in img_store.into_iter() {
                image::imageops::overlay(&mut base_img, &img, 0, 0);
            }

            let output_path = util::create_path(vec![
                &String::from(self.output_path),
                &String::from(DEFAULT_ASSET_DIR_NAME),
                &format!("{}.png", self.idx),
            ]);

            base_img.save(output_path.clone()).unwrap();
            output_paths[idx] = output_path;
            print!(
                "\rGenerating images, pct. complete (image): {:.2}%",
                ((idx as f32 + 1.) / self.num_assets as f32) * 100.
            );
            std::io::stdout().flush().unwrap();
            self.idx += 1;
        }
        self.idx = ImageGenerator::get_start_index(self.network.clone());
        self.gen_img_metadata(&mixed_asset_paths, &output_paths)?;

        Ok(())
    }

    fn get_weight_from_file_name(&self, file_name: &String) -> f32 {
        let reg_exp = &format!(r"{}([0-9\\.]*)[^\\.a-z]", self.delimeter)[..];
        let re = regex::Regex::new(reg_exp).unwrap();
        if file_name.contains(self.delimeter) == false {
            return 1.0;
        }
        let mut w: f32 = 0.0;
        for cap in re.captures_iter(&file_name[..]) {
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
        let mut weights: Vec<f32> = vec![0.0; _layer.len()];
        for (idx, l) in _layer.into_iter().enumerate() {
            weights[idx] = self.get_weight_from_file_name(l);
        }
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

    fn get_weighted_index(&self, prob_range: Vec<(f32, f32)>) -> usize {
        // Generate uniform dist. num and get the index from prob_range
        let mut rng = rand::thread_rng();
        let rand = rng.gen::<f32>();
        let mut _idx: usize = 0;
        for (idx, range) in prob_range.into_iter().enumerate() {
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

    fn get_weighted_asset(&self, layer: &Vec<String>) -> usize {
        let weights = self.get_asset_weights(layer);
        let prob = self.get_prob_from_weights(&weights);
        let prob_range = self.create_prob_range_vec(prob);
        self.get_weighted_index(prob_range)
    }

    fn create_asset_path(&self, assets: &HashMap<String, Vec<String>>) -> (Vec<String>, String) {
        let mut asset_path: Vec<String> = Vec::new();
        let mut concat_path = String::new();

        let mut _idx: usize = 0;
        for odr in &self.layer_order {
            if assets.contains_key(*odr) == false {
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
            let _asset_name = _layer[self.get_weighted_asset(_layer)].clone();
            let full_asset_path = util::create_path(vec![
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

    fn gen_mix_assets_paths(&self) -> Vec<Vec<String>> {
        // Generate the assets that will be mixed
        let assets = self.get_assets();
        let mut _assets_path_store: Vec<Vec<String>> = vec![Vec::new(); self.num_assets as usize];

        let mut ctr: usize = 0;
        let mut dna_store: Vec<String> = vec![String::new(); self.num_assets as usize];
        while ctr < _assets_path_store.len() {
            print!("\rFound: {}/{} unique assets", ctr + 1, self.num_assets);
            std::io::stdout().flush().unwrap();
            let (path, hash) = self.create_asset_path(&assets);
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
            let key = util::get_subdirectory(&p[..], 1 as usize);
            let inner_paths = fs::read_dir(&p[..]).unwrap();
            let mut img_name_store: Vec<String> = Vec::new();
            for inner_path in inner_paths {
                let p = inner_path
                    .unwrap()
                    .path()
                    .into_os_string()
                    .into_string()
                    .unwrap();
                let img_name = util::get_subdirectory(&p[..], 1 as usize);
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

mod cache {
    use std::collections::HashMap;

    pub struct ImageCache {
        data: HashMap<String, image::DynamicImage>,
    }

    impl ImageCache {
        pub fn new() -> ImageCache {
            let data: HashMap<String, image::DynamicImage> = HashMap::new();
            ImageCache { data: data }
        }

        pub fn query(&self, key: &str) -> Result<&image::DynamicImage, String> {
            if self.data.contains_key(key) {
                return Ok(self.data.get(key).unwrap());
            }
            Err("Key not found, adding it to cache...".to_string())
        }

        pub fn update(&mut self, key: String, data: image::DynamicImage) {
            self.data.insert(key, data);
        }
    }
}
