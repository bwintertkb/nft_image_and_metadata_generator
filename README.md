# nft_image_and_metadata_generator

Generate NFT images and their respective metadata for the Ethereum and Solana networks. Please note that this is an initial release, although completely functioning, future updates will
revolve around code optimisation and adding functionality.

### Video guide

The video guide can be found here: https://www.youtube.com/watch?v=XgQ2sTE5CfI&ab_channel=Bartek

### Version

```
[dependencies]
nft_image_and_metadata_generator = "0.2.0"
```

### Example

The following is the example in the example directory. Note, that the images and metadata will be written to ./output/assets/images and
./output/assets/metadata. The concatenated metadata will be written to ./output/metadata.

```
use nft_image_and_metadata_generator::{
    metadata::{Creator, MetadataHeader},
    ImageGenerator, Network,
};

fn main() {
    let collection_name = String::from("Shapes");
    let symbol = "TestSymbol";
    let description = "A test for generating NFT images along with the metadata.";
    let seller_fee_basis_points: u32 = 1000;
    let external_url = "https://www.rust-lang.org/";
    let base_uri = "ipfs://{CID}"; // Not important for sol
    let address1 = "Buqs3mX5xS3XQeQBHxVnaazYXGY2tgeV6Gx4npyWG9gd";
    let share1: u8 = 100;
    let creator1: Creator = Creator::new(address1.to_owned(), share1);
    let creators: Vec<Creator> = vec![creator1];
    let metadata: MetadataHeader = MetadataHeader::new(
        collection_name,
        symbol.to_owned(),
        description.to_owned(),
        seller_fee_basis_points,
        external_url.to_owned(),
        creators,
    );

    let path = "G:/rust_nft_gen_crate/nft_image_and_metadata_generator/example/layers";
    let output_path = "./output";
    let network = Network::Sol;
    let layer_order = vec!["Background", "Square", "Circle"];

    let layer_exclusion_probability = vec![0.0, 0.0, 0.2];

    let delimeter = '#';
    let num_assets: u64 = 20;

    let img = ImageGenerator::new(
        path,
        output_path,
        network,
        base_uri,
        layer_order,
        Option::Some(layer_exclusion_probability),
        num_assets,
        delimeter,
        metadata,
    );

    img.generate().unwrap();
}
```

### License

`nft_image_and_metadata_generator` is distributed under the terms of the MIT license.
