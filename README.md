# nft_image_and_metadata_generator

Generate NFT images and their respective metadata for the Ethereum and Solana networks. Please note that this is an initial release, although completely functioning, future updates will
revolve around code optimisation and adding functionality.

### Video guide

The video guide can be found here: https://www.youtube.com/watch?v=XgQ2sTE5CfI&ab_channel=Bartek

### Version

```
[dependencies]
nft_image_and_metadata_generator = "0.1.3"
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
    let collection_name = String::from("Test collection");
    let symbol = "TestSymbol";
    let description = "A test for generating NFT images along with the metadata.";
    let seller_fee_basis_points: u32 = 1000;
    let external_url = "https://www.rust-lang.org/";
    let base_uri = "ipfs://NewUriToReplace"; // Not important for sol
    let address1 = "0xb12044453f400D9fa1a00DD01B10128FE4720723";
    let share1: u8 = 50;
    let creator1: Creator = Creator::new(address1, share1);
    let address2 = "FjKvLgGVB6dPio31yFvBNsw6HFhqxbC9dzRasbHFwZfM";
    let share2: u8 = 50;
    let creator2: Creator = Creator::new(address2, share2);
    let creators: Vec<Creator> = vec![creator1, creator2];
    let metadata: MetadataHeader = MetadataHeader::new(
        collection_name,
        symbol,
        description,
        seller_fee_basis_points,
        external_url,
        creators,
    );

    let path = "./layers";
    let output_path = "./output";
    let network = Network::Eth;
    let layer_order = vec!["Background", "Square", "Circle"];

    let layer_exclusion_probability = vec![0.0, 0.0, 0.2];

    let delimeter = '#';
    let num_assets: u64 = 5;

    let mut img = ImageGenerator::new(
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
