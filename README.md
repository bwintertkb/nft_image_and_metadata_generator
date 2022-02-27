# nft_image_and_metadata_generator

Generate NFT images and their respective metadata for the Ethereum and Solana networks. Please note that this is an initial release, although completely functioning, future updates will
revolve around code optimisation and adding functionality.

### Version

```
[dependencies]
nft_image_and_metadata_generator = "0.1.1"
```

### Example

The following is an example using hashlips art engine assets (see https://github.com/HashLips/hashlips_art_engine.git).

```
use nft_image_and_metadata_generator::{
    metadata::{Creator, MetadataHeader},
    ImageGenerator, Network,
};
fn main() {
    let collection_name = String::from("Test collection");
    let symbol = "TestSymbol";
    let description = "A test for generating NFT images along with the metadata.";
    let seller_fee_basis_points: u32 = 1000; // Define how much % you want from secondary market sales 1000 = 10%
    let external_url = "https://www.rust-lang.org/";
    let base_uri = "ipfs://NewUriToReplace"; // Not important for sol
    let address1 = "7fXNuer5sbZtaTEPhtJ5g5gNtuyRoKkvxdjEjEnPN4mC";
    let share1: u8 = 50;
    let creator1: Creator = Creator::new(address1, share1);
    let address2 = "99XNuer5sbZtaTEPhtJ5g5gNtuyRoKkvxdjEjEnPN4mC";
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
    let path = "../layers";
    let output_path = "./output";
    let network = Network::Sol;
    let layer_order = vec![
        "Background",
        "Eyeball",
        "Eye color",
        "Goo",
        "Iris",
        "Shine",
        "Bottom lid",
        "Top lid",
    ];
    let layer_exclusion_probability = vec![0.0, 0.8, 0.0, 0.8, 0.5, 0.0, 0.0, 0.0];
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
    img.generate();
}
```

### License

`nft_image_and_metadata_generator` is distributed under the terms of the MIT license.
