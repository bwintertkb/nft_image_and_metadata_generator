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
