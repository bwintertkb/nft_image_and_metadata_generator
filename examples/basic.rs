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
