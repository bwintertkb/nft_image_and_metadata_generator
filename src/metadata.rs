/// The high level metadata representation of the NFT collection.
/// - ```collection_name```: Name of the NFT collection.
/// - ```symbol```: Symbol used to identify the NFT collection.
/// - ```description```: Description of the NFT collection.
/// - ```seller_fee_basis_points```: Define how much % you want from secondary market sales 1000 = 10%
/// - ```external_url```: External url provided in the metadata.
/// - ```creators```: Vector containing ```Creator``` structs instances.

#[derive(Debug, Clone)]
pub struct MetadataHeader<'a> {
    /// Name of the NFT collection
    pub collection_name: String,
    symbol: &'a str,
    description: &'a str,
    seller_fee_basis_points: u32,
    external_url: &'a str,
    /// Vector containing ```Creator``` structs instances.
    pub creators: Vec<Creator<'a>>,
}

impl<'a> MetadataHeader<'a> {
    /// Returns a ```MetadataHeader``` instance
    pub fn new(
        collection_name: String,
        symbol: &'a str,
        description: &'a str,
        seller_fee_basis_points: u32,
        external_url: &'a str,
        creators: Vec<Creator<'a>>,
    ) -> MetadataHeader<'a> {
        MetadataHeader {
            collection_name: collection_name,
            symbol: symbol,
            description: description,
            seller_fee_basis_points: seller_fee_basis_points,
            external_url: external_url,
            creators: creators,
        }
    }
}

/// ```Creator``` object used to define where the distribution of assets will go.
/// Only important for Solana, for Ethereum you can fill these fields with random values of the correct type.
/// ```address``` refers to the Solana address of the payee wallet. ```share``` is the percentage of Sol to be
/// sent to the address. Note that for multiple ```Creator``` objects the sum of the shares must equal 100.
#[derive(Debug, Clone)]
pub struct Creator<'a> {
    /// Solana address of the payee wallet
    pub address: &'a str,
    /// Share of the Sol to be sent to the address, e.g. a value of 80 corresponds to 80%.
    pub share: u8,
}

impl<'a> Creator<'a> {
    /// Returns a ```Creator``` instance
    pub fn new(address: &'a str, share: u8) -> Creator {
        Creator {
            address: address,
            share: share,
        }
    }
}

/// Attributes related to the NFT. This is automatically generated.
#[derive(Debug, Clone)]
pub struct Attributes {
    trait_type: String,
    value: String,
}

impl Attributes {
    /// Returns a ```Attributes``` instance
    pub fn new(trait_type: String, value: String) -> Attributes {
        Attributes {
            trait_type: trait_type,
            value: value,
        }
    }
}

/// Properties related to the NFT. This is automatically generated.
#[derive(Debug, Clone)]
pub struct Properties<'a> {
    /// Returns a ```Properties``` instance
    files: Vec<Files>,
    category: String,
    creators: Vec<Creator<'a>>,
}

impl<'a> Properties<'a> {
    pub fn new(files: Vec<Files>, category: String, creators: Vec<Creator<'a>>) -> Properties {
        Properties {
            files: files,
            category: category,
            creators: creators,
        }
    }
}

/// File information related to the NFT. This is automatically generated.
#[derive(Debug, Clone)]
pub struct Files {
    uri: String,
    _type: String,
}

impl Files {
    /// Returns a ```Files``` instance
    pub fn new(uri: String, _type: String) -> Files {
        Files {
            uri: uri,
            _type: _type,
        }
    }
}

/// Trait used to convert metadata module objects to JSON objects.
pub trait Json {
    /// Convert the structs in the metadata module to JSON objects.
    fn to_json(&self) -> json::JsonValue;
}

impl<'a> Json for MetadataHeader<'a> {
    fn to_json(&self) -> json::JsonValue {
        let mut metadata = json::JsonValue::new_object();
        metadata["name"] = self.collection_name.clone().into();
        metadata["symbol"] = self.symbol.into();
        metadata["description"] = self.description.into();
        metadata["seller_fee_basis_points"] = self.seller_fee_basis_points.into();
        metadata["external_url"] = self.external_url.into();
        let mut creators: Vec<json::JsonValue> = Vec::new();
        for creator in &self.creators {
            creators.push(creator.to_json());
        }
        metadata["creators"] = creators.into();
        metadata
    }
}

impl<'a> Json for Creator<'a> {
    fn to_json(&self) -> json::JsonValue {
        let mut creator = json::JsonValue::new_object();
        creator["address"] = self.address.into();
        creator["share"] = self.share.into();
        creator
    }
}

impl Json for Attributes {
    fn to_json(&self) -> json::JsonValue {
        let mut att = json::JsonValue::new_object();
        att["trait_type"] = self.trait_type.clone().into();
        att["value"] = self.value.clone().into();
        att
    }
}

impl<'a> Json for Properties<'a> {
    fn to_json(&self) -> json::JsonValue {
        let mut prop = json::JsonValue::new_object();
        let mut files: Vec<json::JsonValue> = Vec::new();
        for file in &self.files {
            files.push(file.to_json());
        }
        prop["files"] = files.into();
        prop["category"] = self.category.clone().into();
        let mut creators: Vec<json::JsonValue> = Vec::new();
        for creator in &self.creators {
            creators.push(creator.to_json());
        }
        prop["creators"] = creators.into();
        prop
    }
}

impl Json for Files {
    fn to_json(&self) -> json::JsonValue {
        let mut files = json::JsonValue::new_object();
        files["uri"] = self.uri.clone().into();
        files["type"] = self._type.clone().into();
        files
    }
}
