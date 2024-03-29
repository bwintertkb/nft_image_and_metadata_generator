/// The high level metadata representation of the NFT collection.
/// - ```collection_name```: Name of the NFT collection.
/// - ```symbol```: Symbol used to identify the NFT collection.
/// - ```description```: Description of the NFT collection.
/// - ```seller_fee_basis_points```: Define how much % you want from secondary market sales 1000 = 10%
/// - ```external_url```: External url provided in the metadata.
/// - ```creators```: Vector containing ```Creator``` structs instances.

#[derive(Debug, Clone)]
pub struct MetadataHeader {
    /// Name of the NFT collection
    pub collection_name: String,
    pub symbol: String,
    pub description: String,
    seller_fee_basis_points: u32,
    external_url: String,
    /// Vector containing ```Creator``` structs instances.
    pub creators: Vec<Creator>,
}

impl MetadataHeader {
    /// Returns a ```MetadataHeader``` instance
    pub fn new(
        collection_name: String,
        symbol: String,
        description: String,
        seller_fee_basis_points: u32,
        external_url: String,
        creators: Vec<Creator>,
    ) -> MetadataHeader {
        MetadataHeader {
            collection_name,
            symbol,
            description,
            seller_fee_basis_points,
            external_url,
            creators,
        }
    }
}

/// ```Creator``` object used to define where the distribution of assets will go.
/// Only important for Solana, for Ethereum you can fill these fields with random values of the correct type.
/// ```address``` refers to the Solana address of the payee wallet. ```share``` is the percentage of Sol to be
/// sent to the address. Note that for multiple ```Creator``` objects the sum of the shares must equal 100.
#[derive(Debug, Clone)]
pub struct Creator {
    /// Solana address of the payee wallet
    pub address: String,
    /// Share of the Sol to be sent to the address, e.g. a value of 80 corresponds to 80%.
    pub share: u8,
}

impl Creator {
    /// Returns a ```Creator``` instance
    pub fn new(address: String, share: u8) -> Creator {
        Creator { address, share }
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
        Attributes { trait_type, value }
    }
}

/// Properties related to the NFT. This is automatically generated.
#[derive(Debug, Clone)]
pub struct Properties {
    /// Returns a ```Properties``` instance
    files: Vec<Files>,
    category: String,
    creators: Vec<Creator>,
}

impl Properties {
    pub fn new(files: Vec<Files>, category: String, creators: Vec<Creator>) -> Properties {
        Properties {
            files,
            category,
            creators,
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
        Files { uri, _type }
    }
}

/// Trait used to convert metadata module objects to JSON objects.
pub trait Json {
    /// Convert the structs in the metadata module to JSON objects.
    fn to_json(&self) -> json::JsonValue;
}

impl Json for MetadataHeader {
    fn to_json(&self) -> json::JsonValue {
        let mut metadata = json::JsonValue::new_object();
        metadata["name"] = self.collection_name.clone().into();
        metadata["symbol"] = self.symbol.clone().into();
        metadata["description"] = self.description.clone().into();
        metadata["seller_fee_basis_points"] = self.seller_fee_basis_points.into();
        metadata["external_url"] = self.external_url.clone().into();
        let mut creators: Vec<json::JsonValue> = Vec::new();
        for creator in &self.creators {
            creators.push(creator.to_json());
        }
        metadata["creators"] = creators.into();
        metadata
    }
}

impl Json for Creator {
    fn to_json(&self) -> json::JsonValue {
        let mut creator = json::JsonValue::new_object();
        creator["address"] = self.address.clone().into();
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

impl Json for Properties {
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
