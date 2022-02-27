use std::path::PathBuf;

pub fn get_subdirectory(path: &str, offset: usize) -> String {
    let p = String::from(path);
    let split_path: Vec<&str> = p.split(std::path::MAIN_SEPARATOR).map(|x| x).collect();
    String::from(split_path[split_path.len() - offset])
}

pub fn create_path(path_elements: Vec<&String>) -> String {
    // Generate OS agnostic path String object
    let mut path = PathBuf::default();
    for p in path_elements.iter() {
        path = path.join(p);
    }
    path.into_os_string().into_string().unwrap()
}