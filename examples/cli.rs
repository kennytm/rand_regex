use rand::Rng;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let pattern = std::env::args().nth(1).expect("give me a regex pattern");
    let n = std::env::args()
        .nth(2)
        .and_then(|arg| arg.parse().ok())
        .unwrap_or(1);
    let pattern = rand_regex::Regex::compile(&pattern, 1)?;
    for result in rand::rng().sample_iter::<String, _>(pattern).take(n) {
        println!("{}", result);
    }
    Ok(())
}
