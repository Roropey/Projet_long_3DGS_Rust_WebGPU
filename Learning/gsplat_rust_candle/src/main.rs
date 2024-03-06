use clap::Parser;
use std::path::PathBuf;
use gsplat_rust_candle::trainer;

#[derive(Parser, Debug)]
#[command(version = "1.0", about = "gsplat parameters", long_about = None)]
struct Args {
    /// Height of the image
    #[arg(long, default_value_t = 256)]
    height: u32,

    /// Width of the image
    #[arg(long, default_value_t = 256)]
    width: u32,

    /// Number of points
    #[arg(long, default_value_t = 100000)]
    num_points: u32,

    /// Flag to save images
    #[arg(long, default_value_t = true)]
    save_imgs: bool,

    /// Path to save the image
    #[arg(long, default_value = "image.png")]
    img_path: PathBuf,

    /// Number of iterations
    #[arg(long, default_value_t = 1000)]
    iterations: isize,

    /// Learning rate
    #[arg(long, default_value_t = 0.01)]
    lr: f64,
}

fn main() {
    let args = Args::parse();

    trainer::custom_main(Some(args.height), Some(args.width), Some(args.num_points), Some(args.save_imgs), &args.img_path, Some(args.iterations), Some(args.lr));
}