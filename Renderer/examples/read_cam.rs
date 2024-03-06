use std::env::consts;
use std::path::{Path, PathBuf};
use std::fs::File;
use std::io::{self, Read};
use std::collections::HashMap;
use once_cell::sync::Lazy;

#[derive(Debug, Clone)]
struct CameraModel<'a> {
    model_id: u64,
    model_name: &'a str,
    num_params: usize,
}

#[derive(Debug, Clone)]
struct Camera {
    id: u64,
    model: String,
    width: u64,
    height: u64,
    params: Vec<f64>,
}

struct Image {
    id: u64,
    camera_id: u32,
    qvec: [f64;4],
    tvec: [f64;3],
}

// Définir CAMERA_MODELS si nécessaire
const CAMERA_MODELS: [CameraModel; 11] = [
    CameraModel { model_id: 0, model_name: "SIMPLE_PINHOLE", num_params: 3 },
    CameraModel { model_id: 1, model_name: "PINHOLE", num_params: 4 },
    CameraModel { model_id: 2, model_name: "SIMPLE_RADIAL", num_params: 4 },
    CameraModel { model_id: 3, model_name: "RADIAL", num_params: 5 },
    CameraModel { model_id: 4, model_name: "OPENCV", num_params: 8 },
    CameraModel { model_id: 5, model_name: "OPENCV_FISHEYE", num_params: 8 },
    CameraModel { model_id: 6, model_name: "FULL_OPENCV", num_params: 12 },
    CameraModel { model_id: 7, model_name: "FOV", num_params: 5 },
    CameraModel { model_id: 8, model_name: "SIMPLE_RADIAL_FISHEYE", num_params: 4 },
    CameraModel { model_id: 9, model_name: "RADIAL_FISHEYE", num_params: 5 },
    CameraModel { model_id: 10, model_name: "THIN_PRISM_FISHEYE", num_params: 12 }
];


fn to_euler_angles(q: [f32;4]) -> [f32; 3] {
    let mut angles = [0.0 ; 3];

    // roll (rotation autour de l'axe x)
    let sinr_cosp = 2.0 * (q[0] * q[1] + q[2] * q[3]);
    let cosr_cosp = 1.0 - 2.0 * (q[1] *q[1] + q[2] * q[2]);
    angles[0] = sinr_cosp.atan2(cosr_cosp);

    // pitch (rotation autour de l'axe y)
    let sinp = (1.0 + 2.0 * (q[0] * q[2] - q[1] * q[3])).sqrt();
    let cosp = (1.0 - 2.0 * (q[0] * q[2] - q[1] * q[3])).sqrt();
    angles[1] = 2.0 * sinp.atan2(cosp) - std::f32::consts::PI / 2.0;

    // yaw (rotation autour de l'axe z)
    let siny_cosp = 2.0 * (q[0] * q[3] + q[1] * q[2]);
    let cosy_cosp = 1.0 - 2.0 * (q[2] * q[2] +  q[3]* q[3]);
    angles[2] = siny_cosp.atan2(cosy_cosp);

    angles
}

fn read_extrinsics_binary(path_to_model_file: &PathBuf) -> HashMap<u64, Image> {
    let mut fid = File::open(path_to_model_file).expect("Failed to open file");
    let mut buffer = [0; 8];

    fid.read_exact(&mut buffer).expect("Failed to read num_img");
    let num_reg_images: u64 = u64::from_le_bytes(buffer);
    let mut images: HashMap<u64, Image> = HashMap::new();
    println!("num_reg_images {}", num_reg_images);
    for _ in 0..num_reg_images {
        let mut buffers4 = [0;4];
        fid.read_exact(&mut buffers4).expect("Failed to read image_id");
        let image_id: u32 = u32::from_le_bytes(buffers4);
        println!("image_id {}", image_id);

        let mut qvec = [0.0;4];
        for i in 0..4{
            fid.read_exact(&mut buffer).expect("Failed to read qvec");
            qvec[i] = f64::from_le_bytes(buffer);
        }
        println!("qvec {:?}", qvec);
        let mut tvec = [0.0;3];
        for i in 0..3{
            fid.read_exact(&mut buffer).expect("Failed to read tvec");
            tvec[i] = f64::from_le_bytes(buffer);
        }
        println!("tvec {:?}", tvec);

        fid.read_exact(&mut buffers4).expect("Failed to read camera_id");
        let camera_id: u32 = u32::from_le_bytes(buffers4);
        println!("camera_id {}", camera_id);

        let mut buffer1 = [0;1];
        fid.read_exact(&mut buffer1).expect("Failed to read char");
        let mut current_char = u8::from_le_bytes(buffer1);

        while current_char != 0 {  
            fid.read_exact(&mut buffer1).expect("Failed to read char");
            current_char = u8::from_le_bytes(buffer1);

        }

        fid.read_exact(&mut buffer).expect("Failed to read num_points2D");
        let num_points2D : i64 = i64::from_le_bytes(buffer);
        println!("num_points2D {}", num_points2D);
        let mut bufferfin = vec![0;24*(num_points2D as usize)];
        fid.read_exact(&mut bufferfin).expect("Failed to read end");

        images.insert(image_id as u64, Image {
            id: image_id as u64,
            camera_id : camera_id,
            qvec : qvec,
            tvec: tvec,
        });
    }
    images
}

fn read_intrinsics_binary(path_to_model_file: &PathBuf) -> HashMap<u64, Camera> {
    let mut cameras: HashMap<u64, Camera> = HashMap::new();
    let mut fid = File::open(path_to_model_file).expect("Failed to open file");
    let mut buffer = [0; 8];

    fid.read_exact(&mut buffer).expect("Failed to read num_cameras");
    let num_cameras = u64::from_le_bytes(buffer);
    println!("{}",num_cameras);
    for _ in 0..num_cameras {
        let mut buffer2 = [0; 4];
        fid.read_exact(&mut buffer2).expect("Failed to read camera_id");
        println!("{:?}", buffer2);
        let camera_id = u32::from_le_bytes(buffer2);
        println!("camera_id : {}",camera_id);
        fid.read_exact(&mut buffer2).expect("Failed to read model_id");
        let model_id = u32::from_le_bytes(buffer2);
        println!("model_id : {}",model_id);
          

        fid.read_exact(&mut buffer).expect("Failed to read width");
        let width = u64::from_le_bytes(buffer);

        fid.read_exact(&mut buffer).expect("Failed to read height");
        let height = u64::from_le_bytes(buffer);
        println!("width : {}",width);
        println!("height : {}",height);
        let num_params = match CAMERA_MODELS.iter().find(|&model| model.model_id == (model_id as u64)).map(|model| model.num_params) {
            Some(num_params) => num_params,
            None => 0, //println!("Model ID {} not found", model_id),
        };
        let model_name = match CAMERA_MODELS.iter().find(|&model| model.model_id == (model_id as u64)).map(|model| model.model_name) {
            Some(model_name) => model_name,
            None => "",
        };

        println!("num_params : {}",num_params);
        let mut params = vec![0.0; num_params];

        for i in 0..num_params {
            let mut param_buffer = [0; 8];
            fid.read_exact(&mut param_buffer).expect("Failed to read parameter");
            params[i] = f64::from_le_bytes(param_buffer);
        }
        println!("model_name : {}",model_name);
        println!("params {:?}", params);
        cameras.insert(camera_id as u64, Camera {
            id: camera_id as u64,
            model: model_name.to_string(),
            width: width,
            height: height,
            params: params,
        });
    }

    cameras
}

fn read_colmap_scene_info(path: &str) -> HashMap<u64, Camera> {
    let cameras_intrinsic_file = Path::new(&path).join("sparse/0").join("cameras.bin");
    let cameras_extrinsic_file = Path::new(&path).join("sparse/0").join("images.bin");
    let cam_intrinsics:HashMap<u64, Camera>;
    cam_intrinsics = read_intrinsics_binary(&cameras_intrinsic_file);
    let cameras_extrinsic = read_extrinsics_binary(&cameras_extrinsic_file);
    let cameras: HashMap<u64, Camera> = HashMap::new();
    cameras
}

fn main() {
    let test = to_euler_angles([0.9453782676338611, 0.32410944923190543, -0.01669260238097348, 0.030567188780477653]);
    println!("{:?}",test);
    //read_colmap_scene_info("C:\\3DGS\\gaussian-splatting\\tandt_db\\lefaucheux_7mm");
}