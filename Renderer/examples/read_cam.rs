
use std::path::{Path, PathBuf};
use std::fs::File;
use std::io::Read;
use std::collections::HashMap;
use geometric_algebra::ppga3d::Point;
use geometric_algebra::Zero;
use nalgebra::{Matrix3, Matrix4, Vector3};

#[derive(Debug, Clone)]
struct CameraModel<'a> {
    model_id: u64,
    model_name: &'a str,
    num_params: usize,
}

#[derive(Debug, Clone)]
pub struct Camera {
    id: u64,
    model: String,
    pub width: u64,
    pub height: u64,
    params: Vec<f64>,
}

pub struct Image {
    id: u64,
    camera_id: u32,
    qvec: [f64;4],
    tvec: [f64;3],
}

// Définir CAMERA_MODELS si nécessaire
const CAMERA_MODELS: [CameraModel; 2] = [
    CameraModel { model_id: 0, model_name: "SIMPLE_PINHOLE", num_params: 3 },
    CameraModel { model_id: 1, model_name: "PINHOLE", num_params: 4 },
];

fn qvec2rotmat(qvec: &[f64; 4]) -> Matrix3<f64> {
    Matrix3::new(   
            1.0 - 2.0 * qvec[2].powi(2) - 2.0 * qvec[3].powi(2), 2.0 * qvec[1] * qvec[2] - 2.0 * qvec[0] * qvec[3], 2.0 * qvec[3] * qvec[1] + 2.0 * qvec[0] * qvec[2],
            2.0 * qvec[1] * qvec[2] + 2.0 * qvec[0] * qvec[3], 1.0 - 2.0 * qvec[1].powi(2) - 2.0 * qvec[3].powi(2), 2.0 * qvec[2] * qvec[3] - 2.0 * qvec[0] * qvec[1],
            2.0 * qvec[3] * qvec[1] - 2.0 * qvec[0] * qvec[2], 2.0 * qvec[2] * qvec[3] + 2.0 * qvec[0] * qvec[1], 1.0 - 2.0 * qvec[1].powi(2) - 2.0 * qvec[2].powi(2)
    )
}

fn focal2fov(focal: f64, pixels: f64) -> f64 {
    2.0 * (pixels / (2.0 * focal)).atan()
}

fn read_extrinsics_binary(path_to_model_file: &PathBuf) -> HashMap<u64, Image> {
    let mut fid = File::open(path_to_model_file).expect("Failed to open file");
    let mut buffer = [0; 8];

    fid.read_exact(&mut buffer).expect("Failed to read num_img");
    let num_reg_images: u64 = u64::from_le_bytes(buffer);
    let mut images: HashMap<u64, Image> = HashMap::new();
    for _ in 0..num_reg_images {
        let mut buffers4 = [0;4];
        fid.read_exact(&mut buffers4).expect("Failed to read image_id");
        let image_id: u32 = u32::from_le_bytes(buffers4);

        let mut qvec = [0.0;4];
        for i in 0..4{
            fid.read_exact(&mut buffer).expect("Failed to read qvec");
            qvec[i] = f64::from_le_bytes(buffer);
        }
        let mut tvec = [0.0;3];
        for i in 0..3{
            fid.read_exact(&mut buffer).expect("Failed to read tvec");
            tvec[i] = f64::from_le_bytes(buffer);
        }

        fid.read_exact(&mut buffers4).expect("Failed to read camera_id");
        let camera_id: u32 = u32::from_le_bytes(buffers4);

        let mut buffer1 = [0;1];
        fid.read_exact(&mut buffer1).expect("Failed to read char");
        let mut current_char = u8::from_le_bytes(buffer1);

        while current_char != 0 {  
            fid.read_exact(&mut buffer1).expect("Failed to read char");
            current_char = u8::from_le_bytes(buffer1);

        }

        fid.read_exact(&mut buffer).expect("Failed to read num_points2D");
        let num_points2_d : i64 = i64::from_le_bytes(buffer);
        let mut bufferfin = vec![0;24*(num_points2_d as usize)];
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
    for _ in 0..num_cameras {
        let mut buffer2 = [0; 4];
        fid.read_exact(&mut buffer2).expect("Failed to read camera_id");
        let camera_id = u32::from_le_bytes(buffer2);
        fid.read_exact(&mut buffer2).expect("Failed to read model_id");
        let model_id = u32::from_le_bytes(buffer2);
        fid.read_exact(&mut buffer).expect("Failed to read width");
        let width = u64::from_le_bytes(buffer);
        fid.read_exact(&mut buffer).expect("Failed to read height");
        let height = u64::from_le_bytes(buffer);
        let num_params = match CAMERA_MODELS.iter().find(|&model| model.model_id == (model_id as u64)).map(|model| model.num_params) {
            Some(num_params) => num_params,
            None => 0, //println!("Model ID {} not found", model_id),
        };
        let model_name = match CAMERA_MODELS.iter().find(|&model| model.model_id == (model_id as u64)).map(|model| model.model_name) {
            Some(model_name) => model_name,
            None => "",
        };
        let mut params = vec![0.0; num_params];

        for i in 0..num_params {
            let mut param_buffer = [0; 8];
            fid.read_exact(&mut param_buffer).expect("Failed to read parameter");
            params[i] = f64::from_le_bytes(param_buffer);
        }
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

pub(crate) fn read_colmap_scene_info(path: &str) -> (HashMap<u64, Camera>, HashMap<u64, Image>) {
    let cameras_intrinsic_file = Path::new(&path).join("sparse/0").join("cameras.bin");
    let cameras_extrinsic_file = Path::new(&path).join("sparse/0").join("images.bin");
    let cameras_intrinsics = read_intrinsics_binary(&cameras_intrinsic_file);
    let cameras_extrinsic = read_extrinsics_binary(&cameras_extrinsic_file);
    (cameras_intrinsics,cameras_extrinsic)
}

pub(crate) fn compute_matrix(cameras_intrinsics: &HashMap<u64, Camera>,cameras_extrinsic : &HashMap<u64, Image> ,id_image: &u64 ) -> ([Point; 4], [Point; 4], [Point; 4], f64, f64) {
    let image = cameras_extrinsic.get(id_image).unwrap();
    let camera = cameras_intrinsics.get(&(image.camera_id as u64)).unwrap();
    
    let r = qvec2rotmat(&image.qvec).transpose();
    let t = Vector3::new(image.tvec[0],image.tvec[1],image.tvec[2]);

    let focal_length_x = camera.params[0];
    let focal_length_y = camera.params[1];
    let fo_vy = focal2fov(focal_length_y, camera.height as f64);
    let fo_vx = focal2fov(focal_length_x, camera.width as f64);
    let znear = 0.001;
    let zfar = 100.00;

    let world_view_transform = get_world_to_view2(&r, &t);
    let world_view_transform_p = matrix_vers_points(world_view_transform);
    let projection_matrix = get_projection_matrix(znear, zfar, fo_vx, fo_vy);
    let projection_matrix_p = matrix_vers_points(projection_matrix);
    let camera_center = world_view_transform.try_inverse().expect("erreur d'inversion de world_view_transform");
    let camera_center_p = matrix_vers_points(camera_center);
    (world_view_transform_p,projection_matrix_p,camera_center_p,fo_vy,fo_vx)

}

fn matrix_vers_points(matrix: Matrix4<f32>) -> [Point; 4] {
    let mut points: [Point; 4] = [Point::zero();4];

    for i in 0..4 {
        let x = matrix[(0, i)];
        let y = matrix[(1, i)];
        let z = matrix[(2, i)];
        let w = matrix[(3, i)];

        points[i] = Point::new( x, y, z, w );
    }

    points
}

fn get_world_to_view2(R: &Matrix3<f64>, t: &Vector3<f64>) -> Matrix4<f32> {
    let mut Rt = Matrix4::zeros();
    Rt.fixed_slice_mut::<3, 3>(0, 0).copy_from(&R.transpose());
    Rt.fixed_slice_mut::<3, 1>(0, 3).copy_from(t);
    Rt[(3, 3)] = 1.0;
    Rt.cast::<f32>()
}


fn get_projection_matrix(znear: f64, zfar: f64, fov_x: f64, fov_y: f64) -> Matrix4<f32> {
    let tan_half_fov_y = (fov_y / 2.0).tan();
    let tan_half_fov_x = (fov_x / 2.0).tan();

    let top = tan_half_fov_y * znear;
    let bottom = -top;
    let right = tan_half_fov_x * znear;
    let left = -right;

    let mut p = Matrix4::zeros();

    let z_sign = 1.0;

    p[(0, 0)] = 2.0 * znear / (right - left);
    p[(1, 1)] = 2.0 * znear / (top - bottom);
    p[(0, 2)] = (right + left) / (right - left);
    p[(1, 2)] = (top + bottom) / (top - bottom);
    p[(3, 2)] = z_sign;
    p[(2, 2)] = z_sign * zfar / (zfar - znear);
    p[(2, 3)] = -(zfar * znear) / (zfar - znear);

    p.cast::<f32>()
}

fn main() {}