import copy
from dataclasses import dataclass
import json
import glob
from pathlib import Path
from typing import List
from PIL import Image 

from matplotlib import pyplot as plt
import os
import numpy as np
from naplab.camera import Camera, parse_camera_json
from naplab.frame_data import better_process_data
import pycolmap
from scipy.spatial.transform import Rotation as R
from pycolmap import Reconstruction


@dataclass
class ImageData:
    image_index: int
    image_path: str
    transform: np.ndarray

class ImagesWithTransforms():
    def __init__(self, camera: Camera, gps_left: str, gps_right: str, n = -1, stride = 30, output_dir="images"):
        self.camera = camera
        self.images_with_transforms: List[ImageData] = []
        timestamps = camera.timestamps
        self.frames = better_process_data(gps_left, gps_right, timestamps).take(n, stride=stride)
        self.output_dir = output_dir.split("/")[-1]
        for frame in self.frames:
            transform = camera.get_blender_transform_matrix(frame).tolist()
            image_index = timestamps.index(frame.timestamp)
            image_path = f"{self.output_dir}/{self.camera.id}_{image_index}.png"
            self.images_with_transforms.append(ImageData(image_index, image_path, transform))

    def recompute_images(self):
        self.images_with_transforms: List[ImageData] = []
        for frame in self.frames:
            transform = self.camera.get_blender_transform_matrix(frame).tolist()
            image_index = self.camera.timestamps.index(frame.timestamp)
            image_path = f"{self.output_dir}/{self.camera.id}_{image_index}.png"
            self.images_with_transforms.append(ImageData(image_index, image_path, transform))
        
        
        

class NaplabDataset():
    def __init__(self, gps_left: str, gps_right: str, fps: int, n = 15, rig_json_path: str = "./Trip094/camerasandCanandGnssCalibratedAll_lidars00-virtual.json", skip_image_creation = False, center=True) -> None:
        if fps != -1:
            stride = np.max([30//fps, 1])
        else:
            stride = 30
        self.cameras = parse_camera_json(rig_json_path)
        self.all_images_with_transforms = [ImagesWithTransforms(camera, gps_left, gps_right, n=n, stride=stride) for camera in self.cameras]
        self.skip_image_creation = skip_image_creation
        self.global_translate = np.array((0, 0, 0))
        if center:
            all_frames = []
            all_transforms = []
            for images_with_transform in self.all_images_with_transforms:
                for frame in images_with_transform.frames:
                    all_frames.append(frame)
                    all_transforms.append(frame.center)
            all_transforms = np.array(all_transforms)
            x_min, y_min, z_min, _ = all_transforms.min(axis=0)
            min_vec = np.array([x_min, y_min, z_min, 0])
            for frame in all_frames:
                frame.center -= min_vec
            for images_with_transform in self.all_images_with_transforms:
                images_with_transform.recompute_images()
            self.global_translate = min_vec[:3]
            
        print("translation factor: ", self.global_translate)

    def create_colmap_dataset(self, out_dir: str):
        """
        create images.txt
        cameras.txt
        empty 3dpoint.txt
        """
        os.makedirs(out_dir, exist_ok=True)
        total_offset = 0
        full_str = ""
        full_desc = ""
        for it in self.all_images_with_transforms:
            cam = it.camera
            next_str, offset = cam.get_colmap_image_txt(total_offset, it.frames)
            total_offset += offset
            full_str += next_str
            indices = [i.image_index for i in it.images_with_transforms]
            if not self.skip_image_creation:
                cam.save_frames(indices, f"{out_dir}/images")
            full_desc += cam.get_colmap_camera_description() + "\n"
            
        with open(f"{out_dir}/cameras.txt", "w") as f:
            f.write(full_desc)
        with open(f"{out_dir}/images.txt", "w") as f:
            f.write(full_str)
        print("Images.txt created")

        with open(f"{out_dir}/points3D.txt", "w") as f:
            pass
        
        
        
    def create_nerfstudio_dataset(self, out_dir: str):
        camera_model = self.all_images_with_transforms[0].camera.get_camera_intrinsics()["camera_model"]
        json_data = {"camera_model": camera_model}
        frames = []
        os.makedirs(out_dir, exist_ok=True)
        for images_transforms in self.all_images_with_transforms:
            intrinsics = images_transforms.camera.get_camera_intrinsics()
            intrinsics.pop("camera_model")
            for it in images_transforms.images_with_transforms:
                frame = copy.copy(intrinsics)
                frame["file_path"] = it.image_path
                frame["transform_matrix"] = it.transform
                frames.append(frame)
            if not self.skip_image_creation:
                indices = [i.image_index for i in images_transforms.images_with_transforms]
                images_transforms.camera.save_frames(indices, f"{out_dir}/images")
        if not self.skip_image_creation:
            image_paths = list(map(lambda file: Path(file), glob.glob(f"{out_dir}/images/*.png")))
            for i in [2,4,8]:
                os.makedirs(f"{out_dir}/images_{i}", exist_ok=True)
                for image_path in image_paths:
                    image = Image.open(image_path)
                    downscaled = image.resize((image.width // i, image.height // i))
                    downscaled.save(f"{out_dir}/images_{i}/{image_path.name}")

        json_data["frames"] = frames
        
        with open(f"{out_dir}/transforms.json", "w") as f:
            json.dump(json_data, f, indent=4)
        print(f"Transforms JSON created at {out_dir}/transforms.json")
        
        
    def plot_blender_coordinates(self, figsize=(5, 5), is3D=False, show_scatter=True):
        """Plot 2D or 3D coordinates."""
        if is3D:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection='3d')
        else:
            plt.figure(figsize=figsize)
        colorpallet = ['b', 'g', 'r', 'c', 'm', 'y', 'k', '#c1c3e3', 'b', 'g', 'r', 'c', 'm', 'y', 'k', '#c1c3e3']
        labels = [it.camera for it in self.all_images_with_transforms]
        for i, iwt in enumerate(self.all_images_with_transforms):
            c = colorpallet.pop(0)
            image_positions = [np.array(iwt.transform)[:, 3] for iwt in iwt.images_with_transforms]
            x = [data[0] for data in image_positions]
            z = [data[2] for data in image_positions]
            if is3D:
                y = [data[1] for data in image_positions]
                ax.plot(z, x, y, marker='.', markersize=0.5, color=c, label=labels[i]) # type: ignore
                if show_scatter:
                    ax.scatter(z, x, y, color=c, marker='o') # type: ignore
            else:
                plt.plot(z, x, marker='.', markersize=0.5, color=c, label=labels[i])
                if show_scatter:
                    plt.scatter(z, x, marker='.', color=c, s=100) # type: ignore
        
        if is3D:
            ax.set_xlabel('Z') # type: ignore
            ax.set_ylabel('X') # type: ignore
            ax.set_zlabel('Y') # type: ignore
            ax.set_title('--------------------------------3D plot of ZXY coordinates--------------------------------') # type: ignore
        else:
            plt.xlabel('Z')
            plt.ylabel('X')
            plt.title('--------------------------------2D plot of ZX coordinates--------------------------------')
            
        plt.legend()
        plt.tight_layout()
        plt.show()


from distutils.dir_util import copy_tree

def create_transform_json(reconstruction: pycolmap.Reconstruction, image_source: Path, out_dir: Path):
    frames = []
    cams = reconstruction.cameras
    images = reconstruction.images
    out_dir.mkdir(exist_ok=True)
    images_out_dir = out_dir / "images"
    images_out_dir.mkdir(exist_ok=True)
    image_destination = out_dir / "images"

    if images_out_dir.as_posix() == image_source.as_posix():
        print("image path and destination equal, skipping copy step")
    else:
        print("Copying images into", images_out_dir)
        copy_tree(image_source.as_posix(), images_out_dir.as_posix())

    for im_id, im_data in images.items():
        name: str = im_data.name
        im_path: Path = image_destination / name
        cam = cams[im_data.camera_id]
        rotation = im_data.cam_from_world.matrix()[:3, :3]
        r = R.from_matrix(rotation)
        t = np.array(im_data.cam_from_world.matrix())[:3, 3]
        matrix = np.eye(4)
        opencv_to_blender_matrix = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]) # OPENCV TO BLENDER CAMERA
        matrix[:3, :3] = r.as_matrix().T @ opencv_to_blender_matrix # r^T @ base_change_matrix
        matrix[:3, 3] = -r.as_matrix().T @ t # -r^T @ t

        cam_p = cam.params
        frame = {
            "fl_x": cam_p[0],
            "fl_y": cam_p[1],
            "cx": cam_p[2],
            "cy": cam_p[3],
            "w": cam.width,
            "h": cam.height,
            "k1": cam_p[4],
            "k2": cam_p[5],
            "p1": cam_p[6], 
            "p2": cam_p[7],
            "file_path": (Path("images") / name).as_posix(),
            "transform_matrix": matrix.tolist()
        }
        frames.append(frame)

    transforms_content = {
        "camera_model": "OPENCV",
        "frames": frames
    }

    with open(f"{out_dir}/transforms.json", "w") as f:
        json.dump(transforms_content, f, indent=4)
    print(f"Transforms JSON created at {out_dir}/transforms.json")