import copy
from dataclasses import dataclass
import json

from matplotlib import pyplot as plt
import os
import numpy as np
from naplab.camera import Camera, parse_camera_json
from naplab.frame_data import better_process_data


@dataclass
class ImageData:
    image_index: int
    image_path: str
    transform: np.ndarray

class ImagesWithTransforms():
    def __init__(self, camera: Camera, gps_left: str, gps_right: str, n:int, stride=1, output_dir="images"):
        self.camera = camera
        self.images_with_transforms: list[ImageData] = []
        timestamps = camera.timestamps
        self.frames = better_process_data(gps_left, gps_right, timestamps).take(n, stride=stride)
        self.output_dir = output_dir.split("/")[-1]
        for frame in self.frames:
            transform = camera.get_transform_matrix(frame, as_blender=True).tolist()
            image_index = timestamps.index(frame.timestamp)
            image_path = f"{self.output_dir}/{self.camera.id}_{image_index}.png"
            self.images_with_transforms.append(ImageData(image_index, image_path, transform))
        
        

class NaplabDataset():
    def __init__(self, gps_left: str, gps_right: str, fps: int, rig_json_path: str = "./Trip094/camerasandCanandGnssCalibratedAll_lidars00-virtual.json", skip_image_creation = False) -> None:
        stride = np.max([30//fps, 2])
        self.cameras = parse_camera_json(rig_json_path)
        self.all_images_with_transforms = [ImagesWithTransforms(camera, gps_left, gps_right, n=15, stride=stride) for camera in self.cameras]
        self.skip_image_creation = skip_image_creation
        
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
                ax.plot(z, x, y, marker='.', markersize=0.5, color=c, label=labels[i])
                if show_scatter:
                    ax.scatter(z, x, y, color=c, marker='o')
            else:
                plt.plot(z, x, marker='.', markersize=0.5, color=c, label=labels[i])
                if show_scatter:
                    plt.scatter(z, x, marker='.', color=c, s=100)
        
        if is3D:
            ax.set_xlabel('Z')
            ax.set_ylabel('X')
            ax.set_zlabel('Y')
            ax.set_title('--------------------------------3D plot of ZXY coordinates--------------------------------')
        else:
            plt.xlabel('Z')
            plt.ylabel('X')
            plt.title('--------------------------------2D plot of ZX coordinates--------------------------------')
            
        plt.legend()
        plt.tight_layout()
        plt.show()