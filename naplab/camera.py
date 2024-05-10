from concurrent.futures import ThreadPoolExecutor
import os
import subprocess
import numpy as np
from scipy.spatial.transform import Rotation as R
from .frame_data import FrameData
from .utils import make_homogenous, normalize
import json
from scipy.optimize import curve_fit
import re


class Camera():
    def __init__(self, name: str, cx: float, cy: float, height: int, width: int, translation: tuple, roll_pitch_yaw: tuple, fov=60, bw_poly = np.array([0, 0, 0, 0]), video_path = "", timestamps_path = "", id=0):
        self.name = name
        self.cx = cx
        self.cy = cy
        self.fx = (width * .5) / np.tan(np.radians(fov) / 2)
        self.fy = (height * .5) / np.tan(np.radians(fov) / 2)
        self.fov = fov
        self.height = height
        self.width = width
        self.translation = make_homogenous(np.array(translation))
        self.bw_poly = bw_poly
        #self.translation = self.translation @ make_homogenous(np.array([[0,0,1],[1,0,0],[0,1,0]]))
        self.timestamps = self._read_timestamps(timestamps_path) if timestamps_path else []
        self.video_path = video_path
        self.roll_pitch_yaw = roll_pitch_yaw
        self.description = name
        self.coefficients = None
        self.image_names = None
        self.id = id
    
    def set_description(self, description: str):
        self.description = description
    
    def __repr__(self) -> str:
        return f"Camera({self.decription})"
    
    def get_camera_intrinsics(self):
        k1, k2, k3, k4 = self.calculate_distortion_coeff()
        return {
            "camera_model": "OPENCV_FISHEYE",
            "fl_x": self.fx,
            "fl_y": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "w": self.width,
            "h": self.height,
            "k1": k1,
            "k2": k2,
            "k3": k3,
            "k4": k4,
            "p1": 0,
            "p2": 0,
        }
    
    
    def calculate_distortion_coeff(self) -> np.ndarray[float]:
        # Assumes f-theta camera model
        # Define the backward polynomial b(r)
        if self.coefficients is not None:
            return self.coefficients
        backward_polynomial = lambda r, j1, j2, j3, j4: j1*r + j2*r**2 + j3*r**3 + j4*r**4
        
        # Coefficients from the backward polynomial
        j1, j2, j3, j4 = self.bw_poly

        # Generate radial distance values (r) dynamically from center to various points in the image
        square_length = self.width if self.width < self.height else self.height
        x_values = np.linspace(0, self.width - 1, num=int(self.width))  # Sample across width
        y_values = np.linspace(0, self.height - 1, num=int(self.height))  # Sample across height
        x_values = np.linspace(0,square_length - 1, num=int(square_length))  # Sample across width
        y_values = np.linspace(0, square_length - 1, num=int(square_length))  # Sample across height

        # Compute radial distances from center to all points
        x_grid, y_grid = np.meshgrid(x_values, y_values)
        r_values = np.sqrt((x_grid - self.cx)**2 + (y_grid - self.cy)**2).flatten()
        theta_values = backward_polynomial(r_values, j1, j2, j3, j4)

        # Fit a polynomial to the theta as function of r, seeking the inverse relationship
        # We define the forward polynomial we are trying to fit
        # Finds kannala brandt coefficients
        forward_polynomial = lambda theta, k1, k2, k3, k4: self.fy * (k1*theta + k2*theta**3 + k3*theta**5 + k4*theta**7)

        # Use curve fitting to find the best fit k coefficients
        k_coeffs, _ = curve_fit(forward_polynomial, theta_values, r_values)
        self.coefficients = k_coeffs
        return k_coeffs # k1, k2, k3, k4
        
    def get_colmap_camera_description(self):
        k1, k2, k3, k4 = self.calculate_distortion_coeff()
        return f"{self.id} OPENCV_FISHEYE {self.width} {self.height} {self.fx} {self.fy} {self.cx} {self.cy} {k1} {k2} {k3} {k4}"

    def get_rotation_matrix(self):
        """Create a rotation matrix from roll, pitch, and yaw."""
        # Convert angles from degrees to radians
        roll, pitch, yaw = self.roll_pitch_yaw
        roll = np.radians(roll)
        pitch = np.radians(pitch)
        yaw = np.radians(yaw)

        # Create rotation matrices for each axis
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])
        
        Ry = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        
        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])

        # Combine the rotation matrices
        R = Rz @ Ry @ Rx
        return make_homogenous(R)

    def get_quaternion(self, frame: FrameData):
        rotation = frame.get_rotation_matrix() @ self.get_rotation_matrix()
        return R.from_matrix(rotation[:3, :3]).as_quat(True)
    
    def get_translation_matrix(self):
        translation_matrix = np.identity(4)
        translation_matrix[:, 3] = self.translation
        return translation_matrix
    
    def get_translation_vector(self, frame: FrameData):
        return self.get_camera_position(frame)
    
    def get_colmap_image_repr(self, image_id: int, image_name: str, frame: FrameData):
        """
        Creates a string that corresponds to one entry in the images.txt that colmap uses
        """
        quat = self.get_quaternion(frame)
        t = self.get_translation_vector(frame)
        return f"{image_id} {quat[0]} {quat[1]} {quat[2]} {quat[3]} {t[0]} {t[1]} {t[2]} {self.id} {image_name}"
    
    def get_colmap_image_txt(self, offset: int, frames: list[FrameData]):
        all_images_string = ""
        for i, frame in enumerate(frames):
            frame_index = self.timestamps.index(frame.timestamp)
            image_name = f"cam_{self.id}_frame_{frame_index}.png"
            image_index = i + offset
            all_images_string += self.get_colmap_image_repr(image_index, image_name, frame) + "\n\n"
        return (all_images_string, len(frames))

    def utm_to_blender_rotation(self):
        """Rotation matrix to rotate UTM33N coordinates into Blender's camera-facing setup"""
        return np.eye(4)
        return np.array([
            [ 1, 0,  0, 0],  # North goes forward (X)
            [ 0, 0,  1, 0],  # Altitude goes up (Y)
            [ 0, 1,  0, 0], # East goes to the right (Z)
            [ 0, 0,  0, 1]   # Homogeneous coordinate
        ])
    
    def get_transform_matrix(self, car_translation_matrix: np.ndarray, car_rotation_matrix: np.ndarray, as_blender=False):
        """Get the translation matrix from the given position"""
        rotation_matrix = self.get_rotation_matrix()

        translation_matrix = self.get_translation_matrix()
        camera_local_transform = rotation_matrix @ translation_matrix
        
        transform_matrix = car_translation_matrix @ car_rotation_matrix @ camera_local_transform
        if as_blender:
            return self.utm_to_blender_rotation() @ transform_matrix
        return transform_matrix
    
    
    def get_camera_position(self, data: FrameData):
        """Get the camera position given initial position (x, y, z)"""
        # :)
        return data.center + data.get_rotation_matrix() @ self.translation
    
    
    def get_camera_direction_vector(self, data: FrameData):
        """Get the camera direction given rotation matrix"""
        direction = data.get_rotation_matrix() @ self.get_rotation_matrix() @ np.array([1, 0, 0, 1])
        return normalize(direction)
    
    
    def save_process_batch(self, batch_indexes, output_dir, batch_number) -> str:
        filter_string = '+'.join([f"eq(n,{i})" for i in batch_indexes])
        output_path_template = f"{output_dir}/cam_{self.id}_batch_{batch_number}_frame_%d.png"
        command = [
            'ffmpeg',
            '-i', self.video_path,
            '-vf', f"select='{filter_string}'",
            '-vsync', 'vfr',
            '-frames:v', str(len(batch_indexes)),
            output_path_template
        ]
        subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Rename files immediately after creation
        for i, frame_index in enumerate(batch_indexes):
            original_path = f"{output_dir}/cam_{self.id}_batch_{batch_number}_frame_{i+1}.png"
            new_path = f"{output_dir}/{frame_index}.png"
            os.rename(original_path, new_path)
        return f"Cam {self.id} | Image Saving Batch {batch_number} completed"
    

    def save_frames(self, frame_indexes, output_dir='frames_output', batch_size=1000):
        os.makedirs(output_dir, exist_ok=True)
        batches = [frame_indexes[i:i + batch_size] for i in range(0, len(frame_indexes), batch_size)]
        
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(lambda batch, i: self.save_process_batch(batch, output_dir, i),
                                        batches, range(len(batches))))
            for result in results:
                print(result)

    
    
    def _read_timestamps(self, file_path_to_timestamps: str):
        with open(file_path_to_timestamps, 'r') as f:
            lines = f.readlines()
        return [int(line.split()[1]) for line in lines]



def parse_camera_json(filepath: str) -> list[Camera]:
    with open(filepath, "r") as f:
        folder_path = os.path.dirname(filepath)
        data = json.load(f)
        sensors = data["rig"]["sensors"]
        sensors = [sensor for sensor in sensors if sensor["protocol"] == "camera.virtual"]
    
    cameraList = []
    for i, camera in enumerate(sensors):
        props = camera["properties"]
        sensorProps = camera["nominalSensor2Rig_FLU"]
        parameter = camera["parameter"]
        
        if props["Model"] != "ftheta":
            raise Exception("Only ftheta cameras are supported")
        raw_bw_poly: str = props["bw-poly"]
        bw_poly = np.array([float(x) for x in raw_bw_poly.strip().split(" ")[1:]])
        fov = 120 if "120" in camera["name"] else 60
        video_path, timestamps_path = re.search(r"video=(.*),timestamp=(.*)", parameter).groups()
        cameraList.append(Camera(camera["name"], float(props["cx"]), float(props["cy"]), int(props["height"]), int(props["width"]), sensorProps["t"], sensorProps["roll-pitch-yaw"], 
                                 fov=fov, bw_poly=bw_poly, video_path=f"{folder_path}/{video_path}", timestamps_path=f"{folder_path}/{timestamps_path}", id=i))
    return cameraList


def filter_cameras(cameraList: list[Camera], camera_filter):
    out = [cam for cam in cameraList if cam.name in camera_filter]
    if len(out) == 0:
        raise Exception("No cameras found with the given filter")
    elif len(out) != len(camera_filter):
        raise Exception("Some cameras were not found")
    return out
