import math
import os
import subprocess
import numpy as np

from .frame_data import FrameData
from .utils import make_homogenous, normalize
import json
from scipy.optimize import curve_fit
import re

class Camera():
    def __init__(self, name: str, cx: float, cy: float, height: int, width: int, translation: tuple, roll_pitch_yaw: tuple, fov=60, bw_poly = np.array([0, 0, 0, 0]), video_path = "", timestamps_path = ""):
        self.name = name
        self.cx = cx
        self.cy = cy
        self.fx = (width * .5) / np.tan(np.radians(fov) / 2)
        self.fy = (height * .5) / np.tan(np.radians(fov) / 2)
        self.height = height
        self.width = width
        self.translation = make_homogenous(np.array(translation))
        self.bw_poly = bw_poly
        #self.translation = self.translation @ make_homogenous(np.array([[0,0,1],[1,0,0],[0,1,0]]))
        self.timestamps = self._read_timestamps(timestamps_path) if timestamps_path else []
        self.video_path = video_path
        self.roll_pitch_yaw = roll_pitch_yaw
        self.description = name
    
    def set_description(self, description: str):
        self.description = description
    
    def __repr__(self) -> str:
        return f"Camera({self.decription})"
    
    def get_camera_intrinsics(self):
        k1, k2, k3, k4 = self.calculate_distortion_coeff()
        return {
            "camera_model": "OPENCV",
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
        # Define the backward polynomial b(r)
        backward_polynomial = lambda r, j1, j2, j3, j4: j1*r + j2*r**2 + j3*r**3 + j4*r**4
        
        # Coefficients from the backward polynomial
        j1, j2, j3, j4 = self.bw_poly

        # Generate radial distance values (r) dynamically from center to various points in the image
        x_values = np.linspace(0, self.width - 1, num=int(self.width/2))  # Sample across width
        y_values = np.linspace(0, self.height - 1, num=int(self.height/2))  # Sample across height

        # Compute radial distances from center to all points
        x_grid, y_grid = np.meshgrid(x_values, y_values)
        r_values = np.sqrt((x_grid - self.cx)**2 + (y_grid - self.cy)**2).flatten()
        theta_values = backward_polynomial(r_values, j1, j2, j3, j4)

        # Fit a polynomial to the theta as function of r, seeking the inverse relationship
        # We define the forward polynomial we are trying to fit
        forward_polynomial = lambda theta, k1, k2, k3, k4: k1*theta + k2*theta**2 + k3*theta**3 + k4*theta**4

        # Use curve fitting to find the best fit k coefficients
        k_coeffs, _ = curve_fit(forward_polynomial, theta_values, r_values)

        return k_coeffs # k1, k2, k3, k4
        

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
    
    def get_translation_matrix(self):
        translation_matrix = np.identity(4)
        translation_matrix[:, 3] = self.translation
        return translation_matrix
    
    
    def get_transform_matrix(self, car_translation_matrix: np.ndarray, car_rotation_matrix: np.ndarray):
        """Get the translation matrix from the given position"""
        rotation_matrix = self.get_rotation_matrix()

        translation_matrix = self.get_translation_matrix()
        
        camera_local_transform = rotation_matrix @ translation_matrix
        
        transform_matrix = car_translation_matrix @ car_rotation_matrix @ camera_local_transform

        return transform_matrix
    
    
    def get_camera_position(self, data: FrameData):
        """Get the camera position given initial position (x, y, z)"""
        # :)
        return data.center + data.get_rotation_matrix() @ self.translation
        return np.linalg.inv(self.get_rotation_matrix()) @ self.get_translation_matrix() @ self.get_rotation_matrix() @ data.center
    
    
    def get_camera_direction_vector(self, data: FrameData):
        """Get the camera direction given rotation matrix"""
        direction = data.get_rotation_matrix() @ self.get_rotation_matrix() @ np.array([1, 0, 0, 1])
        return normalize(direction)
    
    
    def save_frames(self, frame_indexes, output_dir='frames_output'):
        try:
            os.makedirs(output_dir, exist_ok=True)
            for index in frame_indexes:
                output_path = os.path.join(output_dir, f"frame_{index}.png")
                subprocess.run(['ffmpeg', '-i', self.video_path, '-vf', f"select='eq(n\,{index})'", '-vsync', 'vfr', output_path], capture_output=True, text=True)
        except Exception as e:
            print("Error:", e)
    
    
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
    for camera in sensors:
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
                                 fov=fov, bw_poly=bw_poly, video_path=f"{folder_path}/{video_path}", timestamps_path=f"{folder_path}/{timestamps_path}"))
    return cameraList


def filter_cameras(cameraList: list[Camera], camera_filter):
    out = [cam for cam in cameraList if cam.name in camera_filter]
    if len(out) == 0:
        raise Exception("No cameras found with the given filter")
    elif len(out) != len(camera_filter):
        raise Exception("Some cameras were not found")
    return out
