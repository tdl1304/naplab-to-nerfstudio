import numpy as np


class Camera():
    def __init__(self, name: str, cx: float, cy: float, height: int, width: int, t: tuple, roll_pitch_yaw: tuple, fov=60):
        self.name = name
        self.cx = cx
        self.cy = cy
        self.fx = (width * .5) / np.tan(np.radians(fov) / 2)
        self.fy = (height * .5) / np.tan(np.radians(fov) / 2)
        self.height = height
        self.width = width
        self.t = t
        self.roll_pitch_yaw = roll_pitch_yaw
    
    def get_camera_intrinsics(self):
        return {
            "camera_model": "OPENCV",
            "fl_x": self.fx,
            "fl_y": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "w": self.width,
            "h": self.height,
            "k1": 0,
            "k2": 0,
            "p1": 0,
            "p2": 0,
        }
    
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
        rotation_matrix = np.zeros((4, 4))
        rotation_matrix[3, 3] = 1
        rotation_matrix[:3, :3] = R
        return rotation_matrix
    
    def get_translation_matrix(self):
        translation_matrix = np.zeros((4, 4))
        translation_matrix[3, 3] = 1
        translation_matrix[:3, 3] = self.t
        return translation_matrix
    
    
    def get_transform_matrix(self, car_translation_matrix: np.ndarray, car_rotation_matrix: np.ndarray):
        """Get the translation matrix from the given position"""
        rotation_matrix = self.get_rotation_matrix()

        translation_matrix = self.get_translation_matrix()
        
        camera_local_transform = rotation_matrix @ translation_matrix
        
        transform_matrix = car_translation_matrix @ car_rotation_matrix @ camera_local_transform

        return transform_matrix
    
    
    def get_camera_position(self, pos: np.array, rotation_matrix: np.array):
        """Get the camera position given initial position (x, y, z)"""
        return rotation_matrix @ self.t + pos
    
    def get_camera_direction_vector(self, car_rotation_matrix: np.array):
        """Get the camera direction given rotation matrix"""
        return car_rotation_matrix @ self.get_rotation_matrix()[:3,:3] @ np.array([1, 0, 0])
    