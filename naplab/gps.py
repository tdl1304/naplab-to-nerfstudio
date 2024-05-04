from dataclasses import dataclass
import numpy as np
import pyproj
import re

transformer = pyproj.Transformer.from_crs("4326", "32633", always_xy=True) # GPS to UTM33

def convert_to_utm(lat, lon, alt):
    """Convert latitude, longitude, and altitude to UTM coordinates."""
    utm_x, utm_y, utm_z = transformer.transform(lon, lat, alt)
    return utm_x, utm_y, utm_z

@dataclass
class GPSPoint:
    timestamp: int
    position: np.ndarray


def process_gps_data(file_path, verbose=False) -> list[GPSPoint]:
    points = []
    """Process a file to read GPS data and convert it to UTM."""
    with open(file_path, 'r') as file:
        for line in file:
            # Check if line contains 'GPS[0]' and match the coordinate pattern, since GPS[0] and GPS[1] are the same
            if 'GPS[0]' in line:
                match = re.search(r"- ([\d]+) lat: ([\d\.]+) lon: ([\d\.]+) alt: ([\d\.]+)", line)
                if match:
                    timestamp = int(match.group(1))
                    lat = float(match.group(2))
                    lon = float(match.group(3))
                    alt = float(match.group(4))
                    utm_x, utm_y, utm_z = convert_to_utm(lat, lon, alt)
                    if verbose:
                        print(f"Original: timestamp={timestamp} lat={lat}, lon={lon}, alt={alt}")
                        print(f"Converted to UTM: X={utm_x}, Y={utm_y}, Z={utm_z}")
                    points.append(GPSPoint(timestamp, np.array([utm_x, utm_y, utm_z])))
    return points