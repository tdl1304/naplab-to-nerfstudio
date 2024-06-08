from hloc_with_prior import create_dataset_from_rig
from pathlib import Path
create_dataset_from_rig(Path("Trip094/singlecam.json"), Path("nerfstudio"), Path("Trip094/gnss094_50.txt"), Path("Trip094/gnss094_52.txt"))