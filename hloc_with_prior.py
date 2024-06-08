from pathlib import Path
from hloc.utils.database import COLMAPDatabase
import numpy as np
import pycolmap
from hloc import (
    extract_features,
    match_features,
    reconstruction,
    visualization,
    pairs_from_retrieval,
    pairs_from_exhaustive
)

from hloc_mod_rec import recon_main_post, recon_main_pre
from naplab.naplab_dataset import NaplabDataset, create_transform_json

def set_image(db: COLMAPDatabase, id, rotation: np.ndarray, translation: np.ndarray, cam_id: int):
    db.execute('''
        UPDATE images
        SET camera_id = ?,
            prior_qw = ?,
            prior_qx = ?,
            prior_qy = ?,
            prior_qz = ?,
            prior_tx = ?,
            prior_ty = ?,
            prior_tz = ?
        WHERE
            image_id = ?;
    ''', (
        cam_id,
        rotation[0],
        rotation[1],
        rotation[2],
        rotation[3],
        translation[0],
        translation[1],
        translation[2],
        id
    ))

def create_dataset_from_rig(rig_path: Path, output_path: Path, left_gps_file: Path, right_gps_file: Path, fps=1, n=-1):
    nap = NaplabDataset(left_gps_file.as_posix(), right_gps_file.as_posix(), rig_json_path=rig_path.as_posix(), fps=fps, n = n)
    nap.create_nerfstudio_dataset(output_path.as_posix())
    images = output_path / "images"
    outputs = output_path / "sfm"
    sfm_pairs = outputs / "pairs-netvlad.txt"
    sfm_dir = outputs / "sfm_superpoint+superglue"

    retrieval_conf = extract_features.confs["netvlad"]
    feature_conf = extract_features.confs["disk"]
    matcher_conf = match_features.confs["disk+lightglue"]
    retrieval_path = extract_features.main(retrieval_conf, images, outputs)
    pairs_from_retrieval.main(retrieval_path, sfm_pairs, num_matched=10)
    feature_path = extract_features.main(feature_conf, images, outputs)
    match_path = match_features.main(
        matcher_conf, sfm_pairs, feature_conf["output"], outputs
    )
    intermediate = recon_main_pre(sfm_dir, images, sfm_pairs, feature_path, match_path, camera_mode=pycolmap.CameraMode.SINGLE)
    image_ids = intermediate[5]

    db = COLMAPDatabase.connect(intermediate[1])
    db.execute("DELETE FROM cameras")
    for cam in nap.cameras:
        k1, k2, k3, k4 = cam.calculate_distortion_coeff()
        db.add_camera(4, cam.width, cam.height, [cam.fx, cam.fy, cam.cx, cam.cy, 0, 0, 0, 0], camera_id=cam.id)

    # modify images in database with priors
    for it in nap.all_images_with_transforms:
        cam = it.camera
        for frame, image_transform in zip(it.frames, it.images_with_transforms):
            name = Path(image_transform.image_path).name
            id = image_ids[name]
            set_image(db, id, cam.get_quaternion(frame), cam.get_translation_vector(frame), cam.id)
    db.commit()
    

    model = recon_main_post(*intermediate)
    create_transform_json(model, images, output_path)