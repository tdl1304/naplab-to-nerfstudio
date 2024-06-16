# naplab-to-nerfstudio
Convert naplab to nerfstudio data format, building a SfM model based on GNSS data and camera rig specified as a json file.

Example camera rig file found in nobadcam.json

## Download Trip094 (Or other trips)
Request permission first, or request access from NTNU IDI
```
pip install gdown
gdown 15YWaN8pq19Oeo0SEx3UKi0Y8O976UM0M
unzip Trip094.zip
```

## Install requirements
```
pip install -r requirements.txt
```

### Install HLOC
https://github.com/cvg/Hierarchical-Localization

## RUN program
```
python main.py
```


## (Optional) Process in nerfstudio with built SfM model
```
ns-process-data images --data ./nerfstudio/images/ --output-dir ./test --skip-colmap --colmap-model-path ../outputs/sfm/sfm_superpoint+superglue/ --sfm-tool hloc
```
