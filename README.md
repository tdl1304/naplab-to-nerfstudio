# naplab-to-nerfstudio
Convert naplab to nerfstudio data format

## Install Trip094
```
pip install gdown
gdown 15YWaN8pq19Oeo0SEx3UKi0Y8O976UM0M
unzip Trip094.zip
```

## Install requirements
```
pip install -r requirements.txt
```

## Process in nerfstudio
```
ns-process-data images --data ./nerfstudio/images/ --output-dir ./test --skip-colmap --colmap-model-path ../outputs/sfm/sfm_superpoint+superglue/ --sfm-tool hloc
```
