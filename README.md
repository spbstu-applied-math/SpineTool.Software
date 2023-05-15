# SpineTool
Dendritic spine analysis tool for dendritic spine image segmentation, dendritic spine morphologies extraction,analysis and clustering.

## System requirements
- Windows 8 or newer
- Minimum 1GB RAM
- Minimum 6 GB disk space

## Install
1. Download code
2. Unzip [CGAL files](https://github.com/pv6/cgal-swig-bindings/releases/download/python-build/CGAL.zip) next to code, e.g. `PATH_TO_CODE\CGAL\...`
3. Install [Anaconda](https://www.anaconda.com/)
4. Open Anaconda
5. Execute
```cmd
cd PATH_TO_CODE
conda create --name spine-analysis -c conda-forge --file requirements.txt -y
```
## Run
1. Open Anaconda
2. Execute
```cmd
cd PATH_TO_CODE
conda activate spine-analysis
jupyter notebook
```

## Example datasets
### example_dendrite
Dataset consist of a .tif image of a dendrite, 22 polygonal
meshes of dendrite spines and a dendrite polygonal mesh computed with `dendrite-segmentation.ipynb` notebook. This 
example dataset provides a demonstration of dendrite image segmentation performance and functionality of the 
methods from the `Utilities.ipynb` notebook.
### 0.025 0.025 0.1 dataset
Dataset consists of 270 polygonal meshes of dendrite spines related to 54 dendrites and of 54
polygonal   meshes   for   dendrites computed with `dendrite-segmentation.ipynb` notebook.  A dataset subdirectory 
named "manual_classification" contains the expert markups from 8 people obtained using the `spine-manual-classification.ipynb` 
and the results of merging the classifications to obtain a consensus classification. This  example dataset provides a 
demonstration of dendrite spines classification and clustering performance and functionality of the 
methods from the `Utilities.ipynb` notebook.
