# SpineTool
[Ekaterina, P., Peter, V., Smirnova, D., Vyacheslav, C., Ilya, B. (2023). SpineTool is an open-source software for analysis of morphology of dendritic spines. Scientific Reports.13. 
10.1038/s41598-023-37406-4.](https://doi.org/10.1038/s41598-023-37406-4)

Dendritic spines form most excitatory synaptic inputs in neurons and these spines are altered in many neurodevelopmental 
and neurodegenerative disorders. Reliable methods to assess and quantify dendritic spines morphology are needed, but most 
existing methods are subjective and labor intensive. To solve this problem, we developed an open-source software that 
allows segmentation of dendritic spines from 3D images, extraction of their key morphological features, and their 
classification and clustering. Instead of commonly used spine descriptors based on numerical metrics we used chord 
length distribution histogram (CLDH) approach. CLDH method depends on distribution of lengths of chords randomly 
generated within dendritic spines volume. To achieve less biased analysis, we developed a classification procedure that 
uses machine-learning algorithm based on experts’ consensus and machine-guided clustering tool. These approaches to 
unbiased and automated measurements, classification and clustering of synaptic spines that we developed should provide 
a useful resource for a variety of neuroscience and neurodegenerative research applications.


- [SpineTool Tutorial](https://static-content.springer.com/esm/art%3A10.1038%2Fs41598-023-37406-4/MediaObjects/41598_2023_37406_MOESM1_ESM.pdf)

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

## Citation
```
TY  - JOUR
AU  - Ekaterina, Pchitskaya
AU  - Peter, Vasiliev
AU  - Smirnova, Daria
AU  - Vyacheslav, Chukanov
AU  - Ilya, Bezprozvanny
PY  - 2023/06/29
SP  - 
T1  - SpineTool is an open-source software for analysis of morphology of dendritic spines
VL  - 13
DO  - 10.1038/s41598-023-37406-4
JO  - Scientific Reports
ER  - 
```
