- # Anhui Soil Dataset

  ## Overview

  The Anhui Soil Dataset comprises soil samples predominantly collected from the Huangshan area and Shitai County, located in southern Anhui Province. The dataset was curated with an emphasis on monitoring urban soil's environmental quality, providing a comprehensive overview and description of soil functionality and conditions. This is particularly relevant for ensuring the sustainable development of Anhui Province.

  ## Data Collection

  A five-point diagonal sampling methodology was employed to amalgamate the soil samples. At every sampling point, one sample was extracted from a depth ranging between 0-20 cm. These individual samples were then amalgamated into a composite sample. Each of these composite samples, approximately weighing 1.5 kg, constituted a part of the 188 specimens collected.

  Post-collection, these specimens underwent an air drying process in a lab, which was followed by comminution. The samples required for the experiment were isolated using a size-20 mesh filtration process. Every sample was further subjected to near-infrared spectral data exploration and a series of chemical tests.

  ## Spectral Information

  The dataset includes near-infrared (NIR) spectral data. The spectral range spans from 901.57 nm to 1701.18 nm, encompassing 228 bands. Near-infrared spectroscopy coincides with the absorption regions associated with overtone and combination vibrations of hydrogen-containing groups (OH, NH, CH, SH) present in organic molecules. By scanning the near-infrared spectrum of a sample, characteristic information of the organic molecular hydrogen-containing groups within the sample can be obtained. This makes NIR a potent medium for capturing sample information. 

  ## Attributes
  
  The primary attributes of interest in this dataset are:

- **Available Nitrogen (N)**: Measures the amount of nitrogen available in the soil.
- **Available Phosphorus (P)**: Measures the amount of phosphorus available in the soil.
  
- **Available Potassium (K)**: Measures the potassium capacity available in the soil.
  

The statistical summary of these attributes is provided in Table 1.

  ![Soil Parameters Distributions](Soil_Parameters_Distributions.png)

  **Table 1. Statistical Summary of the Anhui Soil Dataset**

| Property                      | Mean    | Standard Deviation | Minimum | Maximum |
| ----------------------------- | ------- | ------------------ | ------- | ------- |
| Available nitrogen (g.kg-1)   | 0.17908 | 0.14478            | 0.01932 | 0.73158 |
| Available phosphorus (g.kg-1) | 0.04551 | 0.07367            | 0.00746 | 0.29718 |
| Available potassium (g.kg-1)  | 0.19064 | 0.13309            | 0.06000 | 0.67000 |

## Dataset Files:

- [Original Dataset (CSV)](AnHui.HuangShan.SOIL.csv)
- [Training Set (CSV)](AnHui.HuangShan.SOIL.train.csv)
- [Test Set (CSV)](AnHui.HuangShan.SOIL.test.csv)
  
## Auxiliary Scripts:

- [Dataset Splitting Script](soil_data_lhs_splitter.py)
  
- [Visualization Script](VisualizationOfSoilDistribution.py)
  
  ## Usage
  

The Anhui Soil Dataset can be employed for various research purposes, including but not limited to:

- **Soil Quality Monitoring**: Understand and monitor the environmental quality of urban soil.
- **Soil Attribute Prediction**: Utilize machine learning models to predict certain soil attributes based on the given features.
- **Soil Classification**: Classify soil types based on the attributes provided.
  
  ## License
  
  Please refer to the dataset's original source or repository for licensing information.