# Neural Network w/ Cortical & Subcortical Grey Matter Volume

Neural network model for regression task aimed at predicting intelligence (as measured by the Kaufman Brief Intelligence Test, Second Edition) based on neuroimaging variables (cortical/subcortical grey matter volume, fractional anisotropy, etc) along with blood, urine, metabolic health, and other clinical measures. 

<p align="center">
  <img src="https://github.com/user-attachments/assets/900c3284-bd00-45de-a12f-33429df09b60" alt="image">
</p>


## Project Summary
#### 1. MRI Data Processing: 
-  Analyzed MRI data from a public dataset of 157 subjects obtained from OpenNeuro to extract grey matter volumes in cortical regions of interest (detailed analysis process provided below). Clinical data (blood, urine, vitamin levels, etc) is also obtained for each participant.
#### 2. Neural Network Development: 
-  Manually developed a four-layer neural network model, utilizing the Leaky ReLU activation function.
#### 3. Feature Engineering: 
-  Executed a feature engineering process to identify the most relevant features from over 100 potential variables.
#### 4. Model Training and Validation: 
-  Trained the neural network on the processed dataset and evaluated model performance using k-fold cross-validation.
#### 5. Test on my own T1-weighted MRI scan
-  Analyzed my own T1-weighted MRI data to extract cortical grey matter volumes and tested the model with this data for further validation.

<table align="center">
  <tr>
    <td align="center" style="text-align:center;">
      <div>
        <img src="https://github.com/user-attachments/assets/ad6c01bb-e893-45db-8949-9709736bd5c6" alt="Right Cortical GM Heatmap" width="300px"/>
        <div style="text-align:center; margin-top: 5px;">Right Cortical ROI Grey Matter Volume Heatmap</div>
      </div>
    </td>
    <td align="center" style="text-align:center;">
      <div>
        <img src="https://github.com/user-attachments/assets/15c9336e-bc4c-40f6-9f4f-9cf49c3680bc" alt="Left Cortical GM Heatmap" width="300px"/>
        <div style="text-align:center; margin-top: 5px;">Left Cortical ROI Grey Matter Volume Heatmap</div>
      </div>
    </td>
    <td align="center" style="text-align:center;">
      <div>
        <img src="https://github.com/user-attachments/assets/654e03fb-e7f0-4a02-93d3-b359c538647e" alt="Subcortical Heatmap" width="300px"/>
        <div style="text-align:center; margin-top: 5px;">Subcortical ROI Grey Matter Volume Heatmap</div>
      </div>
    </td>
  </tr>
</table>


## MRI Analysis Process

#### 1. fsl_anat.sh :
-   This uses the fsl_anat function on all the participants. "fsl_anat" is a comprehensive pipeline in FSL that performs several preprocessing steps, including : bias correction, registration to MNI   space, and segmentation. Everything is output in the T1_struct.anat folder. Command "-noseg" and "nosubcortseg" to tell the command to skip tissue-type segmentation and subcortical structure segmentation - will do this later. Also skip the brain extraction step because we want to use the optiBET tool to extract brain instead. The output of this step is called T1_biascorr_brain.nii.gz. 

#### 2. T1_optiBET.sh :
-   Applies Optimized Brain Extraction for Pathological Brains (optiBET) technique in order to segment brain from non-brain tissue (i.e. skull stripping). This is a more optimized brain extraction method compared to others methods like normal BET, but takes a bit longer to run. Calculates initial brain extraction, then employes linear and nonlinear registration to project the extraction to standard MNI template space. It then back-projects a standard brain-only mask from MNI space to the subject's native T1 space, using this brain-only mask to segment out non-brain tissue.
-   This technique is described in [Lutkenhoff ES, Rosenberg M, Chiang J, Zhang K, Pickard JD, et al. (2014) Optimized Brain Extraction for Pathological Brains (optiBET). PLOS ONE 9(12): e115551. https://doi.org/10.1371/journal.pone.0115551]
-   Script (open source) can be found in [https://montilab.psych.ucla.edu/fmri-wiki/optibet/], and I've also uploaded it to the repository
-   Output is T1_biascorr_optiBET_brain.nii.gz

#### 3. fsleyes_optiBET.sh :
-   Iterates through each participants' T1_biascorr_optiBET_brain file, opening it with fsleyes, allowing for manual inspection of the brain extraction

#### 4. fast_tissue_segmentation.sh : 
-   Iterates through each participant, applying FAST tissue segmentation to each participants T1_biascorr_optiBET_brain.
-   Generates output files for CSF, white matter, and grey matter segmentation

#### 5. struct_to_MNI.sh
-   Applies flirt command from fsl in order to apply linear registration on T1_biascorr_optiBET_brain to standard MNI space, and calculates the transformation matrix from participant's native T1 structural space to standard space. I opted to not use non-linear registration due to long computation time with a large dataset. 
-   Then uses flirt command, applying the acquired transformation matrix, to warp the grey matter segmentation (derived from FAST) into standard space

#### 6. Manual Check
-   Manually check the overlay between the atlas and grey matter segmentation in standard space, taking note of any subjects where the masking is particularly bad.

#### 7. ROI_overlay_in_MNI.sh
-   Binarizes the grey matter segmentation
-   Thresholds atlas of choice (in this case Harvard Oxford cortical & subcortical) into regions of interest (ROI) using fslmaths command 
-   Masks the binarized grey matter segmentation with these different ROIs, and then calculates volume of grey matter in each ROI
-   Saves the grey matter volume of each ROI for each participant in a .txt file

#### 8. combine_outputs.sh
-   Loops through each participant, and creates a combined .txt file that consists of a column for their ID's, along with a column for each ROI, as well as their corresponding grey matter volumes for each ROI
-   Can be copy pasted into spreadsheet 

#### 9. Outlier_Impute.py
-   Detects outliers for each variable (can set Z-score, I set 3 standard deviations)
-   Performs kNN imputation 5 nearest neighbours, and keeps track of which values were imputed
-   Outputs a dataset with the outliers imputed along with all the rest of the data, and also outputs an empty dataset with only the imputed values shown 


#### Extra Notes 
Final neuroimaging analyses were conducted in standard MNI space for ease of processing. As the goal of this project was practice creating prediction models that can ultimately be applied in clinical settings, comparing in standard space allows for better comparison across a wide range of individuals. It also allows anyone who wants to try the model to skip the inverse warp step of analysis in which the transformation matrix between the atlas (Harvard Oxford cortical/subcortical) space and individual's native structural space must also be computed. This open source data comes from the National Institute of Mental Health (NIMH) Intramural Healthy Volunteer Dataset from OpenNeuro : 

[Nugent, A. C., Thomas, A. G., Mahoney, M., Gibbons, A., Smith, J. T., Charles, A. J., Shaw, J. S., Stout, J. D., Namyst, A. M., Basavaraj, A., Earl, E., Riddle, T., Snow, J., Japee, S., Pavletic, A. J., Sinclair, S., Roopchansingh, V., Bandettini, P. A., & Chung, J. (2022). The NIMH intramural healthy volunteer dataset: A comprehensive MEG, MRI, and behavioral resource. Scientific Data, 9, Article 518. https://doi.org/10.1038/s41597-022-01623-9]

Model was trained on MRI, DTI, and clinical data from a dataset of 157 healthy research volunteers. 
As two participants lacked structural MRI data, they were excluded from the analysis. Furthermore, four participants had poor brain extractions, resulting in poor alignment grey matter segmentations and standard space atlas, so they were also excluded. This resulted in a final cohort of 151 participants. Participant eligibility and imaging procedures can be found on the website : https://openneuro.org/datasets/ds004215/versions/1.0.3

### Harvard-Oxford Cortical / Subcortical Structural Atlas derived from : 

Makris N, Goldstein JM, Kennedy D, Hodge SM, Caviness VS, Faraone SV, Tsuang MT, Seidman LJ. Decreased volume of left and total anterior insular lobule in schizophrenia. Schizophr Res. 2006 Apr;83(2-3):155-71 Frazier JA, Chiu S, Breeze JL, Makris N, Lange N, Kennedy DN, Herbert MR, Bent EK, Koneru VK, Dieterich ME, Hodge SM, Rauch SL, Grant PE, Cohen BM, Seidman LJ, Caviness VS, Biederman J. Structural brain magnetic resonance imaging of limbic and thalamic volumes in pediatric bipolar disorder. Am J Psychiatry. 2005 Jul;162(7):1256-65 Desikan RS, Ségonne F, Fischl B, Quinn BT, Dickerson BC, Blacker D, Buckner RL, Dale AM, Maguire RP, Hyman BT, Albert MS, Killiany RJ. An automated labeling system for subdividing the human cerebral cortex on MRI scans into gyral based regions of interest. Neuroimage. 2006 Jul 1;31(3):968-80. Goldstein JM, Seidman LJ, Makris N, Ahern T, O'Brien LM, Caviness VS Jr, Kennedy DN, Faraone SV, Tsuang MT. Hypothalamic abnormalities in schizophrenia: sex effects and genetic vulnerability. Biol Psychiatry. 2007 Apr 15;61(8):935-45

Files are in documentation folder 


# Final Features  : 
- Left lateral occipital cortex (inferior division)
- Right juxtapositional lobule cortex
- Left frontal operculum cortex
- Left accumbens 

## Feature Selection Process

1. Computed correlation heatmap using target variables of standardized IQ (stand_IQ) and raw IQ (raw_IQ) along with all other features
2. Utilized Top_correlations.py code to print top 10 features correlated with variable of interest
3. Used RFE_nonlinear_featureselection.py to perform recursive feature elimination in order to find balance between reducing multicollinearity and maximizing correlation between features and variables of interest
4. Also tried Lasso_feature_selection.py code to perform Lasso feature selection to better identify features to include
5. Used tensorflow_permutation.py code to shuffle features and calculate permutation importance 
6. Looked through relevant literature on intelligence and brain volume 
7. Combined information derived from all these areas, along with testing different combinations of features in order to determine best performing set of features to train neural network with

