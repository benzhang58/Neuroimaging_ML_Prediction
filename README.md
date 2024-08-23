# Neural Network w/ Neuroimaging Variables  

Neural network models for regression tasks aimed at predicting intelligence (as measured by the Kaufman Brief Intelligence Test, Second Edition) based on neuroimaging variables (cortical/subcortical grey matter volume, fractional anisotropy, etc) along with blood, urine, metabolic health, and other clinical measures. 

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
  </tr>
</table>







## MRI Analysis Process

### 1. fsl_anat.sh :
-   This uses the fsl_anat function on all the participants. "fsl_anat" is a comprehensive pipeline in FSL that performs several preprocessing steps, including : bias correction, registration to MNI   space, and segmentation. Everything is output in the T1_struct.anat folder. Command "-noseg" and "nosubcortseg" to tell the command to skip tissue-type segmentation and subcortical structure segmentation - will do this later. Also skip the brain extraction step because we want to use the optiBET tool to extract brain instead. The output of this step is called T1_biascorr_brain.nii.gz. 

### 2. T1_optiBET.sh :
-   Applies Optimized Brain Extraction for Pathological Brains (optiBET) technique in order to segment brain from non-brain tissue (i.e. skull stripping). This is a more optimized brain extraction method compared to others methods like normal BET, but takes a bit longer to run. Calculates initial brain extraction, then employes linear and nonlinear registration to project the extraction to standard MNI template space. It then back-projects a standard brain-only mask from MNI space to the subject's native T1 space, using this brain-only mask to segment out non-brain tissue.
-   This technique is described in [Lutkenhoff ES, Rosenberg M, Chiang J, Zhang K, Pickard JD, et al. (2014) Optimized Brain Extraction for Pathological Brains (optiBET). PLOS ONE 9(12): e115551. https://doi.org/10.1371/journal.pone.0115551]
-   Script (open source) can be found in [https://montilab.psych.ucla.edu/fmri-wiki/optibet/], and I've also uploaded it to the repository
-   Output is T1_biascorr_optiBET_brain.nii.gz

### 3. fsleyes_optiBET.sh :
-   Iterates through each participants' T1_biascorr_optiBET_brain file, opening it with fsleyes, allowing for manual inspection of the brain extraction

### 4. fast_tissue_segmentation.sh : 
-   Iterates through each participant, applying FAST tissue segmentation to each participants T1_biascorr_optiBET_brain.
-   Generates output files for CSF, white matter, and grey matter segmentation

### 5. struct_to_MNI.sh
-   Applies flirt command from fsl in order to apply linear registration on T1_biascorr_optiBET_brain to standard MNI space, and calculates the transformation matrix from participant's native T1 structural space to standard space. I opted to not use non-linear registration due to long computation time with a large dataset. 
-   Then uses flirt command, applying the acquired transformation matrix, to warp the grey matter segmentation (derived from FAST) into standard space

### 6. Manual Check
-   Manually check the overlay between the atlas and grey matter segmentation in standard space, taking note of any subjects where the masking is particularly bad.

### 7. ROI_overlay_in_MNI.sh
-   Binarizes the grey matter segmentation
-   Thresholds atlas of choice (in this case Harvard Oxford cortical & subcortical) into regions of interest (ROI) using fslmaths command 
-   Masks the binarized grey matter segmentation with these different ROIs, and then calculates volume of grey matter in each ROI
-   Saves the grey matter volume of each ROI for each participant in a .txt file

### 8. combine_outputs.sh
-   Loops through each participant, and creates a combined .txt file that consists of a column for their ID's, along with a column for each ROI, as well as their corresponding grey matter volumes for each ROI
-   Can be copy pasted into spreadsheet 

### 9. Outlier_Impute.py
-   Detects outliers for each variable (can set Z-score, I set 3 standard deviations)
-   Performs kNN imputation 5 nearest neighbours, and keeps track of which values were imputed
-   Outputs a dataset with the outliers imputed along with all the rest of the data, and also outputs an empty dataset with only the imputed values shown 


Final neuroimaging analyses were conducted in standard MNI space for ease of processing. As the goal of this project was practice creating prediction models that can ultimately be applied in clinical settings, comparing in standard space allows for better comparison across a wide range of individuals. It also allows anyone who wants to try the model to skip the inverse warp step of analysis in which the transformation matrix between the atlas (Harvard Oxford cortical/subcortical) space and individual's native structural space must also be computed. This open source data comes from the National Institute of Mental Health (NIMH) Intramural Healthy Volunteer Dataset from OpenNeuro : 

[Nugent, A. C., Thomas, A. G., Mahoney, M., Gibbons, A., Smith, J. T., Charles, A. J., Shaw, J. S., Stout, J. D., Namyst, A. M., Basavaraj, A., Earl, E., Riddle, T., Snow, J., Japee, S., Pavletic, A. J., Sinclair, S., Roopchansingh, V., Bandettini, P. A., & Chung, J. (2022). The NIMH intramural healthy volunteer dataset: A comprehensive MEG, MRI, and behavioral resource. Scientific Data, 9, Article 518. https://doi.org/10.1038/s41597-022-01623-9]

Model was trained on MRI, DTI, and clinical data from a dataset of 157 healthy research volunteers. 
As two participants lacked structural MRI data, they were excluded from the analysis. Furthermore, four participants had poor brain extractions, resulting in poor alignment grey matter segmentations and standard space atlas, so they were also excluded. This resulted in a final cohort of 151 participants. Participant eligibility and imaging procedures can be found on the website : https://openneuro.org/datasets/ds004215/versions/1.0.3
