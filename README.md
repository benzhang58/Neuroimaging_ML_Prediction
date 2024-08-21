# MRI Machine Learning Analysis 

Neural network models for regression and classification tasks aimed at predicting cognitionm, intelligence, and depression scores based on neuroimaging variables (grey matter volume, fractional anisotropy, etc) along with blood, urine, and other markers.



##MRI Analysis Steps

1. ###fsl_anat.sh :
-   This uses the fsl_anat function on all the participants. "fsl_anat" is a comprehensive pipeline in FSL that performs several preprocessing steps, including : bias correction, registration to MNI   space, and segmentation. Everything is output in the T1_struct.anat folder. We use command "-noseg" and "nosubcortseg" to tell the command to skip tissue-type segmentation and subcortical structure segmentation - we will do this later. The brain extraction is called T1_biascorr_brain.nii.gz. 

2. T1_optiBET.sh :
-   Applies Optimized Brain Extraction for Pathological Brains (optiBET) technique in order to segment brain from non-brain tissue (i.e. skull stripping). This is a more optimized brain extraction method compared to others methods like normal BET, but takes a bit longer to run. Calculates initial brain extraction, then employes linear and nonlinear registration to project the extraction to standard MNI template space. It then back-projects a standard brain-only mask from MNI space to the subject's native T1 space, using this brain-only mask to segment out non-brain tissue.
-   This technique is described in [Lutkenhoff ES, Rosenberg M, Chiang J, Zhang K, Pickard JD, et al. (2014) Optimized Brain Extraction for Pathological Brains (optiBET). PLOS ONE 9(12): e115551. https://doi.org/10.1371/journal.pone.0115551]
-   Script (open source) can be found in [https://montilab.psych.ucla.edu/fmri-wiki/optibet/], and I've also uploaded it to the repository
-   Output is T1_biascorr_optiBET_brain.nii.gz

3. fsleyes_optiBET.sh :
-   Iterates through each participants' T1_biascorr_optiBET_brain file, opening it with fsleyes, allowing for manual inspection of the brain extraction

4. fast_tissue_segmentation.sh : 
-   Iterates through each participant, applying FAST tissue segmentation to each participants T1_biascorr_optiBET_brain.
-   Generates output files for CSF, white matter, and grey matter segmentation

5. struct_to_MNI.sh
-   Applies flirt command from fsl in order to warp T1_biascorr_optiBET_brain to standard MNI space, and calculates the transformation matrix from participant's native T1 structural space to standard space
-   Then uses flirt command, applying the acquired transformation matrix, to warp the grey matter segmentation (derived from FAST) into standard space

6. ROI_overlay_in_MNI.sh
-   Binarizes the grey matter segmentation
-   Thresholds atlas of choice (in this case Harvard Oxford cortical & subcortical) into regions of interest (ROI) using fslmaths command 
-   Masks the binarized grey matter segmentation with these different ROIs, and then calculates volume of grey matter in each ROI
-   Saves the grey matter volume of each ROI for each participant in a .txt file

7. combine_outputs.sh
-   Loops through each participant, and creates a combined .txt file that consists of a column for their ID's, along with a column for each ROI, as well as their corresponding grey matter volumes for each ROI
-   Can be copy pasted into spreadsheet 




