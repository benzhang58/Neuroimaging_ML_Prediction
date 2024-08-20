# MRI_DTI_Machinelearning_Analysis
Neural network models for regression and classification tasks aimed at predicting cognition and depression scores based on neuroimaging variables (grey matter volume, fractional anisotropy, etc) along with blood &amp; urine markers.



MRI Analysis Steps
1. fsl_anat.sh :
-    This uses the fsl_anat function on all the participants. "fsl_anat" is a comprehensive pipeline in FSL that performs several preprocessing steps, including : bias correction, registration to MNI   space, and segmentation. Everything is output in the T1_struct.anat folder. We use command "-noseg" and "nosubcortseg" to tell the command to skip tissue-type segmentation and subcortical structure segmentation - we will do this later. The brain extraction is called T1_biascorr_brain.nii.gz. 

2. 
