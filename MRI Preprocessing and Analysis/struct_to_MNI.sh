#!/bin/bash

base_directory="/Volumes/exhale/MRI/OpenNeuro/Participants/"

# MNI template path
mni_template="/Volumes/exhale/MRI/MNI/MNI152_T1_1mm_brain.nii.gz"

# Loop through each participant folder
for participant_dir in "${base_directory}"*/; 
do
    echo "Processing: ${participant_dir}"
    
    # Define the paths to the input files
    t1_file="${participant_dir}ses-01/T1_struct.anat/T1_biascorr_optiBET_brain.nii.gz"
    gm_segmentation="${participant_dir}ses-01/T1_struct.anat/T1_biascorr_optiBET_brain_pve_1.nii.gz"
    
    # Define output paths
    output_t1_mni="${participant_dir}ses-01/T1_struct.anat/T1_optiBET_in_MNI.nii.gz"
    output_gm_mni="${participant_dir}ses-01/T1_struct.anat/GM_seg_MNI.nii.gz"
    transformation_matrix="${participant_dir}ses-01/T1_struct.anat/T1_optiBET_to_MNI.mat"
    
    # Warp the T1 to MNI space
    flirt -in "$t1_file" -ref "$mni_template" -out "$output_t1_mni" -omat "$transformation_matrix" -dof 6 -interp nearestneighbour
    
    # Apply the warp to the grey matter segmentation
    flirt -in "$gm_segmentation" -ref "$mni_template" -out "$output_gm_mni" -applyxfm -init "$transformation_matrix" -interp nearestneighbour
    
    echo "Finished processing: ${participant_dir}"
done
