#!/bin/bash

base_directory="/Volumes/exhale/MRI/OpenNeuro/Participants/"

# Loop through each participant folder
for participant_dir in "${base_directory}"*/; 
do
    file_path="${participant_dir}ses-01/T1_struct.anat/T1_biascorr_optiBET_brain.nii.gz" # Assuming you have normalized them to 1mm resolution
    
    if [ -f "$file_path" ]; then
        echo "Performing FAST tissue segmentation on: ${file_path}"
        
        # Perform FAST segmentation
        fast "$file_path"
        
        echo "FAST tissue segmentation completed for: ${file_path}"
        
    else
        echo "File not found: ${file_path}"
    fi
    
done
