#!/bin/bash

base_directory="/Volumes/exhale/MRI/OpenNeuro/Participants/"

# Loop through each participant folder
for participant_dir in "${base_directory}"*/; 
do
    file_path="${participant_dir}ses-01/T1_struct.anat/T1_biascorr_optiBET_brain.nii.gz"
    
    if [ -f "$file_path" ]; then
        echo "Opening FSLeyes for: ${file_path}"
        fsleyes "$file_path"
    else
        echo "File not found: ${file_path}"
    fi
    
done
