#!/bin/bash

base_directory="/Volumes/exhale/MRI/OpenNeuro/Participants/"

# List of participants to exclude
excluded_participants=() ;

# Loop through each participant folder
for participant_dir in "${base_directory}"*/; 
do
    participant=$(basename "${participant_dir}")
    
    # Check if the participant is in the exclusion list
    if [[ " ${excluded_participants[@]} " =~ " ${participant} " ]]; then
        echo "Skipping: ${participant}"
        continue
    fi
    
    echo "Processing: ${participant_dir}"
    
    # Define the path to the T1_biascorr.nii.gz file within the current participant directory
    input_file="${participant_dir}ses-01/T1_struct.anat/T1_biascorr.nii.gz"
    
    # Check if the file exists
    if [ -f "$input_file" ]; then
        echo "Running optiBET on $input_file"
        sh /Volumes/exhale/MRI/OpenNeuro/scripts/optiBET.sh -i "$input_file"
    else
        echo "File not found: $input_file"
    fi
done
