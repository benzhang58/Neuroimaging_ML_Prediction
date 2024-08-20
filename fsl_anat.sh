#!/bin/bash

# Directory containing participant folders
base_directory="/Volumes/exhale/MRI/OpenNeuro/Participants/"

# Loop through each participant folder
for participant_dir in "${base_directory}"*/; 
do
    echo "Processing ${participant_dir}"

    # Define the session 1 directory
    ses01_dir="${participant_dir}ses-01/"
    
    # Define the anat directory inside ses-01
    anat_dir="${ses01_dir}anat/"

    # Find the FSPGR or MPRAGE file in the anat directory
    fspgr_file=$(find "$anat_dir" -type f -name "sub*FSPGR*.nii*")
    mprage_file=$(find "$anat_dir" -type f -name "sub*MPRAGE*.nii*")

    # Check which file exists and run fsl_anat accordingly
    if [ -n "$fspgr_file" ]; then
        echo "Found FSPGR file: $fspgr_file"
        fsl_anat --noreg --nononlinreg --noseg --nosubcortseg -o "${ses01_dir}T1_struct" -i "$fspgr_file"
    elif [ -n "$mprage_file" ]; then
        echo "Found MPRAGE file: $mprage_file"
        fsl_anat --noreg --nononlinreg --noseg --nosubcortseg -o "${ses01_dir}T1_struct" -i "$mprage_file"
    else
        echo "No FSPGR or MPRAGE file found in ${anat_dir}."
    fi
done
