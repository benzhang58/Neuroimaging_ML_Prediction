#!/bin/bash

# Define directories
base_directory="/Volumes/exhale/MRI/OpenNeuro/Participants/"
atlas="/Volumes/exhale/MRI/Atlases/HarvardOxford-sub-maxprob-thr25-1mm.nii.gz" # Replace with your actual atlas path
output_dir="${base_directory}/GM_ROI_subcortical_Volumes/"

# Create output directory if it doesn't exist
mkdir -p "$output_dir"

# Loop through each participant folder
for participant_dir in "${base_directory}"*/; 
do
    echo "Processing: ${participant_dir}"

    # Define paths to the grey matter mask and output files
    gm_mask="${participant_dir}ses-01/T1_struct.anat/GM_seg_MNI.nii.gz"
    participant_id=$(basename "$participant_dir")
    roi_output="${output_dir}/${participant_id}_HarvardOxford_subcortical_Bin_Volumes.txt"
    
    # Check if the grey matter mask exists
    if [ ! -f "$gm_mask" ]; then
        echo "Grey matter mask not found for participant ${participant_id}, skipping."
        continue
    fi

    # Binarize the grey matter mask
  	# fslmaths "$gm_mask" -bin "${participant_dir}ses-01/T1_struct.anat/GM_seg_MNI_bin.nii.gz"
    
    # Create ROIs directory within the participant's folder
    rois_dir="${participant_dir}ses-01/T1_struct.anat/subcortical_ROIs/"
    mkdir -p "$rois_dir"
    
    # Initialize ROI volume output file
    echo "ROI_ID,Volume" > "$roi_output"
    
    # Apply mask using the atlas and calculate grey matter volume in each ROI
    for roi_id in {1..21}; do  # Replace 48 with the actual number of ROIs in your atlas
    
        # Isolate the specific ROI in the atlas
        fslmaths "$atlas" -thr $roi_id -uthr $roi_id "${rois_dir}ROI_${roi_id}_subcortical_mask.nii.gz"
        
        # Mask the binarized GM segmentation with this ROI
        fslmaths "${participant_dir}ses-01/T1_struct.anat/GM_seg_MNI_bin.nii.gz" -mas "${rois_dir}ROI_${roi_id}_subcortical_mask.nii.gz" "${rois_dir}GM_in_ROI_${roi_id}_subcortical.nii.gz"
        
        # Calculate the volume of the GM in the specific ROI
        roi_volume=$(fslstats "${rois_dir}GM_in_ROI_${roi_id}_subcortical.nii.gz" -V | awk '{print $2}')
        echo "${roi_id},${roi_volume}" >> "$roi_output"
    done

    echo "Finished processing: ${participant_id}"
done

echo "All participants processed."
