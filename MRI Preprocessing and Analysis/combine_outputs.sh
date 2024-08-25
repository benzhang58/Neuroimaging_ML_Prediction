#!/bin/bash

# Define directories
base_directory="/Volumes/exhale/MRI/OpenNeuro/Participants/"
output_dir="${base_directory}/GM_ROI_subcortical_Volumes/"
final_output="${base_directory}/GM_ROI_subcortical_volumes_bin.txt"

# Initialize the header for the final output file
header="Participant_ID"
for roi_id in {1..21}; do  # Replace 48 with the actual number of ROIs in your atlas
    header="${header},ROI_${roi_id}"
done
echo "$header" > "$final_output"

# Loop through each participant's ROI volume file
for roi_file in "${output_dir}"*_subcortical_Bin_Volumes.txt; do
    participant_id=$(basename "$roi_file" | cut -d'_' -f1)
    
    # Initialize a string to store the participant's ID and ROI volumes
    participant_data="$participant_id"

    # Loop through each ROI ID and extract the volume
    for roi_id in {1..21}; do  # Replace 48 with the actual number of ROIs in your atlas
        roi_volume=$(awk -F, -v roi="$roi_id" '$1 == roi {print $2}' "$roi_file")
        participant_data="${participant_data},${roi_volume}"
    done
    
    # Append the participant's data to the final output file
    echo "$participant_data" >> "$final_output"
done

echo "All participants' ROI volumes combined into $final_output."
