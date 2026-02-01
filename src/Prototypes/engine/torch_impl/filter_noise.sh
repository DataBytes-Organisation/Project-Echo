#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# The directory containing the metadata file
META_DIR="ESC-50-master/meta"
# The directory containing the audio files
AUDIO_DIR="ESC-50-master/audio"
# The target directory for the filtered files
OUTPUT_DIR="background_noise"
# The CSV file with metadata
CSV_FILE="$META_DIR/esc50.csv"

# The categories to filter for
declare -a CATEGORIES=("rain" "sea_waves" "crickets" "wind" "thunderstorm")

# Create the output directory if it doesn't exist
echo "Creating output directory: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

# Function to process a single category
process_category() {
    local category=$1
    echo "Processing category: $category"
    
    # Use awk to find filenames for the given category
    # We skip the header row (NR>1) and set the field separator to a comma.
    # If the 4th field ($4) matches the category, we print the first field ($1), which is the filename.
    awk -F, -v cat="$category" 'NR>1 && $4==cat {print $1}' "$CSV_FILE" | while IFS= read -r filename; do
        # For each filename found, copy the file from the audio directory to the output directory
        if [ -f "$AUDIO_DIR/$filename" ]; then
            echo "Copying $filename to $OUTPUT_DIR"
            cp "$AUDIO_DIR/$filename" "$OUTPUT_DIR/"
        else
            echo "Warning: File not found: $AUDIO_DIR/$filename"
        fi
    done
}

# Loop through the array of categories and process each one
for category in "${CATEGORIES[@]}"; do
    process_category "$category"
done

echo "Script finished."
echo "All files from the specified categories have been copied to $OUTPUT_DIR."

# List the contents of the new directory
ls -l "$OUTPUT_DIR" | head
