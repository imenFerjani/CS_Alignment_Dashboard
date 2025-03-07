#!/bin/bash

# Create necessary directories
mkdir -p data

# Create empty CSV files if they don't exist
touch data/cs_course_outcomes.csv
touch data/cs_job_postings_500.csv
touch data/cs2023_standards.csv
touch data/csta_standards.csv
touch data/abet_standards.csv
touch data/global_cs_standards.csv

# Create a simple Header_img.png placeholder if it doesn't exist
if [ ! -f "Header_img.png" ]; then
    echo "Creating placeholder header image..."
    # This creates a very basic colored rectangle as a placeholder
    # Install ImageMagick if needed on your system to use this
    if command -v convert >/dev/null 2>&1; then
        convert -size 800x150 gradient:#1A2A44-#2E4066 -gravity center -pointsize 36 -fill "#F4E3B2" -annotate 0 "IAU Computer Science" Header_img.png
    fi
fi

echo "Setup complete!"