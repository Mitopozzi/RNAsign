#!/bin/bash

# A script that only aggregates three pre-made bedtools coverage files.
# Usage: ./Aggregate.sh <total_cov.csv> <prime3_cov.csv> <prime5_cov.csv> <output_file.csv>

set -e # Exit immediately if a command exits with a non-zero status.

if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <total_cov_file> <prime3_cov_file> <prime5_cov_file> <output_file>"
    exit 1
fi

# Assign arguments to variables
total_file="$1"
prime3_file="$2"
prime5_file="$3"
output_file="$4"

echo "Aggregating the following files:"
echo "  - Total: ${total_file}"
echo "  - 3-prime: ${prime3_file}"
echo "  - 5-prime: ${prime5_file}"

# Use awk to join the three files based on sequence name and position.
# ARGIND is an awk variable that tracks which file is being processed.
awk '
BEGIN {
# Set output field separator to whitespace
    OFS = " " 
}
# Process the 1st file (total coverage)
ARGIND == 1 {
    key = $1 SUBSEP $2; total[key] = $3; positions[key] = 1; next
}
# Process the 2nd file (3-prime coverage)
ARGIND == 2 {
    key = $1 SUBSEP $2; prime3[key] = $3; positions[key] = 1; next
}
# Process the 3rd file (5-prime coverage)
ARGIND == 3 {
    key = $1 SUBSEP $2; prime5[key] = $3; positions[key] = 1
}
END {
    # Add a header to the output
    print "RegionID,position,coverage,3prime,5prime"
    
    # Iterate through all unique positions and print the combined data
    for (pos in positions) {
        split(pos, coords, SUBSEP)
        seq = coords[1]; position = coords[2]
        
        # Default to 0 if a value is missing for a given position
        total_cov = (pos in total) ? total[pos] : 0
        prime3_cov = (pos in prime3) ? prime3[pos] : 0
        prime5_cov = (pos in prime5) ? prime5_cov : 0
        
        print seq, position, total_cov, prime3_cov, prime5_cov
    }
}' "$total_file" "$prime3_file" "$prime5_file" | sort -t, -k1,1 -k2,2n > "$output_file"

echo "Aggregated file saved to: $output_file"