#!/bin/bash

set -e

if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <total_cov_file> <prime3_cov_file> <prime5_cov_file> <output_file>"
    exit 1
fi

total_file="$1"
prime3_file="$2"
prime5_file="$3"
output_file="$4"

echo "Aggregating the following files:"
echo "  - Total: ${total_file}"
echo "  - 3-prime: ${prime3_file}"
echo "  - 5-prime: ${prime5_file}"

# Three-pass awk (more portable)
awk -v total_file="$total_file" -v prime3_file="$prime3_file" -v prime5_file="$prime5_file" '
BEGIN {
    FS="\t"
    OFS = ","
    
    # Read total coverage
    while ((getline < total_file) > 0) {
        key = $1 SUBSEP $2
        total[key] = $3
        positions[key] = 1
    }
    close(total_file)
    
    # Read 3-prime coverage
    while ((getline < prime3_file) > 0) {
        key = $1 SUBSEP $2
        prime3[key] = $3
        positions[key] = 1
    }
    close(prime3_file)
    
    # Read 5-prime coverage
    while ((getline < prime5_file) > 0) {
        key = $1 SUBSEP $2
        prime5[key] = $3
        positions[key] = 1
    }
    close(prime5_file)
    
    # Output header
    print "RegionID,position,coverage,3prime,5prime"
    
    # Output data
    for (pos in positions) {
        split(pos, coords, SUBSEP)
        seq = coords[1]
        position = coords[2]
        
        total_cov = (pos in total) ? total[pos] : 0
        prime3_cov = (pos in prime3) ? prime3[pos] : 0
        prime5_cov = (pos in prime5) ? prime5[pos] : 0
        
        print seq, position, total_cov, prime3_cov, prime5_cov
    }
}' | sort -t, -k1,1 -k2,2n > "$output_file"

echo "Aggregated file saved to: $output_file"