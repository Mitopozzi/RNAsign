#!/usr/bin/env nextflow

// Check files with a glob pattern
Channel
    .fromPath("${baseDir}/tests/*.bam")
    .view { "Found file: $it" }

// Or check specific file types
// Channel
//     .fromPath("*.txt")
//     .view { "Found text file: $it" }

// Or check files in subdirectories
// Channel
//     .fromPath("**/*")
//     .view { "Found file: $it" }