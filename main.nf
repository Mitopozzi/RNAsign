#!/usr/bin/env nextflow
nextflow.enable.dsl=2

log.info """
    RNAsign Processing Pipeline
    =========================
    Input BAMs        : ${params.input_bams}
    Output Dir        : ${params.output_dir}
    Mode              : ${params.mode}
    Quantify          : ${params.quantify}
    """
    .trim()

// --- Workflows Definition ---

// ─── Main Workflow ─────────────────────────────────────────────────────────────
workflow {


    // Run workflow based on starting point
    switch(params.mode) {
        case 'complete':
            // Full pipeline: bedtools → aggregation → filter → prediction → (optional) featureCounts
            bam_ch = Channel.fromPath(params.input_bams)
            
            bedtools_out = bedtools_wf(bam_ch)
            aggregation_out = aggregation_wf(bedtools_out.aggregated_input_ch)
            filter_out = filter_wf(aggregation_out.filter_input_ch)
            prediction_out = prediction_wf(filter_out.prediction_input_ch)
            
            if (params.run_featurecounts == true) {
                featurecounts_wf(bedtools_out.bam_ch, prediction_out.final_prediction_output)
            }
            break
        
        case 'partial':
            // Partial pipeline: filter → prediction → (optional) featureCounts
            // It needs a correctly formatted coverage file as input
            filter_input_ch = Channel.fromPath(params.aggregated_file)
                .map { file -> tuple(file.simpleName.replaceAll(/_aggregated$/, ''), file) }
            
            filter_out = filter_wf(filter_input_ch)
            prediction_out = prediction_wf(filter_out.prediction_input_ch)
            
            if (params.run_featurecounts == true) {
                bam_ch = Channel.fromPath(params.input_bams)
                    .map { bam -> tuple(bam.simpleName, bam) }
                featurecounts_wf(bam_ch, prediction_out.final_prediction_output)
            }
            break
    }
}

// ─── Sub-Workflows ─────────────────────────────────────────────────────────────

// ─── Bedtools Workflow ─────────────────────────────────────────────────────────
workflow bedtools_wf {
    take: input_bams
    main:
        bam_ch = input_bams.map { bam -> tuple(bam.simpleName, bam) }

        def bedtools_variants = [
            ['Total', ''],
            ['3prime', '-3'],
            ['5prime', '-5']
        ]
        bedtools_flags_ch = Channel.fromList(bedtools_variants)

        bam_ch
            .combine(bedtools_flags_ch)
            .set { bedtools_input_ch }

        run_bedtools(bedtools_input_ch)

        run_bedtools.out.results
            .groupTuple()  // [sample_id, [file1, file2, file3]]
            .map { sample_id, files ->
                // Sort files by name
                def sorted = files.sort { it.name }
                // sorted[0] = 3prime, sorted[1] = 5prime, sorted[2] = Total
                tuple(sample_id, sorted[2], sorted[0], sorted[1])  // Total, 3prime, 5prime
    }
    .set { aggregated_input_ch }

    emit:
        aggregated_input_ch
        bam_ch
}

// ─── Aggregation Workflow ──────────────────────────────────────────────────────
workflow aggregation_wf {
    take: aggregated_input_ch
    main:
        run_aggregation(aggregated_input_ch)
        filter_input_ch = run_aggregation.out.aggregated_file
    emit:
        filter_input_ch
}

// ─── Filtering Workflow ────────────────────────────────────────────────────────
workflow filter_wf {
    take:
        filter_input_ch
    main:
        run_sequence_filter(filter_input_ch)
        prediction_input_ch = run_sequence_filter.out.filtered_file
    emit:
        prediction_input_ch
}

// ─── Prediction Workflow ───────────────────────────────────────────────────────
workflow prediction_wf {
    take:
        prediction_input_ch
    main:
        run_prediction_script(prediction_input_ch)
        final_prediction_output = run_prediction_script.out.prediction_outputs
            .collect()
            .map { it.last() }
    emit:
        final_prediction_output
}

// ─── FeatureCounts Workflow ────────────────────────────────────────────────────
workflow featurecounts_wf {
    take:
        bam_ch
        final_prediction_output
    main:
        bam_ch.combine(final_prediction_output).set { featurecounts_input_ch }
        run_featurecounts(featurecounts_input_ch)
    emit:
        counts = run_featurecounts.out.counts
        summary = run_featurecounts.out.summary
}


// --- Process Definitions ---

process run_bedtools {
    publishDir "${params.output_dir}/bedtools/${sample_id}", mode: 'copy'

    input:
    tuple val(sample_id), path(bam_file), val(name), val(flag)

    output:
    // Use the descriptive 'name' for the output filename (e.g., sample1.Total.csv)
    tuple val(sample_id), path("${sample_id}_${name}.csv"), emit: results

    script:
    """
    echo "Running bedtools '${name}' on ${sample_id}"
    micromamba run -p /home/micromamba/RNAsign/envs/bedtools_env \
    bedtools genomecov -ibam ${bam_file} -dz ${flag} > ${sample_id}_${name}.csv
    """
}

process run_aggregation {
    publishDir "${params.output_dir}/aggregated/${sample_id}", mode: 'copy'

    input:
    tuple val(sample_id), path(total_file), path(prime3_file), path(prime5_file)

    output:
    tuple val(sample_id), path("${sample_id}_aggregated.csv"), emit: aggregated_file

    script:
    """
    bash ${baseDir}/bin/Aggregate.sh \
        ${total_file} \
        ${prime3_file} \
        ${prime5_file} \
        ${sample_id}_aggregated.csv
    """
}

process run_sequence_filter {
    publishDir "${params.output_dir}/filtered/${sample_id}", mode: 'copy'
    

    input:
    tuple val(sample_id), path(combinedfile)

    output:
    tuple val(sample_id), path("${sample_id}_filtered.csv"), emit: filtered_file

    script:
    """
    micromamba run -p /home/micromamba/RNAsign/envs/python_env_gpu \
    python ${baseDir}/bin/Filtering.py ${combinedfile} ${sample_id}_filtered.csv
    """
}

process run_prediction_script {
    publishDir "${params.output_dir}/predictions/${sample_id}", mode: 'copy'

    input:
    tuple val(sample_id), path(filtered_file)

    output:
    // Use a glob pattern to capture all potential outputs
    path "PRED_*", emit: prediction_outputs

    script:
    """
    micromamba run -p /home/micromamba/RNAsign/envs/python_env_gpu \
    python ${baseDir}/bin/Prediction.py -I ${filtered_file} -O . -M TM 
    # Rename outputs to match the glob pattern, ensuring a predictable order
    """
}

process run_featurecounts {
    publishDir "${params.output_dir}/featurecounts", mode: 'copy'

    input:
    tuple path(bam_file), path(saf_file)

    output:
    path "${bam_file.simpleName}_featurecounts.txt", emit: counts
    path "${bam_file.simpleName}_featurecounts.txt.summary", emit: summary

    script:
    """
    micromamba run -p /home/micromamba/RNAsign/envs/featurecounts_env \
    featureCounts -F SAF -a ${saf_file} -o ${bam_file.simpleName}_featurecounts.txt ${bam_file}
    """
}