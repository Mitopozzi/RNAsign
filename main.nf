#!/usr/bin/env nextflow
nextflow.enable.dsl=2

log.info """
    RNAsign Processing Pipeline
    =========================
    Input BAMs        : ${params.input_bams}
    Output Dir        : ${params.output_dir}
    Start From        : ${params.start_from}
    Stop At           : ${params.stop_at}
    Run featureCounts : ${params.run_featurecounts}
    """
    .trim()

// --- Workflows Definition ---
// ─── Bedtools Workflow ─────────────────────────────────────────────────────────
workflow bedtools_wf {
    take: input_bams
    main:
        bam_ch = input_bams.map { bam -> tuple(bam.simpleName, bam) }
        bam_ch.view { "BAM: $it[0]" }

        def bedtools_variants = [
            ['Total', ''],
            ['3prime', '-3'],
            ['5prime', '-5']
        ]
        bedtools_flags_ch = Channel.fromList(bedtools_variants)

        bam_ch
            .combine(bedtools_flags_ch)
            .map { id_path, flag_list -> tuple(id_path, flag_list) }
            .set { bedtools_input_ch }

        run_bedtools(bedtools_input_ch)

        aggregated_input_ch = run_bedtools.out.results.groupTuple()
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
        python_env
    main:
        run_sequence_filter(filter_input_ch, python_env)
        prediction_input_ch = run_sequence_filter.out.filtered_file
    emit:
        prediction_input_ch
}

// ─── Prediction Workflow ───────────────────────────────────────────────────────
workflow prediction_wf {
    take:
        prediction_input_ch
        python_env
    main:
        run_prediction_script(prediction_input_ch, python_env)
        final_prediction_output = run_prediction_script.out.prediction_outputs.last()
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
        run_featurecounts.out
}

// ─── Main Workflow ─────────────────────────────────────────────────────────────
workflow {
    def nametest = "${params.input_bams}"
    println ">>> Scanning for BAM files in: ${nametest}"
    Channel
    .fromPath(nametest)
    .view { "Found BAM file: $it" }

    // 0. GPU availability
    platform_ch = check_gpu()
    def python_env = platform_ch.view().trim() == 'gpu' ? 'python_env_gpu' : 'python_env_cpu'

    // Entry BAM channel
    bam_ch = Channel.fromPath(params.input_bams)

    // Decide starting point
    def aggregated_input_ch
    def filter_input_ch
    def prediction_input_ch
    def final_prediction_output

    switch(params.start_from) {
        case 'bedtools':
            bedtools_out = bedtools_wf(bam_ch)
            aggregated_input_ch = bedtools_out.aggregated_input_ch
            bam_ch = bedtools_out.bam_ch
            if (params.stop_at == 'bedtools') return
            // fallthrough
        case 'aggregation':
            if (!aggregated_input_ch && params.bedtools_results)
                aggregated_input_ch = Channel.fromPath(params.bedtools_results)
            aggregation_out = aggregation_wf(aggregated_input_ch)
            filter_input_ch = aggregation_out.filter_input_ch
            if (params.stop_at == 'aggregation') return
            // fallthrough
        case 'filter':
            filter_out = filter_wf(filter_input_ch, python_env)
            prediction_input_ch = filter_out.prediction_input_ch
            if (params.stop_at == 'filter') return
            // fallthrough
        case ['prediction', 'python']:
            prediction_out = prediction_wf(prediction_input_ch, python_env)
            final_prediction_output = prediction_out.final_prediction_output
            if (params.stop_at == 'prediction') return
            // fallthrough
        case 'featurecounts':
            if (!final_prediction_output && params.prediction_output)
                final_prediction_output = Channel.fromPath(params.prediction_output)
            if (params.run_featurecounts)
                featurecounts_wf(bam_ch, final_prediction_output)
            return
    }
}


// --- Process Definitions ---
process check_gpu {
    // This process can run in the base container; no special conda env is needed.
    
    output:
    stdout emit: platform_ch // Output the result to a channel

    script:
    """
    nvidia-smi &> /dev/null && echo 'gpu' || echo 'cpu'
    """
}

process run_bedtools {
    publishDir "${params.output_dir}/bedtools/${sample_id}", mode: 'copy'

    // Conda environment for bedtools from the Docker image
    conda 'bedtools_env'

    input:
    tuple(val(sample_id), path(bam_file)), tuple(val(name), val(flag))

    output:
    // Use the descriptive 'name' for the output filename (e.g., sample1.Total.csv)
    val(sample_id), path("${sample_id}.${name}.csv"), emit: results

    script:
    """
    echo "Running bedtools '${name}' on ${sample_id}"
    bedtools genomecov -ibam ${bam_file} -dz ${flag} > ${sample_id}.${name}.csv
    """
}

process run_aggregation {
    publishDir "${params.output_dir}/aggregated/${sample_id}", mode: 'copy'

    input:
    val(sample_id), path(results) // 

    output:
    tuple val(sample_id), path("${sample_id}_aggregated.csv"), emit: aggregated_file

    script:
    """
    # NOTE: It's critical that the files are in the correct order for the script.
    # We then pass them to the script in the order it expects: Total, 3prime, 5prime.
    # Sorting the files might not be necessary but it is good to ensure consistency.
    
    sorted_files=( \$(echo "${results}" | tr ' ' '\\n' | sort) )

    Aggregate.sh \
        \${sorted_files[2]} \
        \${sorted_files[0]} \
        \${sorted_files[1]} \
        ${sample_id}.aggregated.csv
    """
}

process run_sequence_filter {
    publishDir "${params.output_dir}/filtered/${sample_id}", mode: 'copy'
    conda python_env

    input:
    val(sample_id), path(aggregated_file)

    output:
    tuple val(sample_id), path("${sample_id}_filtered.csv"), emit: filtered_file

    script:
    """
    python ${baseDir}/bin/Chunking.py ${aggregated_file} ${sample_id}_filtered.csv
    """
}

process run_prediction_script {
    publishDir "${params.output_dir}/prediction_predictions/${sample_id}", mode: 'copy'
    conda python_env

    input:
    val(sample_id), path(filtered_file)

    output:
    // Use a glob pattern to capture all potential outputs
    path "PRED_*", emit: prediction_outputs

    script:
    """
    Prediction.py -I ${filtered_file} -O . -M TM 
    # Rename outputs to match the glob pattern, ensuring a predictable order
    """
}

process run_featurecounts {
    publishDir "${params.output_dir}/featurecounts", mode: 'copy'

    // Conda environment for featureCounts from the Docker image
    conda 'featurecounts_env'

    input:
    path(bam_file), path(saf_file)

    output:
    path "${bam_file.simpleName}_featurecounts.txt", emit: counts
    path "${bam_file.simpleName}_featurecounts.txt.summary", emit: summary

    script:
    """
    featureCounts -F SAF -a ${saf_file} -o ${bam_file.simpleName}_featurecounts.txt ${bam_file}
    """
}