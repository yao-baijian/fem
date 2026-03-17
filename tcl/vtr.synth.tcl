
set benchmarks {
    mkDelayWorker32B /home/byao/Desktop/fem_rev/fem/benchmarks/vtr/verilog/mkDelayWorker32B.v
    mkPktMerge /home/byao/Desktop/fem_rev/fem/benchmarks/vtr/verilog/mkPktMerge.v
    mkSMAdapter4B /home/byao/Desktop/fem_rev/fem/benchmarks/vtr/verilog/mkSMAdapter4B.v
    or1200_flat /home/byao/Desktop/fem_rev/fem/benchmarks/vtr/verilog/or1200.v
    paj_raygentop_hierarchy_no_mem /home/byao/Desktop/fem_rev/fem/benchmarks/vtr/verilog/raygentop.v
    sv_chip0_hierarchy_no_mem /home/byao/Desktop/fem_rev/fem/benchmarks/vtr/verilog/stereovision0.v
    sv_chip1_hierarchy_no_mem /home/byao/Desktop/fem_rev/fem/benchmarks/vtr/verilog/stereovision1.v
    sv_chip2_hierarchy_no_mem /home/byao/Desktop/fem_rev/fem/benchmarks/vtr/verilog/stereovision2.v
    sv_chip3_hierarchy_no_mem /home/byao/Desktop/fem_rev/fem/benchmarks/vtr/verilog/stereovision3.v
}

# bgm /home/byao/Desktop/fem_rev/fem/benchmarks/vtr/verilog/bgm.v
# RLE_BlobMerging /home/byao/Desktop/fem_rev/fem/benchmarks/vtr/verilog/blob_merge.v
# paj_boundtop_hierarchy_no_mem /home/byao/Desktop/fem_rev/fem/benchmarks/vtr/verilog/boundtop.v
# memory_controller /home/byao/Desktop/fem_rev/fem/benchmarks/vtr/verilog/ch_intrinsics.v
# diffeq_paj_convert /home/byao/Desktop/fem_rev/fem/benchmarks/vtr/verilog/diffeq1.v
# diffeq_f_systemC /home/byao/Desktop/fem_rev/fem/benchmarks/vtr/verilog/diffeq2.v
# LU8PEEng /home/byao/Desktop/fem_rev/fem/benchmarks/vtr/verilog/LU8PEEng.v
# LU32PEEng /home/byao/Desktop/fem_rev/fem/benchmarks/vtr/verilog/LU32PEEng.v
# mcml /home/byao/Desktop/fem_rev/fem/benchmarks/vtr/verilog/mcml.v
# mkDelayWorker32B /home/byao/Desktop/fem_rev/fem/benchmarks/vtr/verilog/mkDelayWorker32B.v
# mkPktMerge /home/byao/Desktop/fem_rev/fem/benchmarks/vtr/verilog/mkPktMerge.v
# mkSMAdapter4B /home/byao/Desktop/fem_rev/fem/benchmarks/vtr/verilog/mkSMAdapter4B.v
# or1200_flat /home/byao/Desktop/fem_rev/fem/benchmarks/vtr/verilog/or1200.v
# paj_raygentop_hierarchy_no_mem /home/byao/Desktop/fem_rev/fem/benchmarks/vtr/verilog/raygentop.v
# sha1 /home/byao/Desktop/fem_rev/fem/benchmarks/vtr/verilog/sha.v
# sv_chip0_hierarchy_no_mem /home/byao/Desktop/fem_rev/fem/benchmarks/vtr/verilog/stereovision0.v
# sv_chip1_hierarchy_no_mem /home/byao/Desktop/fem_rev/fem/benchmarks/vtr/verilog/stereovision1.v
# sv_chip2_hierarchy_no_mem /home/byao/Desktop/fem_rev/fem/benchmarks/vtr/verilog/stereovision2.v
# sv_chip3_hierarchy_no_mem /home/byao/Desktop/fem_rev/fem/benchmarks/vtr/verilog/stereovision3.v

set part_name {xcvu065-ffvc1517-1-i}
set base_output_dir {output_dir}

# Process each benchmark
foreach {top_module rtl_file} $benchmarks {
    puts "========================================"
    puts "Processing benchmark: $top_module"
    puts "========================================"
    
    # Create output directory
    set output_dir [file join $base_output_dir $top_module]
    puts "Creating output directory: $output_dir"
    file mkdir $output_dir
    
    # Create temp project directory
    set temp_project_dir [file join ./temp_projects $top_module]
    file mkdir $temp_project_dir
    
    # Create project
    create_project -part $part_name -force $top_module $temp_project_dir
    add_files -norecurse $rtl_file /home/byao/Desktop/fem_rev/fem/benchmarks/vtr/verilog/vtr_primitives.v
    set_property top $top_module [current_fileset]
    
    # Synthesis
    puts "Running synthesis for $top_module..."
    synth_design -top $top_module -part $part_name -flatten_hierarchy rebuilt -mode out_of_context
    write_checkpoint -force [file join $output_dir post_synth.dcp]
    
    # Optimization
    puts "Running optimization for $top_module..."
    opt_design
    
    # Placement
    puts "Running placement for $top_module..."
    place_design
    
    # Routing
    puts "Running routing for $top_module..."
    route_design
    write_checkpoint -force [file join $output_dir post_impl.dcp]
    
    # Reports
    puts "Generating reports for $top_module..."
    report_timing_summary -file [file join $output_dir timing_summary.rpt] -delay_type min_max
    report_design_analysis -file [file join $output_dir placement_analysis.rpt] -name placement_analysis
    report_route_status -file [file join $output_dir route_status.rpt]
    
    puts "Completed processing $top_module"
    puts ""
    
    # Close project
    close_project
}
