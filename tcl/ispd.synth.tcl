set synth_dcp "C:\\Project\\fem\\vivado\\output_dir\\FPGA-example2\\design.dcp"  ;# 综合后的DCP
set output_dir "C:\\Project\\fem\\vivado\\output_dir\\FPGA-example2"                  ;# 输出目录
set impl_dcp [file join $output_dir "post_impl.dcp"]                                  ;# 实现后的DCP

open_checkpoint $synth_dcp

place_design
route_design

write_checkpoint -force $impl_dcp
set edif_file [file join $output_dir "post_impl.edf"]
write_edif -force $edif_file

report_route_status

# report_timing_summary -file [file join $output_dir "timing_summary.rpt"] -delay_type min_max
# report_timing -max_paths 10 -file [file join $output_dir "timing_paths.rpt"]
# report_utilization -file [file join $output_dir "utilization.rpt"] -hierarchical
# report_design_analysis -file [file join $output_dir "design_analysis.rpt"]
# report_route_status -file [file join $output_dir "route_status.rpt"]
# report_power -file [file join $output_dir "power.rpt"]