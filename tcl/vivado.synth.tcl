# 设置项目参数
# set rtl_file /home/byao/Desktop/Benchmarks/ISCAS85/c1355/c1355.v
# set top_module {c1355}
# set part_name {xcvu065-ffvc1517-1-i}
# set output_dir {output_dir}

# set rtl_file /home/byao/Desktop/Benchmarks/ISCAS85/c1908/c1908.v
# set top_module {c1908}
# set part_name {xcvu065-ffvc1517-1-i}
# set output_dir {output_dir}

set rtl_file /home/byao/Desktop/Benchmarks/ISCAS89/s15850.v
set top_module {s15850}
set part_name {xcvu065-ffvc1517-1-i}
set output_dir {output_dir}

create_project -part $part_name -force temp_project ./temp_project
add_files -norecurse $rtl_file
set_property top $top_module [current_fileset]
synth_design -top $top_module -part $part_name -flatten_hierarchy rebuilt
write_checkpoint -force $output_dir/post_synth.dcp
opt_design
place_design
route_design
write_checkpoint -force $output_dir/post_impl.dcp


report_timing_summary -file $output_dir/timing_summary.rpt -delay_type min_max
report_design_analysis -file $output_dir/placement_analysis.rpt -name placement_analysis
report_route_status -file $output_dir/route_status.rpt
# report_timing_violations -file $output_dir/timing_violations.rpt
# report_clock_utilization -file $output_dir/clock_utilization.rpt
# report_design_analysis -wire_length -file $output_dir/wire_length_analysis.rpt
# report_power -file $output_dir/power_analysis.rpt

# report_timing -max_paths 100 -file $output_dir/timing_max_paths.rpt -delay_type max
# report_timing -max_paths 100 -file $output_dir/timing_min_paths.rpt -delay_type min

# foreach clock [get_clocks] {
#     set clock_name [get_property NAME $clock]
#     report_timing -max_paths 50 -file $output_dir/timing_${clock_name}.rpt -delay_type min_max -name timing_${clock_name}
# }

# report_timing -of_objects [get_timing_paths -max_paths 10 -nworst 1] -file $output_dir/critical_paths.rpt

# set wns [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1]]
# puts "WNS: $wns"

# set tns [get_property SLACK [get_timing_paths -nworst 1000]]
# puts "TNS: $tns"

# set total_wirelength [get_property CONFIG.WIRE_LENGTH [get_design]]
# puts "Total Wire Length: $total_wirelength"



# set summary_file [open "$output_dir/summary.rpt" w]

# puts $summary_file "=== DESIGN IMPLEMENTATION SUMMARY ==="
# puts $summary_file "Design: $top_module"
# puts $summary_file "Part: $part_name"
# puts $summary_file "Timestamp: [clock format [clock seconds]]"

# # 获取时序信息
# set timing_paths [get_timing_paths -max_paths 1 -nworst 1]
# if {[llength $timing_paths] > 0} {
#     set wns [get_property SLACK [lindex $timing_paths 0]]
#     set tns 0
#     foreach path [get_timing_paths -nworst 1000] {
#         set slack [get_property SLACK $path]
#         if {$slack < 0} {
#             set tns [expr $tns + $slack]
#         }
#     }
    
#     puts $summary_file "WNS (Worst Negative Slack): $wns ns"
#     puts $summary_file "TNS (Total Negative Slack): $tns ns"
# } else {
#     puts $summary_file "WNS: N/A"
#     puts $summary_file "TNS: N/A"
# }

# # 获取资源利用率
# set lut_util [get_property LUT [get_utilization]]
# set ff_util [get_property FF [get_utilization]]
# set dsp_util [get_property DSP [get_utilization]]
# set bram_util [get_property BRAM [get_utilization]]

# puts $summary_file "LUT Utilization: $lut_util"
# puts $summary_file "FF Utilization: $ff_util"
# puts $summary_file "DSP Utilization: $dsp_util"
# puts $summary_file "BRAM Utilization: $bram_util"

# # 布线状态
# set route_status [get_property STATUS [get_drc_checks]]
# puts $summary_file "Route Status: $route_status"

# close $summary_file

# # 显示关键信息
# puts "=========================================="
# puts "Implementation Complete!"
# puts "Output directory: $output_dir"
# puts "Key reports generated:"
# puts "  - Timing summary: $output_dir/final_timing_summary.rpt"
# puts "  - Critical paths: $output_dir/critical_paths.rpt"
# puts "  - Utilization: $output_dir/utilization.rpt"
# puts "  - Wire length analysis: $output_dir/wire_length_analysis.rpt"
# puts "  - Design summary: $output_dir/summary.rpt"
# puts "=========================================="

# 关闭项目
close_project