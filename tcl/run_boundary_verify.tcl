# Usage: cd vivado; vivado -mode batch -source ../tcl/run_boundary_verify.tcl -tclargs <top_module> <verilog_file> [clock_region] [part_name]
# Example: cd vivado; vivado -mode batch -source ../tcl/run_boundary_verify.tcl -tclargs c1355 ../benchmarks/ISCAS85/c1355/c1355.v X2Y2 xcvu065-ffvc1517-1-i

if { $argc < 2 } {
    puts "Error: Usage: cd vivado; vivado -mode batch -source ../tcl/run_boundary_verify.tcl -tclargs <top_module> <verilog_file> [clock_region] [part_name]"
    exit 1
}

set top_module [lindex $argv 0]
set verilog_file [lindex $argv 1]

if { $argc > 2 } {
    set clock_region [lindex $argv 2]
} else {
    set clock_region "X2Y2"
}

if { $argc > 3 } {
    set part_name [lindex $argv 3]
} else {
    set part_name "xcvu065-ffvc1517-1-i"
}

set output_dir "output_dir/${top_module}_boundary"
file mkdir $output_dir

puts "Starting verification for $top_module"
puts "  - Verilog File: $verilog_file"
puts "  - Clock Region: $clock_region"
puts "  - Part Name:    $part_name"
puts "  - Output Dir:   $output_dir"

# 1. Generate Wrapper
puts "Generating ${top_module} wrapper..."
set wrapper_file "${output_dir}/${top_module}_wrapper.v"
set python_cmd "python3 ../scripts/gen_boundary_wrapper.py $verilog_file $wrapper_file $top_module $clock_region"
if { [catch {exec {*}$python_cmd} msg] } {
    puts "Error running python script: $msg"
    exit 1
}

# 2. Setup Project
create_project -in_memory -part $part_name

read_verilog $verilog_file
read_verilog $wrapper_file

# Check if this is a VTR benchmark and include primitives if so
if { [string match "*vtr*" $verilog_file] } {
    puts "VTR benchmark detected, reading vtr_primitives.v..."
    read_verilog "/home/byao/Desktop/fem_rev/fem/benchmarks/vtr/verilog/vtr_primitives.v"
}

# 3. Synthesize
set wrapper_top "${top_module}_wrapper"
synth_design -top $wrapper_top -part $part_name -flatten_hierarchy rebuilt

# 4. Define Clock
create_clock -period 5.0 -name clk [get_ports clk]

# 5. Place IO Registers on Boundary
source ../tcl/place_boundary_io.tcl
place_io_registers $clock_region "${top_module}_boundary"

# 6. Implementation Flow
opt_design

set place_start [clock seconds]
place_design
set place_end [clock seconds]
set place_time [expr {$place_end - $place_start}]

# Write placement time to a file for Python script to read
set fp [open "${output_dir}/place_time.txt" w]
puts $fp $place_time
close $fp

route_design

# 7. Reports
report_timing_summary -file ${output_dir}/timing_summary.rpt
report_utilization -file ${output_dir}/utilization.rpt
write_checkpoint -force ${output_dir}/post_impl.dcp

puts "Done! Results in $output_dir"
quit
