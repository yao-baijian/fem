# Usage: cd vivado; vivado -mode batch -source ../tcl/run_boundary_verify_dcp.tcl -tclargs <top_module> <dcp_file> [clock_region] [part_name]
if { $argc < 2 } {
    puts "Error: Usage: cd vivado; vivado -mode batch -source ../tcl/run_boundary_verify_dcp.tcl -tclargs <top_module> <dcp_file> [clock_region] [part_name]"
    exit 1
}

set top_module [lindex $argv 0]
set dcp_file [lindex $argv 1]

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
puts "  - DCP File: $dcp_file"
puts "  - Clock Region: $clock_region"
puts "  - Part Name:    $part_name"
puts "  - Output Dir:   $output_dir"

# 1. Open DCP and generate stub
puts "Extracting interface from DCP..."
create_project -in_memory -part $part_name
open_checkpoint $dcp_file
set stub_file "${output_dir}/${top_module}_stub.v"
write_verilog -mode synth_stub -force $stub_file
close_project

# 2.2 Find the actual module name from the stub
set module_name "${top_module}"
set fp [open $stub_file r]
while { [gets $fp line] >= 0 } {
    if { [regexp {^module\s+([A-Za-z0-9_]+)} $line match name] } {
        set module_name $name
        break
    }
}
close $fp

# 2. Generate Wrapper
puts "Generating wrapper for ${module_name}..."
set wrapper_file "${output_dir}/${top_module}_wrapper.v"
set python_cmd "python3 ../scripts/gen_boundary_wrapper.py $stub_file $wrapper_file $module_name $clock_region"
if { [catch {exec {*}$python_cmd} msg] } {
    puts "Error running python script: $msg"
    exit 1
}

# 3. Setup Project for final implementation
create_project -in_memory -part $part_name

read_verilog $wrapper_file
add_files $dcp_file

# 4. Synthesize
set wrapper_top "${module_name}_wrapper"
synth_design -top $wrapper_top -part $part_name -flatten_hierarchy rebuilt

# 5. Define Clock
# Usually clk pin is named clk, but the stub has clk1. We need to find the clock port of wrapper
# Assuming the wrapper has a 'clk' port created by the script.
create_clock -period 5.0 -name clk [get_ports clk]

# 6. Place IO Registers on Boundary
source ../tcl/place_boundary_io.tcl
place_io_registers $clock_region "${top_module}_boundary"

# 7. Implementation Flow
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

# 8. Reports
report_timing_summary -file ${output_dir}/timing_summary.rpt
report_utilization -file ${output_dir}/utilization.rpt
write_checkpoint -force ${output_dir}/post_impl.dcp

puts "Done! Results in $output_dir"
quit
