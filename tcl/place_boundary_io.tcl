# Usage: source tcl/place_boundary_io.tcl
# Must be run after synthesis/link_design

proc place_io_registers {base_clock_region {top_module "default"}} {
    puts "Estimating required area starting from clock region: $base_clock_region"
    
    # 1. Get IO Registers and Logic Cells
    set io_regs [get_cells -hierarchical -filter {REF_NAME == FDRE && NAME =~ "*u_io_reg_*"}]
    set io_regs [lsort $io_regs]

    if {[llength $io_regs] == 0} {
        puts "Warning: No IO registers found matching pattern *u_io_reg_*"
        return
    }

    set logic_cells [get_cells -hierarchical -filter {IS_PRIMITIVE == 1 && NAME !~ "*u_io_reg_*" && PRIMITIVE_TYPE =~ "FLIPFLOP.*|LUT.*|CARRY.*|MUX.*|SHIFT.*"}]
    # Divide roughly by 8 (assuming 8 LUTs/FFs per SLICE on average) to estimate required SLICE sites
    set num_logic [llength $logic_cells]
    set required_sites [expr {($num_logic + 7) / 8}]
    
    # Start from the base clock region and, when possible, include
    # one additional clock region vertically so the boundary spans
    # at least two clock regions in height (e.g. ~120 slice rows).
    regexp {X(\d+)Y(\d+)} $base_clock_region match cr_x cr_y
    set cr_list {}
    lappend cr_list $base_clock_region

    # Prefer the region just above (higher Y). If that doesn't exist,
    # fall back to the region just below.
    set added_cr 0
    set next_y [expr {$cr_y + 1}]
    set next_cr "X${cr_x}Y${next_y}"
    set cr_obj [get_clock_regions -quiet $next_cr]
    if {[llength $cr_obj] == 0} {
        set prev_y [expr {$cr_y - 1}]
        set prev_cr "X${cr_x}Y${prev_y}"
        set cr_obj_prev [get_clock_regions -quiet $prev_cr]
        if {[llength $cr_obj_prev] != 0} {
            lappend cr_list $prev_cr
            set added_cr 1
        }
    } else {
        lappend cr_list $next_cr
        set added_cr 1
    }

    if {!$added_cr} {
        puts "Warning: Could not find adjacent clock region for $base_clock_region; using single region."
    }

    # Gather slice sites for the current list of clock regions
    set sites [get_sites -of_objects [get_clock_regions $cr_list] -filter {SITE_TYPE =~ "SLICE*"}]

    # If logic still does not fit, keep expanding further in Y
    # direction starting from the maximum Y used so far.
    set max_cr_y $cr_y
    foreach cr $cr_list {
        if {[regexp {X(\d+)Y(\d+)} $cr -> _x _y]} {
            if {$_y > $max_cr_y} {
                set max_cr_y $_y
            }
        }
    }
    set current_cr_y $max_cr_y

    while {[llength $sites] < $required_sites} {
        incr current_cr_y
        set next_cr "X${cr_x}Y${current_cr_y}"
        set cr_obj [get_clock_regions -quiet $next_cr]
        if {[llength $cr_obj] == 0} {
            puts "Warning: Reached edge of device, cannot expand to $next_cr"
            break
        }
        lappend cr_list $next_cr
        set sites [get_sites -of_objects [get_clock_regions $cr_list] -filter {SITE_TYPE =~ "SLICE*"}]
        puts "Expanded clock regions to $cr_list, total SLICE sites: [llength $sites]"
    }
    
    puts "Placing IO registers on boundary of clock regions: $cr_list"

    if {[llength $sites] == 0} {
        puts "Error: No slice sites found in clock region $base_clock_region"
        return
    }

    # 2. Find bounding box
    set x_coords {}
    set y_coords {}
    array set site_map {}

    foreach s $sites {
        set s_name [get_property NAME $s]
        # Parse logic location, e.g., SLICE_X12Y50
        if {[regexp {SLICE_X(\d+)Y(\d+)} $s_name match x y]} {
            lappend x_coords $x
            lappend y_coords $y
            set site_map($x,$y) $s
        }
    }

    set x_coords [lsort -unique -integer $x_coords]
    set y_coords [lsort -unique -integer $y_coords]

    set min_x [lindex $x_coords 0]
    set max_x [lindex $x_coords end]
    set min_y [lindex $y_coords 0]
    set max_y [lindex $y_coords end]

    puts "Clock Regions $cr_list Bounding Box: X\[$min_x:$max_x\] Y\[$min_y:$max_y\]"

    # 3. Collect boundary sites in order (Bottom -> Right -> Top -> Left)
    set boundary_sites {}
    set ring 0
    set collected_sites 0
    set required_sites [llength $io_regs]

    while {$collected_sites < $required_sites} {
        set cur_min_x [lindex $x_coords $ring]
        set cur_max_x [lindex $x_coords end-$ring]
        set cur_min_y [lindex $y_coords $ring]
        set cur_max_y [lindex $y_coords end-$ring]

        if {$cur_min_x == "" || $cur_max_x == "" || $cur_min_x > $cur_max_x || $cur_min_y > $cur_max_y} {
            break
        }

        # Bottom Edge (y = cur_min_y, x increases)
        foreach x $x_coords {
            if {$x >= $cur_min_x && $x <= $cur_max_x} {
                if {[info exists site_map($x,$cur_min_y)]} {
                    lappend boundary_sites $site_map($x,$cur_min_y)
                    incr collected_sites
                }
            }
        }

        # Right Edge (x = cur_max_x, y increases, exclude corners)
        foreach y $y_coords {
            if {$y > $cur_min_y && $y < $cur_max_y} {
                if {[info exists site_map($cur_max_x,$y)]} {
                    lappend boundary_sites $site_map($cur_max_x,$y)
                    incr collected_sites
                }
            }
        }

        # Top Edge (y = cur_max_y, x decreases)
        if {$cur_max_y > $cur_min_y} {
            set x_coords_rev [lsort -decreasing -integer $x_coords]
            foreach x $x_coords_rev {
                if {$x >= $cur_min_x && $x <= $cur_max_x} {
                    if {[info exists site_map($x,$cur_max_y)]} {
                        lappend boundary_sites $site_map($x,$cur_max_y)
                        incr collected_sites
                    }
                }
            }
        }

        # Left Edge (x = cur_min_x, y decreases, exclude corners)
        if {$cur_max_x > $cur_min_x} {
            set y_coords_rev [lsort -decreasing -integer $y_coords]
            foreach y $y_coords_rev {
                if {$y > $cur_min_y && $y < $cur_max_y} {
                    if {[info exists site_map($cur_min_x,$y)]} {
                        lappend boundary_sites $site_map($cur_min_x,$y)
                        incr collected_sites
                    }
                }
            }
        }

        incr ring
    }
    
    puts "Found [llength $boundary_sites] boundary slice sites in $ring rings."

    if {[llength $io_regs] > [llength $boundary_sites]} {
        puts "Warning: More IO registers ([llength $io_regs]) than boundary sites ([llength $boundary_sites]). Some will be unplaced."
    }

    # Create PBLOCK for boundary sites to prevent logic from mixing into it
    create_pblock pblock_boundary
    resize_pblock [get_pblocks pblock_boundary] -add $boundary_sites
    add_cells_to_pblock [get_pblocks pblock_boundary] $io_regs
    set_property EXCLUDE_PLACEMENT 1 [get_pblocks pblock_boundary]

    # Create PBLOCK for core logic to keep it strictly within the allocated clock regions
    create_pblock pblock_core
    foreach cr $cr_list {
        resize_pblock [get_pblocks pblock_core] -add CLOCKREGION_${cr}:CLOCKREGION_${cr}
    }
    
    set core_cells [get_cells -hierarchical -filter {IS_PRIMITIVE == 1 && NAME !~ "*u_io_reg_*" && PRIMITIVE_TYPE !~ "I/O.*" && PRIMITIVE_TYPE !~ "OTHERS.CLOCK.*"}]
    if {[llength $core_cells] > 0} {
        add_cells_to_pblock [get_pblocks pblock_core] $core_cells
    }

    # Open output txt file for placed IO mapping
    file mkdir "../result/${top_module}"
    set fp [open "../result/${top_module}/io_locations.txt" w]

    # Output dimension file
    set width [expr $max_y - $min_y + 1]
    set length [expr $max_x - $min_x + 1]
    set fp_dim [open "../result/${top_module}/io_dimensions.txt" w]
    puts $fp_dim "$length $width $ring"
    close $fp_dim

    # Shuffle boundary sites for random placement
    set n [llength $boundary_sites]
    for {set i 0} {$i < $n} {incr i} {
        set j [expr {int(rand() * $n)}]
        set temp [lindex $boundary_sites $i]
        lset boundary_sites $i [lindex $boundary_sites $j]
        lset boundary_sites $j $temp
    }

    # 5. Assign Locations
    set idx 0
    foreach reg $io_regs {
        if {$idx >= [llength $boundary_sites]} {
            break
        }
        set site [lindex $boundary_sites $idx]
        set site_name [get_property NAME $site]
        # Lock placement
        set_property LOC $site_name $reg
        
        # Write to txt log file
        if {[regexp {SLICE_X(\d+)Y(\d+)} $site_name match site_x site_y]} {
            puts $fp "[get_property NAME $reg] $site_name $site_x $site_y"
        } else {
            puts $fp "[get_property NAME $reg] $site_name"
        }
        
        incr idx
    }
    close $fp
    puts "Placed $idx IO registers on boundary."
}

# # Auto-execute if sourced
# # But let the caller call it with a specific CR.
# # Default to X2Y2 roughly in middle for safety if args provided
# if { $argc > 0 } {
#     place_io_registers [lindex $argv 0]
# }
