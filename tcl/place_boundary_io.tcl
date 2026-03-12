# Usage: source tcl/place_boundary_io.tcl
# Must be run after synthesis/link_design

proc place_io_registers {clock_region} {
    puts "Placing IO registers on boundary of clock region: $clock_region"
    
    # 1. Get all Slice sites in the clock region
    set sites [get_sites -of_objects [get_clock_regions $clock_region] -filter {SITE_TYPE =~ "SLICE*"}]
    
    if {[llength $sites] == 0} {
        puts "Error: No slice sites found in clock region $clock_region"
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

    puts "Clock Region $clock_region Bounding Box: X\[$min_x:$max_x\] Y\[$min_y:$max_y\]"

    # 3. Collect boundary sites in order (Bottom -> Right -> Top -> Left)
    set boundary_sites {}

    # Bottom Edge (y = min_y, x increases)
    foreach x $x_coords {
        if {[info exists site_map($x,$min_y)]} {
            lappend boundary_sites $site_map($x,$min_y)
        }
    }

    # Right Edge (x = max_x, y increases, exclude corners to avoid dupes)
    foreach y $y_coords {
        if {$y == $min_y || $y == $max_y} continue
        if {[info exists site_map($max_x,$y)]} {
            lappend boundary_sites $site_map($max_x,$y)
        }
    }

    # Top Edge (y = max_y, x decreases)
    set x_coords_rev [lsort -decreasing -integer $x_coords]
    foreach x $x_coords_rev {
        if {[info exists site_map($x,$max_y)]} {
            lappend boundary_sites $site_map($x,$max_y)
        }
    }

    # Left Edge (x = min_x, y decreases, exclude corners)
    set y_coords_rev [lsort -decreasing -integer $y_coords]
    foreach y $y_coords_rev {
        if {$y == $min_y || $y == $max_y} continue
        if {[info exists site_map($min_x,$y)]} {
            lappend boundary_sites $site_map($min_x,$y)
        }
    }
    
    puts "Found [llength $boundary_sites] boundary slice sites."

    # 4. Get IO Registers
    # Assuming names like u_io_reg_in_* and u_io_reg_out_*
    # We want FDRE cells.
    set io_regs [get_cells -hierarchical -filter {REF_NAME == FDRE && NAME =~ "*u_io_reg_*"}]
    set io_regs [lsort $io_regs]

    if {[llength $io_regs] == 0} {
        puts "Warning: No IO registers found matching pattern *u_io_reg_*"
        return
    }
    
    if {[llength $io_regs] > [llength $boundary_sites]} {
        puts "Warning: More IO registers ([llength $io_regs]) than boundary sites ([llength $boundary_sites]). Some will be unplaced."
    }

    # 5. Assign Locations
    set idx 0
    foreach reg $io_regs {
        if {$idx >= [llength $boundary_sites]} {
            break
        }
        set site [lindex $boundary_sites $idx]
        
        # Lock placement
        set_property LOC $site $reg
        
        # Also need to fix the BEL if we want exact placement, but LOC is often enough for slice granularity. 
        # But if we want 1 reg per slice, it's safer.
        # Actually, a slice has 8 FFs. We can pack 8 regs per slice if needed.
        # To keep it simple (as described "boundary nodes"), let's just LOC to the SLICE.
        # Vivado will pick a BEL inside.
        
        incr idx
    }
    puts "Placed $idx IO registers on boundary."
}

# Auto-execute if sourced
# But let the caller call it with a specific CR.
# Default to X2Y2 roughly in middle for safety if args provided
if { $argc > 0 } {
    place_io_registers [lindex $argv 0]
}
