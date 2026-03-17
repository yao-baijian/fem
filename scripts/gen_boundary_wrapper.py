import re
import sys
import os

def generate_wrapper(verilog_file, output_file, clock_region="X2Y2"):
    with open(verilog_file, 'r') as f:
        content = f.read()

    # Find module name
    module_match = re.search(r'module\s+(\w+)\s*\(', content)
    if not module_match:
        print(f"Error: Could not find module definition in {verilog_file}")
        return
    module_name = module_match.group(1)

    # Find ports
    # This is a simple regex, might fail on complex Verilog with comments inside port list.
    # Assuming standard format as seen in c1355.v attachment.
    inputs = []
    outputs = []
    
    # Extract input and output declarations efficiently
    # Normalize whitespace
    normalized_content = re.sub(r'\s+', ' ', content)
    
    # Find inputs
    input_matches = re.finditer(r'input\s+([^;]+);', normalized_content)
    for match in input_matches:
        ports = match.group(1).split(',')
        inputs.extend([p.strip() for p in ports])

    # Find outputs
    output_matches = re.finditer(r'output\s+([^;]+);', normalized_content)
    for match in output_matches:
        ports = match.group(1).split(',')
        outputs.extend([p.strip() for p in ports])

    if not inputs and not outputs:
        print("Error: No inputs or outputs found.")
        return

    # Identify clock ports
    clk_patterns = [r'^clk$', r'^clock$', r'^clk\d*$', r'^clock\d*$']
    clock_ports = []
    
    for port in inputs:
        for pat in clk_patterns:
            if re.match(pat, port.lower()):
                clock_ports.append(port)
                break
                
    data_inputs = [p for p in inputs if p not in clock_ports]

    wrapper_name = f"{module_name}_wrapper"
    
    with open(output_file, 'w') as f:
        f.write(f"module {wrapper_name} (\n")
        f.write("    input clk,\n")
        
        all_ports = inputs + outputs
        port_decls = []
        for port in all_ports:
            if port in clock_ports:
                continue
            direction = "input" if port in inputs else "output"
            port_decls.append(f"    {direction} {port}")
            
        f.write(",\n".join(port_decls))
        f.write("\n);\n\n")

        # Internal wires for connection to the original module
        f.write("    // Internal wires connecting wrapper registers to core logic\n")
        for port in data_inputs:
            f.write(f"    wire {port}_int;\n")
        for port in outputs:
            f.write(f"    wire {port}_int;\n")
        f.write("\n")

        # Instantiate Input Registers
        f.write("    // Input Registers\n")
        for port in data_inputs:
            # FDRE instance for input
            # D = port (external input), Q = internal wire to core, C = clk, CE=1, R=0
            f.write(f"    FDRE #(.INIT(1'b0)) u_io_reg_in_{port} (\n")
            f.write(f"        .C(clk),\n")
            f.write(f"        .CE(1'b1),\n")
            f.write(f"        .D({port}),\n")
            f.write(f"        .Q({port}_int),\n")
            f.write(f"        .R(1'b0)\n")
            f.write(f"    );\n")
        f.write("\n")

        # Instantiate Output Registers
        f.write("    // Output Registers\n")
        for port in outputs:
            # FDRE instance for output
            # D = internal wire from core, Q = port (external output), C = clk, CE=1, R=0
            f.write(f"    FDRE #(.INIT(1'b0)) u_io_reg_out_{port} (\n")
            f.write(f"        .C(clk),\n")
            f.write(f"        .CE(1'b1),\n")
            f.write(f"        .D({port}_int),\n")
            f.write(f"        .Q({port}),\n")
            f.write(f"        .R(1'b0)\n")
            f.write(f"    );\n")
        f.write("\n")

        # Instantiate Original Module
        f.write(f"    {module_name} inst_{module_name} (\n")
        inst_connections = []
        for port in all_ports:
            if port in clock_ports:
                inst_connections.append(f"        .{port}(clk)")
            else:
                inst_connections.append(f"        .{port}({port}_int)")
        f.write(",\n".join(inst_connections) + "\n")
        f.write("    );\n")
        
        f.write("endmodule\n")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 gen_boundary_wrapper.py <input_verilog> <output_verilog> [clock_region]")
        sys.exit(1)
        
    verilog_file = sys.argv[1]
    output_file = sys.argv[2]
    clock_region = sys.argv[3] if len(sys.argv) > 3 else "X2Y2"
    
    generate_wrapper(verilog_file, output_file, clock_region)
