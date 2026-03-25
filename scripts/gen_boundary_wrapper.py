import re
import sys
import os

def generate_wrapper(verilog_file, output_file, top_module=None, clock_region="X2Y2"):
    with open(verilog_file, 'r') as f:
        content = f.read()

    # If top_module is provided, try to find that specific module
    # otherwise pick the first module found
    if top_module:
        pattern = r'(?s)module\s+' + re.escape(top_module) + r'\b(.*?)(?:\s*endmodule\b)'
        module_match = re.search(pattern, content)
        if not module_match:
            print(f"Error: Could not find module definition for {top_module} in {verilog_file}")
            sys.exit(1)
        module_body = module_match.group(0)
        module_name = top_module
    else:
        # Fallback to first module
        module_match = re.search(r'(?s)module\s+(\w+)\b(.*?)(?:\s*endmodule\b)', content)
        if not module_match:
            print(f"Error: Could not find any module definition in {verilog_file}")
            sys.exit(1)
        module_name = module_match.group(1)
        module_body = module_match.group(0)

        # Remove block comments
    module_body = re.sub(r'/\*.*?\*/', '', module_body, flags=re.DOTALL)
    # Remove single line comments
    module_body = re.sub(r'//.*', '', module_body)

    # Normalize whitespace inside the extracted module only
    normalized_content = re.sub(r'\s+', ' ', module_body)

    # Try to extract ANSI-style port list from module header
    header_ports = None
    header_match = re.search(r'module\s+' + re.escape(module_name) + r'\s*\((.*?)\)\s*;', module_body, re.DOTALL)
    if header_match:
        header_ports = re.sub(r'\s+', ' ', header_match.group(1))
    
    inputs = []
    outputs = []
    # Map port name -> width string, e.g. "[31:0]" or "[WIDTH-1:0]"; empty string means scalar
    port_widths = {}
    
    def _parse_ports_with_width(ports_str, direction):
        """Parse a declaration like 'input [31:0] a, b, c' or 'output a, b'"""
        # Common style is one width applied to all ports in this declaration
        ports_str = ports_str.strip()
        width = ''
        m = re.match(r"(\[[^]]+\])\s*(.*)", ports_str)
        if m:
            width = m.group(1).strip()
            rest = m.group(2)
        else:
            rest = ports_str

        for raw in rest.split(','):
            raw = raw.strip()
            if not raw:
                continue
            # Drop any remaining keywords like 'wire', 'reg'
            tokens = raw.split()
            name = tokens[-1]
            if direction == 'input':
                inputs.append(name)
            else:
                outputs.append(name)
            # Only set width if we don't already have a more specific one
            if name not in port_widths and width:
                port_widths[name] = width

    # Always scan body for non-ANSI input/output declarations
    input_matches = re.finditer(r'\binput\s+([^;]+);', normalized_content)
    for match in input_matches:
        _parse_ports_with_width(match.group(1), 'input')

    output_matches = re.finditer(r'\boutput\s+([^;]+);', normalized_content)
    for match in output_matches:
        _parse_ports_with_width(match.group(1), 'output')

    # Also handle ANSI-style declarations in the module header, if present
    if header_ports is not None:
        ansi_source = header_ports
    else:
        ansi_source = None

    if ansi_source is not None:
        # e.g. module m(input CK, input [7:0] A, output [7:0] Y);
        ansi_matches = re.finditer(r'\b(input|output)\s+(?:wire\s+|reg\s+)?([^,\)]+)', ansi_source)
        for match in ansi_matches:
            direction = match.group(1)
            tail = match.group(2).strip()
            # tail may be '[31:0] A' or just 'A'
            m = re.match(r'(\[[^]]+\])\s*(\w+)', tail)
            if m:
                width = m.group(1).strip()
                port_name = m.group(2)
            else:
                width = ''
                port_name = tail.split()[-1]

            if direction == "input":
                inputs.append(port_name)
            else:
                outputs.append(port_name)
            if width and port_name not in port_widths:
                port_widths[port_name] = width
                
    if not inputs and not outputs:
        print("Error: No inputs or outputs found.")
        sys.exit(1)

    # Deduplicate while preserving order
    inputs = list(dict.fromkeys(inputs))
    outputs = list(dict.fromkeys(outputs))

    # Identify clock ports
    clk_patterns = [r'^clk$', r'^clock$', r'^clk\d*$', r'^clock\d*$', r'^ck$', r'^ck\d*$']
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
            width = port_widths.get(port, '')
            width_str = (width + ' ') if width else ''
            port_decls.append(f"    {direction} {width_str}{port}")
            
        f.write(",\n".join(port_decls))
        f.write("\n);\n\n")

        # Internal wires for connection to the original module
        f.write("    // Internal wires connecting wrapper registers to core logic\n")
        for port in data_inputs:
            width = port_widths.get(port, '')
            width_str = (width + ' ') if width else ''
            f.write(f"    wire {width_str}{port}_int;\n")
        for port in outputs:
            width = port_widths.get(port, '')
            width_str = (width + ' ') if width else ''
            f.write(f"    wire {width_str}{port}_int;\n")
        f.write("\n")

        # Instantiate Input Registers
        f.write("    // Input Registers\n")
        for port in data_inputs:
            width = port_widths.get(port, '')
            if width:
                # Vector input: generate per-bit FDREs
                m = re.match(r'\[(.+):(.+)\]', width)
                if m:
                    msb = m.group(1).strip()
                    lsb = m.group(2).strip()
                else:
                    # Fallback: treat as [WIDTH-1:0]
                    msb = width.strip('[]')
                    lsb = '0'
                width_expr = f"(({msb}) - ({lsb}) + 1)"
                idx_name = f"i_{port}"
                f.write(f"    genvar {idx_name};\n")
                f.write("    generate\n")
                f.write(f"        for ({idx_name} = 0; {idx_name} < {width_expr}; {idx_name} = {idx_name} + 1) begin : gen_in_{port}\n")
                index_expr = f"{idx_name}" if lsb == '0' else f"{idx_name} + ({lsb})"
                f.write(f"            FDRE #(.INIT(1'b0)) u_io_reg_in_{port}_bit (\n")
                f.write(f"                .C(clk),\n")
                f.write(f"                .CE(1'b1),\n")
                f.write(f"                .D({port}[{index_expr}]),\n")
                f.write(f"                .Q({port}_int[{index_expr}]),\n")
                f.write(f"                .R(1'b0)\n")
                f.write(f"            );\n")
                f.write("        end\n")
                f.write("    endgenerate\n")
            else:
                # Scalar input: single FDRE
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
            width = port_widths.get(port, '')
            if width:
                m = re.match(r'\[(.+):(.+)\]', width)
                if m:
                    msb = m.group(1).strip()
                    lsb = m.group(2).strip()
                else:
                    msb = width.strip('[]')
                    lsb = '0'
                width_expr = f"(({msb}) - ({lsb}) + 1)"
                idx_name = f"j_{port}"
                f.write(f"    genvar {idx_name};\n")
                f.write("    generate\n")
                f.write(f"        for ({idx_name} = 0; {idx_name} < {width_expr}; {idx_name} = {idx_name} + 1) begin : gen_out_{port}\n")
                index_expr = f"{idx_name}" if lsb == '0' else f"{idx_name} + ({lsb})"
                f.write(f"            FDRE #(.INIT(1'b0)) u_io_reg_out_{port}_bit (\n")
                f.write(f"                .C(clk),\n")
                f.write(f"                .CE(1'b1),\n")
                f.write(f"                .D({port}_int[{index_expr}]),\n")
                f.write(f"                .Q({port}[{index_expr}]),\n")
                f.write(f"                .R(1'b0)\n")
                f.write(f"            );\n")
                f.write("        end\n")
                f.write("    endgenerate\n")
            else:
                # Scalar output
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
        print("Usage: python3 gen_boundary_wrapper.py <input_verilog> <output_verilog> [top_module] [clock_region]")
        sys.exit(1)
        
    verilog_file = sys.argv[1]
    output_file = sys.argv[2]
    top_module = sys.argv[3] if len(sys.argv) > 3 else None
    clock_region = sys.argv[4] if len(sys.argv) > 4 else "X2Y2"
    
    generate_wrapper(verilog_file, output_file, top_module, clock_region)
