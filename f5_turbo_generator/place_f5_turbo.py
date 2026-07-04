import sys
import re

def parse_netlist(netlist_file):
    nets = {}
    comps = {}
    
    with open(netlist_file, 'r') as f:
        content = f.read()
        
    # Extract components
    comp_blocks = re.findall(r'\(comp\s+\(ref\s+"([^"]+)"\).*?\(footprint\s+"([^"]+)"\)', content, re.DOTALL)
    for ref, fp in comp_blocks:
        comps[ref] = {'footprint': fp}

    # Extract nets
    net_blocks = re.findall(r'\(net\s+\(code\s+([0-9]+)\)\s+\(name\s+"([^"]+)"\)(.*?)(?=\s+\(net\s|\)\s*\)\s*$)', content, re.DOTALL)
    for code, name, nodes_str in net_blocks:
        nets[name] = {'code': code, 'nodes': []}
        node_matches = re.findall(r'\(node\s+\(ref\s+"([^"]+)"\)\s+\(pin\s+"([^"]+)"\)', nodes_str)
        for ref, pin in node_matches:
            nets[name]['nodes'].append((ref, pin))
            
    return comps, nets

def get_footprint_pads(fp_name):
    # Returns a list of (pin_num, dx, dy, shape_size, drill) for generating pads
    if "TO-247" in fp_name:
        # 5.45mm pitch, 3 pins
        return [("1", -5.45, 0, 3.5, 1.5), ("2", 0, 0, 3.5, 1.5), ("3", 5.45, 0, 3.5, 1.5)]
    elif "TO-92" in fp_name:
        # 1.27mm pitch
        return [("1", -1.27, 0, 1.7, 0.8), ("2", 0, 1.27, 1.7, 0.8), ("3", 1.27, 0, 1.7, 0.8)]
    elif "DIN0918" in fp_name: # 3W resistor
        return [("1", -12.7, 0, 3.0, 1.2), ("2", 12.7, 0, 3.0, 1.2)]
    elif "DIN0207" in fp_name: # 1/4W resistor
        return [("1", -5.08, 0, 2.0, 0.9), ("2", 5.08, 0, 2.0, 0.9)]
    elif "Potentiometer" in fp_name:
        # Inline 2.54 pitch
        return [("1", -2.54, 0, 2.0, 0.9), ("2", 0, 0, 2.0, 0.9), ("3", 2.54, 0, 2.0, 0.9)]
    elif "Radial_D10" in fp_name:
        return [("1", -2.5, 0, 2.5, 1.0), ("2", 2.5, 0, 2.5, 1.0)]
    elif "Disc_D20" in fp_name:
        return [("1", -5.0, 0, 2.5, 1.2), ("2", 5.0, 0, 2.5, 1.2)]
    elif "TerminalBlock" in fp_name:
        pads = []
        pins = int(fp_name.split('1x')[1].split('_')[0])
        for i in range(pins):
            pads.append((str(i+1), i*5.0, 0, 3.0, 1.5))
        return pads
    else:
        # Fallback 2 pins
        return [("1", -2.54, 0, 2.0, 0.9), ("2", 2.54, 0, 2.0, 0.9)]

def generate_pcb(comps, nets, out_file):
    with open(out_file, 'w') as f:
        f.write("(kicad_pcb (version 20240108) (generator pcbnew)\n")
        f.write("  (general\n    (thickness 1.6)\n  )\n")
        f.write("  (paper \"A4\")\n")
        f.write("  (layers\n    (0 \"F.Cu\" signal)\n    (1 \"In1.Cu\" signal)\n    (2 \"In2.Cu\" signal)\n    (31 \"B.Cu\" signal)\n  )\n")
        
        # Write Nets
        f.write("  (net 0 \"\")\n")
        net_codes = {}
        for net_name, net_data in nets.items():
            code = net_data['code']
            net_codes[net_name] = code
            f.write(f"  (net {code} \"{net_name}\")\n")
            
        # Layout Algorithm
        # Group components
        power_devices = [r for r in comps if "Q" in r and int(r[1:]) > 2] + [r for r in comps if "D" in r and "Dual" in comps[r]['footprint']]
        input_devices = [r for r in comps if "Q" in r and int(r[1:]) <= 2]
        connectors = [r for r in comps if "J" in r]
        
        placed_x = 20.0
        placed_y = 20.0
        
        # Thermal Edge (Top)
        thermal_x = 30.0
        thermal_y = 20.0
        
        comp_pos = {}
        
        for ref in power_devices:
            comp_pos[ref] = (thermal_x, thermal_y, 0)
            thermal_x += 20.0
            
        # Input stage (Center)
        center_x = 50.0
        center_y = 60.0
        for ref in input_devices:
            comp_pos[ref] = (center_x, center_y, 0)
            center_x += 15.0
            
        # Connectors (Bottom)
        conn_x = 30.0
        conn_y = 100.0
        for ref in connectors:
            comp_pos[ref] = (conn_x, conn_y, 0)
            conn_x += 20.0
            
        # Others (Scatter below center)
        misc_x = 30.0
        misc_y = 80.0
        for ref in comps:
            if ref not in comp_pos:
                comp_pos[ref] = (misc_x, misc_y, 0)
                misc_x += 15.0
                if misc_x > 120.0:
                    misc_x = 30.0
                    misc_y += 15.0

        # Build pad-to-net lookup
        pad_nets = {}
        for net_name, net_data in nets.items():
            for ref, pin in net_data['nodes']:
                pad_nets[f"{ref}_{pin}"] = (net_data['code'], net_name)

        # Write components
        for ref, comp in comps.items():
            x, y, rot = comp_pos[ref]
            fp = comp['footprint']
            f.write(f"  (footprint \"{fp}\" (layer \"F.Cu\")\n")
            f.write(f"    (at {x} {y} {rot})\n")
            f.write(f"    (fp_text reference \"{ref}\" (at 0 -2.5) (layer \"F.SilkS\") (effects (font (size 1 1) (thickness 0.15))))\n")
            
            pads = get_footprint_pads(fp)
            for pin_num, dx, dy, size, drill in pads:
                net_info = pad_nets.get(f"{ref}_{pin_num}")
                net_str = f"(net {net_info[0]} \"{net_info[1]}\")" if net_info else ""
                f.write(f"    (pad \"{pin_num}\" thru_hole circle (at {dx} {dy}) (size {size} {size}) (drill {drill}) (layers \"*.Cu\" \"*.Mask\") {net_str})\n")
            f.write("  )\n")
            
        # Board Outline
        f.write("  (gr_line (start 10 10) (end 150 10) (layer \"Edge.Cuts\") (width 0.1))\n")
        f.write("  (gr_line (start 150 10) (end 150 140) (layer \"Edge.Cuts\") (width 0.1))\n")
        f.write("  (gr_line (start 150 140) (end 10 140) (layer \"Edge.Cuts\") (width 0.1))\n")
        f.write("  (gr_line (start 10 140) (end 10 10) (layer \"Edge.Cuts\") (width 0.1))\n")
        f.write(")\n")

if __name__ == "__main__":
    import glob
    net_files = glob.glob("*.net")
    for net in net_files:
        print(f"Placing {net}...")
        comps, nets = parse_netlist(net)
        out_name = net.replace(".net", ".kicad_pcb")
        generate_pcb(comps, nets, out_name)
        print(f"Generated {out_name}")
