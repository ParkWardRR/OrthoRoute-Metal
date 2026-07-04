import os
sym_dir = os.path.abspath("kicad-symbols")
os.environ['KICAD8_SYMBOL_DIR'] = sym_dir

from skidl import *
from components import *

# Tell SKiDL we are using KiCad v8+ (.kicad_sym format)
set_default_tool(KICAD8)
lib_search_paths[KICAD8].append(sym_dir)

def build_f5_turbo_v2():
    """Defines the Nelson Pass F5 Turbo V2 schematic using SKiDL."""
    # Reset internal SKiDL state to allow multiple circuits
    default_circuit.reset()

    # Define main nets
    gnd = Net('GND')
    v_plus = Net('+32V')
    v_minus = Net('-32V')
    v_in = Net('IN')
    v_out = Net('OUT')
    
    # Input Stage
    R1 = Resistor('R1', '1K')
    R2 = Resistor('R2', '47.5K')
    v_in & R1 & Net('Q1_G')
    v_in & R2 & gnd
    
    Q1 = JFET_N('Q1', '2SK170')
    Q2 = JFET_P('Q2', '2SJ74')
    
    # Connect input JFETs gates to R1
    Q1['G'] += R1[2]
    Q2['G'] += R1[2]
    
    # Symmetry Potentiometer (P3) and R3/R4
    P3 = Potentiometer('P3', '200')
    R3 = Resistor('R3', '10')
    R4 = Resistor('R4', '10')
    
    Q1['S'] += P3[1], R3[1]
    P3[2] += gnd
    P3[3] += Q2['S']
    R3[2] += gnd
    R4[1] += gnd
    R4[2] += Q2['S']
    
    # Feedback resistors (R7, R8, R9, R10)
    R7 = Resistor('R7', '220', power='3W')
    R8 = Resistor('R8', '220', power='3W')
    R9 = Resistor('R9', '220', power='3W')
    R10 = Resistor('R10', '220', power='3W')
    
    Q1['S'] += R7[1], R8[1]
    Q2['S'] += R9[1], R10[1]
    
    v_out += R7[2], R8[2], R9[2], R10[2]
    
    # Bias Network
    R5 = Resistor('R5', '1K')
    R6 = Resistor('R6', '1K')
    P1 = Potentiometer('P1', '5K')
    P2 = Potentiometer('P2', '5K')
    C1 = Capacitor('C1', '10uF', type='electrolytic')
    C2 = Capacitor('C2', '10uF', type='electrolytic')
    R11 = Resistor('R11', '2.2K')
    R12 = Resistor('R12', '2.2K')
    TH1 = Thermistor('TH1', 'TH')
    TH2 = Thermistor('TH2', 'TH')
    
    # P-channel bias (Top half)
    v_plus += R5[1], P1[1], P1[2], C1[1]
    gnd += C1[2]
    Q1['D'] += R5[2], P1[3], R11[1]
    R11[2] += TH1[1]
    TH1[2] += Net('Q3_S') # Source node of top MOSFETs
    
    # N-channel bias (Bottom half)
    v_minus += R6[2], P2[2], P2[3], C2[2]
    gnd += C2[1]
    Q2['D'] += R6[1], P2[1], R12[2]
    R12[1] += TH2[2]
    TH2[1] += Net('Q5_S') # Source node of bottom MOSFETs
    
    # Output Stage (2 parallel pairs for V2)
    Q3 = MOSFET_P('Q3', 'FQA12P20')
    Q4 = MOSFET_P('Q4', 'FQA12P20')
    Q5 = MOSFET_N('Q5', 'FQA19N20')
    Q6 = MOSFET_N('Q6', 'FQA19N20')
    
    # Gate resistors
    R13 = Resistor('R13', '47.5')
    R14 = Resistor('R14', '47.5')
    R15 = Resistor('R15', '47.5')
    R16 = Resistor('R16', '47.5')
    
    Q1['D'] += R13[1], R14[1]
    Q2['D'] += R15[1], R16[1]
    
    Q3['G'] += R13[2]
    Q4['G'] += R14[2]
    Q5['G'] += R15[2]
    Q6['G'] += R16[2]
    
    # Source Resistors & Parallel Diodes (The V2 specific feature)
    # Top Half
    R17 = Resistor('R17', '1', power='3W')
    R18 = Resistor('R18', '1', power='3W')
    D1 = PowerDiode('D1', 'MUR3020W')
    R19 = Resistor('R19', '1', power='3W')
    R20 = Resistor('R20', '1', power='3W')
    D2 = PowerDiode('D2', 'MUR3020W')
    
    v_plus += R17[1], R18[1], D1['A1', 'A2'], R19[1], R20[1], D2['A1', 'A2']
    Q3['S'] += R17[2], R18[2], D1['K'], TH1[2]
    Q4['S'] += R19[2], R20[2], D2['K']
    
    # Bottom Half
    R21 = Resistor('R21', '1', power='3W')
    R22 = Resistor('R22', '1', power='3W')
    D3 = PowerDiode('D3', 'MUR3020W')
    R23 = Resistor('R23', '1', power='3W')
    R24 = Resistor('R24', '1', power='3W')
    D4 = PowerDiode('D4', 'MUR3020W')
    
    v_minus += R21[2], R22[2], D3['K'], R23[2], R24[2], D4['K']
    Q5['S'] += R21[1], R22[1], D3['A1', 'A2'], TH2[1]
    Q6['S'] += R23[1], R24[1], D4['A1', 'A2']
    
    # Connect drains to Output
    v_out += Q3['D'], Q4['D'], Q5['D'], Q6['D']
    
    # Connectors
    Conn_In = Connector('J1', 'INPUT')
    Conn_In[1] += v_in
    Conn_In[2] += gnd
    
    Conn_Out = Connector('J2', 'OUTPUT')
    Conn_Out[1] += v_out
    Conn_Out[2] += gnd
    
    Conn_Pwr = Connector('J3', 'POWER', pins=3)
    Conn_Pwr[1] += v_plus
    Conn_Pwr[2] += gnd
    Conn_Pwr[3] += v_minus

    # Generate Netlist
    generate_netlist(file_="f5_turbo_v2.net")
    generate_pcb(file_="f5_turbo_v2.kicad_pcb")
    print("Generated F5 Turbo V2.")


def build_f5_turbo_v1():
    """F5 Turbo V1"""
    default_circuit.reset()
    # Stub: V1 has 2 pairs of output devices but no parallel diodes
    # (Implementation omitted for brevity, generating an empty board for now to show pipeline)
    generate_pcb(file_="f5_turbo_v1.kicad_pcb")
    print("Generated F5 Turbo V1.")

def build_f5_turbo_v3():
    """F5 Turbo V3"""
    default_circuit.reset()
    generate_pcb(file_="f5_turbo_v3.kicad_pcb")
    print("Generated F5 Turbo V3.")

def build_f5_turbo_psu():
    """F5 Turbo Power Supply"""
    default_circuit.reset()
    generate_pcb(file_="f5_turbo_psu.kicad_pcb")
    print("Generated F5 Turbo PSU.")

if __name__ == "__main__":
    print("Generating F5 Turbo Netlists and Base PCBs...")
    build_f5_turbo_v1()
    build_f5_turbo_v2()
    build_f5_turbo_v3()
    build_f5_turbo_psu()
    print("Done!")
