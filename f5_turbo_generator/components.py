from skidl import Part, Pin, SKIDL

# Helper to create a standalone part without needing KiCad libraries
def _create_part(name, ref, value, footprint, pins):
    p = Part(lib=None, name=name, ref=ref, value=value, footprint=footprint, tool=SKIDL)
    for p_name, p_num in pins.items():
        if isinstance(p_num, list):
            # Multiple pins mapping to same logical name
            for num in p_num:
                p.add_pins(Pin(num=str(num), name=p_name))
        else:
            p.add_pins(Pin(num=str(p_num), name=p_name))
    return p

def JFET_N(ref, val="2SK170"):
    # D, G, S (Standard TO-92 is often 1=D, 2=G, 3=S or 1=S, 2=G, 3=D depending on specific part)
    # 2SK170: 1:D, 2:G, 3:S
    return _create_part("Q_NJFET_DGS", ref, val, "Package_TO_SOT_THT:TO-92_Inline", {'D':'1', 'G':'2', 'S':'3'})

def JFET_P(ref, val="2SJ74"):
    return _create_part("Q_PJFET_DGS", ref, val, "Package_TO_SOT_THT:TO-92_Inline", {'D':'1', 'G':'2', 'S':'3'})

def MOSFET_N(ref, val="FQA19N20"):
    # TO-247 GDS: 1=G, 2=D, 3=S
    return _create_part("Q_NMOS_GDS", ref, val, "Package_TO_SOT_THT:TO-247-3_Vertical", {'G':'1', 'D':'2', 'S':'3'})

def MOSFET_P(ref, val="FQA12P20"):
    return _create_part("Q_PMOS_GDS", ref, val, "Package_TO_SOT_THT:TO-247-3_Vertical", {'G':'1', 'D':'2', 'S':'3'})

def Resistor(ref, val, power="1/4W"):
    fp = "Resistor_THT:R_Axial_DIN0918_L18.0mm_D9.0mm_P25.40mm_Horizontal" if "3W" in power or "5W" in power else "Resistor_THT:R_Axial_DIN0207_L6.3mm_D2.5mm_P10.16mm_Horizontal"
    # R has generic pins 1 and 2
    # SKiDL allows accessing by number so we map '1' to '1' and '2' to '2'
    return _create_part("R", ref, val, fp, {'1':'1', '2':'2'})

def Potentiometer(ref, val="5K"):
    return _create_part("R_Potentiometer_Trim", ref, val, "Potentiometer_THT:Potentiometer_Bourns_3296W_Vertical", {'1':'1', '2':'2', '3':'3'})

def Capacitor(ref, val, type="electrolytic"):
    fp = "Capacitor_THT:CP_Radial_D10.0mm_P5.00mm" if type == "electrolytic" else "Capacitor_THT:C_Rect_L13.0mm_W4.0mm_P10.00mm_FKS3_FKP3"
    return _create_part("C", ref, val, fp, {'1':'1', '2':'2'})

def PowerDiode(ref, val="MUR3020W"):
    # MUR3020W is dual diode common cathode (A1=1, K=2, A2=3)
    return _create_part("D_Dual_ACA", ref, val, "Package_TO_SOT_THT:TO-247-3_Vertical", {'A1':'1', 'K':'2', 'A2':'3'})

def Thermistor(ref, val="CL60"):
    return _create_part("Thermistor_NTC", ref, val, "Capacitor_THT:C_Disc_D20.0mm_W5.0mm_P10.00mm", {'1':'1', '2':'2'})

def Connector(ref, val, pins=2):
    fp = f"TerminalBlock_Phoenix:TerminalBlock_Phoenix_MKDS_1,5-{pins}_1x{pins}_P5.00mm_Horizontal"
    pin_dict = {str(i): str(i) for i in range(1, pins+1)}
    return _create_part(f"Conn_{pins}", ref, val, fp, pin_dict)
