# F5 Turbo v2 — PCB Generator

Generates a KiCad `.kicad_pcb` board layout from a SKiDL-generated netlist (`.net`) for the
[F5 Turbo v2 Class-A power amplifier](https://www.diyaudio.com/community/threads/f5-turbo-v2.297544/).

## Overview

This directory contains the full pipeline for:

1. **Schematic capture** → SKiDL Python generates the `.net` netlist
2. **Placement & PCB generation** → Go program parses the netlist, places components with
   proper footprint outlines, and writes a valid `.kicad_pcb`
3. **PDF export** → `kicad-cli` renders the board to PDF for review

### Circuit Description

The F5 Turbo v2 is a Nelson Pass complementary JFET-input, MOSFET-output, single-ended
Class-A power amplifier. Key components:

| Ref     | Device       | Role                              |
|---------|-------------|-----------------------------------|
| Q1      | 2SK170      | N-channel JFET input              |
| Q2      | 2SJ74       | P-channel JFET input              |
| Q3, Q4  | FQA12P20    | P-channel MOSFET output stage     |
| Q5, Q6  | FQA19N20    | N-channel MOSFET output stage     |
| D1–D4   | MUR3020W    | Ultrafast dual diodes (protection)|
| R7–R10  | 220Ω 3W     | Source/drain resistors             |
| R17–R24 | 1Ω 3W       | Source sense resistors             |
| TH1, TH2| NTC         | Thermal compensation              |

## Directory Structure

```
f5_turbo_generator/
  cmd/
    place_f5_turbo.go       # Go-based PCB generator (replaces Python version)
  components.py             # SKiDL component definitions
  generate_netlists.py      # SKiDL netlist generator → .net output
  place_f5_turbo.py         # (Legacy) Python PCB generator — superseded by Go
  f5_turbo_v2.net           # Generated netlist (44 components, 20 nets)
  f5_turbo_v2.kicad_pcb     # Generated board layout
  f5_turbo_v2.pdf           # Rendered PDF for review
  orthoroute.json           # OrthoRoute configuration for this board
```

## Quick Start

### Prerequisites

- Go 1.21+ (`brew install go`)
- KiCad 9+ with `kicad-cli` on PATH
- Python 3.10+ with SKiDL (`pip install skidl`) — only for netlist generation

### Generate the Netlist (if not present)

```bash
python generate_netlists.py
```

### Build & Run the Go Placer

```bash
go build -o place_f5_turbo cmd/place_f5_turbo.go
./place_f5_turbo
```

Output:
```
→ Placing f5_turbo_v2.net …
  44 components, 20 nets
✓ Generated f5_turbo_v2.kicad_pcb (86680 bytes)
```

### Export PDF

```bash
kicad-cli pcb export pdf f5_turbo_v2.kicad_pcb \
  -o f5_turbo_v2.pdf \
  --layers F.Cu,B.Cu,F.SilkS,F.Fab,Edge.Cuts \
  --mode-single
```

## What the Go Generator Does

The Go placer (`cmd/place_f5_turbo.go`) parses a SKiDL `.net` netlist and produces a
fully-valid KiCad `.kicad_pcb` file with:

- **Pads** — Through-hole pads with correct net assignments
- **Silkscreen outlines** — `fp_line` rectangles on `F.SilkS` for every footprint
- **Fabrication outlines** — Matching outlines on `F.Fab`
- **Courtyard** — `F.CrtYd` outlines (body + 0.5mm margin)
- **Pin 1 markers** — Small triangle indicators on silkscreen
- **Reference designators** — `fp_text reference` on silkscreen
- **Value text** — `fp_text value` on fabrication layer
- **Board outline** — Auto-calculated `Edge.Cuts` rectangle (bounding box + 5mm margin)

### Placement Algorithm

Components are grouped by function and placed in zones:

| Zone        | Y position | Components                    |
|-------------|-----------|-------------------------------|
| Power rail  | Top       | Q3–Q6 (MOSFETs), D1–D4 (diodes) |
| Input stage | Center    | Q1 (2SK170), Q2 (2SJ74)      |
| Connectors  | Left      | J1 (INPUT), J2 (OUTPUT), J3 (POWER) |
| Passives    | Grid below| Resistors, caps, pots, thermistors |

### Why Go Over Python

The original Python placer was a quick prototype with several issues:
- No component body outlines (just bare pad holes)
- Missing silkscreen/fab layer content
- Components placed on top of board edge
- Regex-based net parser that silently skipped half the nets

The Go rewrite provides:
- Correct S-expression output verified against KiCad 10
- Balanced-paren net parser (finds all 20 nets reliably)
- Footprint shapes with proper body dimensions per package type
- Auto-calculated board bounds

## Using with OrthoRoute

To auto-route this board with OrthoRoute-Metal:

```bash
cd ..
python main.py cli f5_turbo_generator/f5_turbo_v2.kicad_pcb \
  -o f5_turbo_generator/f5_turbo_v2_routed.kicad_pcb
```

## License

Same as the parent OrthoRoute-Metal project. See [../LICENSE.md](../LICENSE.md).
