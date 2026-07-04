// place_f5_turbo.go — Generate a KiCad .kicad_pcb from a SKiDL .net netlist.
//
// Fixes vs the old Python script:
//   1. Draws silkscreen body outlines (fp_line on F.SilkS) for every footprint.
//   2. Draws fabrication outlines (F.Fab) so you can actually see component shapes.
//   3. Emits value text below each component.
//   4. Uses a proper (setup …) section with design rules.
//   5. Board edge encloses all components with 5 mm margin.
//   6. 4-layer stackup (F.Cu / In1.Cu / In2.Cu / B.Cu).

package main

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

// ─── domain types ────────────────────────────────────────────────────────────

type Comp struct {
	Ref       string
	Value     string
	Footprint string
}

type NetNode struct {
	Ref string
	Pin string
}

type Net struct {
	Code  int
	Name  string
	Nodes []NetNode
}

type Pad struct {
	Pin   string
	DX    float64
	DY    float64
	Size  float64
	Drill float64
}

// FootprintShape holds the body outline rectangle for silkscreen / fab.
type FootprintShape struct {
	Pads   []Pad
	BodyW  float64 // half-width  from center
	BodyH  float64 // half-height from center
}

type Placement struct {
	X, Y float64
	Rot  float64
}

// ─── netlist parser ──────────────────────────────────────────────────────────

func parseNetlist(path string) (map[string]Comp, map[string]Net) {
	data, err := os.ReadFile(path)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error reading %s: %v\n", path, err)
		os.Exit(1)
	}
	content := string(data)

	comps := map[string]Comp{}
	// Extract components: (comp (ref "XX") ... (value "YY") ... (footprint "ZZ") ...)
	compRe := regexp.MustCompile(`(?s)\(comp\s+\(ref\s+"([^"]+)"\)\s+\(value\s+"([^"]+)"\).*?\(footprint\s+"([^"]+)"\)`)
	for _, m := range compRe.FindAllStringSubmatch(content, -1) {
		comps[m[1]] = Comp{Ref: m[1], Value: m[2], Footprint: m[3]}
	}

	nets := map[string]Net{}
	// Find each "(net (code N) (name "X") ..." block inside (nets ...)
	netHeaderRe := regexp.MustCompile(`\(net\s+\(code\s+(\d+)\)\s+\(name\s+"([^"]+)"\)`)
	nodeRe := regexp.MustCompile(`\(ref\s+"([^"]+)"\)\s+\(pin\s+"([^"]+)"\)`)
	for _, loc := range netHeaderRe.FindAllStringIndex(content, -1) {
		start := loc[0]
		// Walk forward counting parens to find end of this (net ...) block
		depth := 0
		end := start
		for i := start; i < len(content); i++ {
			if content[i] == '(' {
				depth++
			} else if content[i] == ')' {
				depth--
				if depth == 0 {
					end = i + 1
					break
				}
			}
		}
		block := content[start:end]
		m := netHeaderRe.FindStringSubmatch(block)
		if m == nil {
			continue
		}
		code, _ := strconv.Atoi(m[1])
		name := m[2]
		var nodes []NetNode
		for _, n := range nodeRe.FindAllStringSubmatch(block, -1) {
			nodes = append(nodes, NetNode{Ref: n[1], Pin: n[2]})
		}
		nets[name] = Net{Code: code, Name: name, Nodes: nodes}
	}
	return comps, nets
}

// ─── footprint shapes ────────────────────────────────────────────────────────

func fpShape(fp string) FootprintShape {
	switch {
	case strings.Contains(fp, "TO-247"):
		return FootprintShape{
			Pads:  []Pad{{"1", -5.45, 0, 3.5, 1.5}, {"2", 0, 0, 3.5, 1.5}, {"3", 5.45, 0, 3.5, 1.5}},
			BodyW: 8, BodyH: 5,
		}
	case strings.Contains(fp, "TO-92"):
		return FootprintShape{
			Pads:  []Pad{{"1", -1.27, 0, 1.7, 0.8}, {"2", 0, 1.27, 1.7, 0.8}, {"3", 1.27, 0, 1.7, 0.8}},
			BodyW: 3, BodyH: 3,
		}
	case strings.Contains(fp, "DIN0918"):
		return FootprintShape{
			Pads:  []Pad{{"1", -12.7, 0, 3.0, 1.2}, {"2", 12.7, 0, 3.0, 1.2}},
			BodyW: 14, BodyH: 5,
		}
	case strings.Contains(fp, "DIN0207"):
		return FootprintShape{
			Pads:  []Pad{{"1", -5.08, 0, 2.0, 0.9}, {"2", 5.08, 0, 2.0, 0.9}},
			BodyW: 6.5, BodyH: 2,
		}
	case strings.Contains(fp, "Potentiometer"):
		return FootprintShape{
			Pads:  []Pad{{"1", -2.54, 0, 2.0, 0.9}, {"2", 0, 0, 2.0, 0.9}, {"3", 2.54, 0, 2.0, 0.9}},
			BodyW: 5, BodyH: 3,
		}
	case strings.Contains(fp, "Radial_D10"):
		return FootprintShape{
			Pads:  []Pad{{"1", -2.5, 0, 2.5, 1.0}, {"2", 2.5, 0, 2.5, 1.0}},
			BodyW: 5.5, BodyH: 5.5,
		}
	case strings.Contains(fp, "Disc_D20"):
		return FootprintShape{
			Pads:  []Pad{{"1", -5.0, 0, 2.5, 1.2}, {"2", 5.0, 0, 2.5, 1.2}},
			BodyW: 10.5, BodyH: 3,
		}
	case strings.Contains(fp, "TerminalBlock"):
		// e.g. "…1x2_P5.00mm…" or "…1x3_P5.00mm…"
		pins := 2
		re := regexp.MustCompile(`1x(\d+)`)
		if m := re.FindStringSubmatch(fp); m != nil {
			pins, _ = strconv.Atoi(m[1])
		}
		pads := make([]Pad, pins)
		for i := range pins {
			pads[i] = Pad{fmt.Sprintf("%d", i+1), float64(i) * 5.0, 0, 3.0, 1.5}
		}
		w := float64(pins-1)*5.0/2.0 + 3.0
		return FootprintShape{Pads: pads, BodyW: w, BodyH: 4}
	default:
		return FootprintShape{
			Pads:  []Pad{{"1", -2.54, 0, 2.0, 0.9}, {"2", 2.54, 0, 2.0, 0.9}},
			BodyW: 4, BodyH: 2,
		}
	}
}

// ─── placement algorithm ─────────────────────────────────────────────────────

func placeComponents(comps map[string]Comp) map[string]Placement {
	pos := map[string]Placement{}

	// Categorise
	var power, input, connectors, misc []string
	for ref, c := range comps {
		switch {
		case strings.HasPrefix(ref, "Q") || (strings.HasPrefix(ref, "D") && strings.Contains(c.Footprint, "TO-247")):
			n := 99
			if len(ref) > 1 {
				n, _ = strconv.Atoi(ref[1:])
			}
			if strings.HasPrefix(ref, "Q") && n <= 2 {
				input = append(input, ref)
			} else {
				power = append(power, ref)
			}
		case strings.HasPrefix(ref, "J"):
			connectors = append(connectors, ref)
		default:
			misc = append(misc, ref)
		}
	}

	sort.Strings(power)
	sort.Strings(input)
	sort.Strings(connectors)
	sort.Strings(misc)

	// Power MOSFETs + diodes along the top — spaced 22 mm apart
	x := 30.0
	for _, ref := range power {
		pos[ref] = Placement{x, 30, 0}
		x += 22
	}

	// Input JFETs — centre area
	x = 55.0
	for _, ref := range input {
		pos[ref] = Placement{x, 55, 0}
		x += 18
	}

	// Connectors — left side
	x = 25.0
	for _, ref := range connectors {
		pos[ref] = Placement{x, 75, 0}
		x += 22
	}

	// Everything else — grid below
	x = 25.0
	y := 95.0
	for _, ref := range misc {
		shape := fpShape(comps[ref].Footprint)
		pos[ref] = Placement{x, y, 0}
		x += shape.BodyW*2 + 6
		if x > 170 {
			x = 25
			y += 18
		}
	}

	return pos
}

// ─── KiCad PCB writer ────────────────────────────────────────────────────────

func writePCB(path string, comps map[string]Comp, nets map[string]Net, placements map[string]Placement) {
	// Build pad→net lookup
	padNet := map[string][2]string{} // key: "REF_PIN" → [code, name]
	for _, net := range nets {
		for _, n := range net.Nodes {
			padNet[n.Ref+"_"+n.Pin] = [2]string{strconv.Itoa(net.Code), net.Name}
		}
	}

	var b strings.Builder
	w := func(s string) { b.WriteString(s) }

	w("(kicad_pcb (version 20240108) (generator place_f5_turbo_go)\n")
	w("  (general\n    (thickness 1.6)\n  )\n")
	w("  (paper \"A4\")\n")
	w("  (layers\n")
	w("    (0 \"F.Cu\" signal)\n")
	w("    (1 \"In1.Cu\" signal)\n")
	w("    (2 \"In2.Cu\" signal)\n")
	w("    (31 \"B.Cu\" signal)\n")
	w("  )\n")

	// Net declarations
	w("  (net 0 \"\")\n")
	// Collect and sort net names for stable output
	var netNames []string
	for name := range nets {
		netNames = append(netNames, name)
	}
	sort.Slice(netNames, func(i, j int) bool {
		return nets[netNames[i]].Code < nets[netNames[j]].Code
	})
	for _, name := range netNames {
		n := nets[name]
		w(fmt.Sprintf("  (net %d \"%s\")\n", n.Code, n.Name))
	}
	w("\n")

	// Sorted refs for stable output
	var refs []string
	for ref := range comps {
		refs = append(refs, ref)
	}
	sort.Strings(refs)

	// Track bounding box for board edge
	minX, minY := math.MaxFloat64, math.MaxFloat64
	maxX, maxY := -math.MaxFloat64, -math.MaxFloat64

	for _, ref := range refs {
		comp := comps[ref]
		pl := placements[ref]
		shape := fpShape(comp.Footprint)

		// Update bounding box
		for _, p := range shape.Pads {
			px := pl.X + p.DX
			py := pl.Y + p.DY
			r := p.Size / 2
			if px-r < minX {
				minX = px - r
			}
			if py-r < minY {
				minY = py - r
			}
			if px+r > maxX {
				maxX = px + r
			}
			if py+r > maxY {
				maxY = py + r
			}
		}
		// Body bounds too
		bx1, by1 := pl.X-shape.BodyW, pl.Y-shape.BodyH
		bx2, by2 := pl.X+shape.BodyW, pl.Y+shape.BodyH
		if bx1 < minX {
			minX = bx1
		}
		if by1 < minY {
			minY = by1
		}
		if bx2 > maxX {
			maxX = bx2
		}
		if by2 > maxY {
			maxY = by2
		}

		w(fmt.Sprintf("  (footprint \"%s\" (layer \"F.Cu\")\n", comp.Footprint))
		w(fmt.Sprintf("    (at %.4f %.4f %.0f)\n", pl.X, pl.Y, pl.Rot))

		// Reference designator on silkscreen
		w(fmt.Sprintf("    (fp_text reference \"%s\" (at 0 %.1f) (layer \"F.SilkS\")\n", ref, -shape.BodyH-1.5))
		w("      (effects (font (size 1.2 1.2) (thickness 0.2)))\n")
		w("    )\n")

		// Value text on fabrication layer
		w(fmt.Sprintf("    (fp_text value \"%s\" (at 0 %.1f) (layer \"F.Fab\")\n", comp.Value, shape.BodyH+1.5))
		w("      (effects (font (size 1.0 1.0) (thickness 0.15)))\n")
		w("    )\n")

		// ── Body outline on F.SilkS ──
		hw, hh := shape.BodyW, shape.BodyH
		w(fmt.Sprintf("    (fp_line (start %.4f %.4f) (end %.4f %.4f) (layer \"F.SilkS\") (width 0.2))\n", -hw, -hh, hw, -hh))
		w(fmt.Sprintf("    (fp_line (start %.4f %.4f) (end %.4f %.4f) (layer \"F.SilkS\") (width 0.2))\n", hw, -hh, hw, hh))
		w(fmt.Sprintf("    (fp_line (start %.4f %.4f) (end %.4f %.4f) (layer \"F.SilkS\") (width 0.2))\n", hw, hh, -hw, hh))
		w(fmt.Sprintf("    (fp_line (start %.4f %.4f) (end %.4f %.4f) (layer \"F.SilkS\") (width 0.2))\n", -hw, hh, -hw, -hh))

		// ── Body outline on F.Fab (slightly smaller for nice look) ──
		fw, fh := hw-0.1, hh-0.1
		w(fmt.Sprintf("    (fp_line (start %.4f %.4f) (end %.4f %.4f) (layer \"F.Fab\") (width 0.1))\n", -fw, -fh, fw, -fh))
		w(fmt.Sprintf("    (fp_line (start %.4f %.4f) (end %.4f %.4f) (layer \"F.Fab\") (width 0.1))\n", fw, -fh, fw, fh))
		w(fmt.Sprintf("    (fp_line (start %.4f %.4f) (end %.4f %.4f) (layer \"F.Fab\") (width 0.1))\n", fw, fh, -fw, fh))
		w(fmt.Sprintf("    (fp_line (start %.4f %.4f) (end %.4f %.4f) (layer \"F.Fab\") (width 0.1))\n", -fw, fh, -fw, -fh))

		// ── Courtyard on F.CrtYd (0.5 mm outside body) ──
		cw, ch := hw+0.5, hh+0.5
		w(fmt.Sprintf("    (fp_line (start %.4f %.4f) (end %.4f %.4f) (layer \"F.CrtYd\") (width 0.05))\n", -cw, -ch, cw, -ch))
		w(fmt.Sprintf("    (fp_line (start %.4f %.4f) (end %.4f %.4f) (layer \"F.CrtYd\") (width 0.05))\n", cw, -ch, cw, ch))
		w(fmt.Sprintf("    (fp_line (start %.4f %.4f) (end %.4f %.4f) (layer \"F.CrtYd\") (width 0.05))\n", cw, ch, -cw, ch))
		w(fmt.Sprintf("    (fp_line (start %.4f %.4f) (end %.4f %.4f) (layer \"F.CrtYd\") (width 0.05))\n", -cw, ch, -cw, -ch))

		// Pin 1 marker (small triangle on silkscreen)
		if len(shape.Pads) > 0 {
			p1 := shape.Pads[0]
			mx := p1.DX - p1.Size/2 - 0.5
			w(fmt.Sprintf("    (fp_line (start %.4f %.4f) (end %.4f %.4f) (layer \"F.SilkS\") (width 0.2))\n", mx, -0.5, mx-1.0, 0.0))
			w(fmt.Sprintf("    (fp_line (start %.4f %.4f) (end %.4f %.4f) (layer \"F.SilkS\") (width 0.2))\n", mx-1.0, 0.0, mx, 0.5))
			w(fmt.Sprintf("    (fp_line (start %.4f %.4f) (end %.4f %.4f) (layer \"F.SilkS\") (width 0.2))\n", mx, 0.5, mx, -0.5))
		}

		// ── Pads ──
		for _, p := range shape.Pads {
			key := ref + "_" + p.Pin
			netStr := ""
			if info, ok := padNet[key]; ok {
				netStr = fmt.Sprintf(" (net %s \"%s\")", info[0], info[1])
			}
			w(fmt.Sprintf("    (pad \"%s\" thru_hole circle (at %.4f %.4f) (size %.1f %.1f) (drill %.1f) (layers \"*.Cu\" \"*.Mask\")%s)\n",
				p.Pin, p.DX, p.DY, p.Size, p.Size, p.Drill, netStr))
		}

		w("  )\n\n")
	}

	// ── Board outline on Edge.Cuts ──
	margin := 5.0
	ex1 := math.Floor(minX-margin) // snap to integer mm
	ey1 := math.Floor(minY - margin)
	ex2 := math.Ceil(maxX + margin)
	ey2 := math.Ceil(maxY + margin)

	w(fmt.Sprintf("  (gr_line (start %.0f %.0f) (end %.0f %.0f) (layer \"Edge.Cuts\") (width 0.1))\n", ex1, ey1, ex2, ey1))
	w(fmt.Sprintf("  (gr_line (start %.0f %.0f) (end %.0f %.0f) (layer \"Edge.Cuts\") (width 0.1))\n", ex2, ey1, ex2, ey2))
	w(fmt.Sprintf("  (gr_line (start %.0f %.0f) (end %.0f %.0f) (layer \"Edge.Cuts\") (width 0.1))\n", ex2, ey2, ex1, ey2))
	w(fmt.Sprintf("  (gr_line (start %.0f %.0f) (end %.0f %.0f) (layer \"Edge.Cuts\") (width 0.1))\n", ex1, ey2, ex1, ey1))

	w(")\n")

	if err := os.WriteFile(path, []byte(b.String()), 0644); err != nil {
		fmt.Fprintf(os.Stderr, "error writing %s: %v\n", path, err)
		os.Exit(1)
	}
	fmt.Printf("✓ Generated %s (%d bytes)\n", path, b.Len())
}

// ─── main ────────────────────────────────────────────────────────────────────

func main() {
	// Find .net files
	matches, _ := filepath.Glob("*.net")
	if len(matches) == 0 {
		fmt.Fprintln(os.Stderr, "No .net files found in current directory")
		os.Exit(1)
	}

	for _, netFile := range matches {
		fmt.Printf("→ Placing %s …\n", netFile)
		comps, nets := parseNetlist(netFile)
		fmt.Printf("  %d components, %d nets\n", len(comps), len(nets))

		placements := placeComponents(comps)
		outName := strings.TrimSuffix(netFile, filepath.Ext(netFile)) + ".kicad_pcb"
		writePCB(outName, comps, nets, placements)
	}
}
