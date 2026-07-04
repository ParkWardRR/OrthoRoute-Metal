"""KiCad file parser (direct file parsing)."""
import logging
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Any

from ...domain.models.board import Board, Component, Net, Pad, Layer, Coordinate
from ...domain.models.constraints import DRCConstraints, NetClass

logger = logging.getLogger(__name__)


class KiCadFileParser:
    """Parser for KiCad board files (.kicad_pcb)."""
    
    def __init__(self):
        """Initialize file parser."""
        pass
    
    def load_board(self, file_path: str) -> Optional[Board]:
        """Load board from KiCad file."""
        try:
            board_data = self.parse_file(file_path)
            return self._convert_to_domain_board(board_data, file_path)
        except Exception as e:
            logger.error(f"Failed to load board from {file_path}: {e}")
            return None
    
    def parse_file(self, file_path: str) -> Dict[str, Any]:
        """Parse KiCad board file."""
        try:
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"Board file not found: {file_path}")
            
            if path.suffix.lower() == '.kicad_pcb':
                return self._parse_kicad_pcb(file_path)
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")
                
        except Exception as e:
            logger.error(f"Error parsing board file {file_path}: {e}")
            raise
    
    def _parse_kicad_pcb(self, file_path: str) -> Dict[str, Any]:
        """Parse .kicad_pcb file format."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Basic S-expression parsing
            # This is a simplified parser - a full implementation would use a proper S-expression parser
            board_data = {
                'title': self._extract_title(content),
                'layers': self._extract_layers(content),
                'components': self._extract_components(content),
                'nets': self._extract_nets(content),
                'design_rules': self._extract_design_rules(content),
                'tracks': self._extract_tracks(content),
                'vias': self._extract_vias(content)
            }
            
            return board_data
            
        except Exception as e:
            logger.error(f"Error parsing .kicad_pcb file: {e}")
            raise
    
    def _extract_title(self, content: str) -> str:
        """Extract board title from content."""
        match = re.search(r'\(title\s+"([^"]+)"\)', content)
        return match.group(1) if match else "Untitled Board"
    
    def _extract_layers(self, content: str) -> List[Dict[str, Any]]:
        """Extract layer information."""
        layers = []
        
        # Find layers section
        layers_match = re.search(r'\(layers\s+(.*?)\n\s*\)', content, re.DOTALL)
        if not layers_match:
            # Default layers if not found
            return [
                {'id': 0, 'name': 'F.Cu', 'type': 'signal'},
                {'id': 31, 'name': 'B.Cu', 'type': 'signal'}
            ]
        
        layers_content = layers_match.group(1)
        
        # Parse individual layer definitions
        layer_pattern = r'\((\d+)\s+"([^"]+)"\s+(\w+)'
        for match in re.finditer(layer_pattern, layers_content):
            layer_id = int(match.group(1))
            layer_name = match.group(2)
            layer_type = match.group(3)
            
            layers.append({
                'id': layer_id,
                'name': layer_name,
                'type': layer_type,
                'stackup_position': layer_id
            })
        
        return layers
    
    def _extract_components(self, content: str) -> List[Dict[str, Any]]:
        """Extract component (footprint) information."""
        components = []
        
        # Find all footprint definitions
        footprint_pattern = r'\(footprint\s+"([^"]+)"\s+(.*?)\n  \)'
        
        for match in re.finditer(footprint_pattern, content, re.DOTALL):
            footprint_lib = match.group(1)
            footprint_content = match.group(2)
            
            # Extract footprint details
            component = {
                'footprint': footprint_lib,
                'reference': '',
                'value': '',
                'x': 0.0,
                'y': 0.0,
                'angle': 0.0,
                'layer': 'F.Cu',
                'pads': []
            }
            
            # Extract reference
            ref_match = re.search(r'\(fp_text\s+reference\s+"([^"]+)"', footprint_content)
            if ref_match:
                component['reference'] = ref_match.group(1)
            
            # Extract value
            val_match = re.search(r'\(fp_text\s+value\s+"([^"]+)"', footprint_content)
            if val_match:
                component['value'] = val_match.group(1)
            
            # Extract position
            pos_match = re.search(r'\(at\s+([\d.-]+)\s+([\d.-]+)(?:\s+([\d.-]+))?\)', footprint_content)
            if pos_match:
                component['x'] = float(pos_match.group(1))
                component['y'] = float(pos_match.group(2))
                if pos_match.group(3):
                    component['angle'] = float(pos_match.group(3))
            
            # Extract layer
            layer_match = re.search(r'\(layer\s+"([^"]+)"\)', footprint_content)
            if layer_match:
                component['layer'] = layer_match.group(1)
            
            # Extract pads
            component['pads'] = self._extract_pads(footprint_content, component['reference'], component['x'], component['y'])
            
            if component['reference']:  # Only add if has reference
                components.append(component)
        
        return components
    
    def _extract_pads(self, footprint_content: str, component_ref: str, comp_x: float = 0.0, comp_y: float = 0.0) -> List[Dict[str, Any]]:
        """Extract pad information from footprint."""
        pads = []
        
        # Find all pads in footprint
        pad_pattern = r'\(pad\s+"?([^"]+)"?\s+([\w_]+)\s+([\w_]+)\s+([^\n\r]*)'
        
        for match in re.finditer(pad_pattern, footprint_content, re.DOTALL):
            pad_number = match.group(1)
            pad_type = match.group(2)  # thru_hole, smd, etc.
            pad_shape = match.group(3)  # circle, rect, etc.
            pad_details = match.group(4)
            
            pad = {
                'id': f"{component_ref}_{pad_number}",
                'number': pad_number,
                'type': pad_type,
                'shape': pad_shape,
                'x': 0.0,
                'y': 0.0,
                'width': 1.0,
                'height': 1.0,
                'drill_size': None,
                'layer': 'F.Cu',
                'net_id': None
            }
            
            # Extract relative position
            pos_match = re.search(r'\(at\s+([\d.-]+)\s+([\d.-]+)', pad_details)
            if pos_match:
                pad['x'] = float(pos_match.group(1)) + comp_x
                pad['y'] = float(pos_match.group(2)) + comp_y
            
            # Extract size
            size_match = re.search(r'\(size\s+([\d.-]+)\s+([\d.-]+)', pad_details)
            if size_match:
                pad['width'] = float(size_match.group(1))
                pad['height'] = float(size_match.group(2))
            
            # Extract drill size
            drill_match = re.search(r'\(drill\s+([\d.-]+)', pad_details)
            if drill_match:
                pad['drill_size'] = float(drill_match.group(1))
            
            # Extract layers
            layers_match = re.search(r'\(layers\s+"([^"]+)"', pad_details)
            if layers_match:
                pad['layer'] = layers_match.group(1)
            
            # Extract net
            net_match = re.search(r'\(net\s+(\d+)\s+"([^"]*)"', pad_details)
            print("pad_details:", repr(pad_details))
            print("net_match:", net_match)
            if net_match:
                net_code = int(net_match.group(1))
                if net_code > 0:  # Skip unconnected (net 0)
                    pad['net_id'] = str(net_code)
            
            pads.append(pad)
        
        return pads
    
    def _extract_nets(self, content: str) -> List[Dict[str, Any]]:
        """Extract net information."""
        nets = []
        
        # Find all net definitions
        net_pattern = r'\(net\s+(\d+)\s+"([^"]*)"\)'
        
        for match in re.finditer(net_pattern, content):
            net_code = int(match.group(1))
            net_name = match.group(2)
            
            if net_code > 0 and net_name:  # Skip unconnected and empty names
                nets.append({
                    'id': str(net_code),
                    'name': net_name,
                    'netclass': 'Default'
                })
        
        return nets
    
    def _extract_design_rules(self, content: str) -> Dict[str, Any]:
        """Extract design rule information."""
        rules = {
            'min_track_width': 0.1,
            'min_track_spacing': 0.1,
            'min_via_diameter': 0.2,
            'min_via_drill': 0.1,
            'default_track_width': 0.2,  # KiCad default
            'default_clearance': 0.2,
            'default_via_diameter': 0.8,  # KiCad default
            'default_via_drill': 0.4,  # Typical for 0.8mm via
            'netclasses': {}
        }
        
        # Find setup section
        setup_match = re.search(r'\(setup\s+(.*?)\n\s*\)', content, re.DOTALL)
        if setup_match:
            setup_content = setup_match.group(1)
            
            # Extract various rules
            rules_map = {
                'min_track_width': r'\(min_track_width\s+([\d.]+)\)',
                'min_via_diameter': r'\(min_via_diameter\s+([\d.]+)\)',
                'min_via_drill': r'\(min_via_drill\s+([\d.]+)\)',
                'default_track_width': r'\(default_track_width\s+([\d.]+)\)',
                'default_via_diameter': r'\(default_via_diameter\s+([\d.]+)\)',
                'default_via_drill': r'\(default_via_drill\s+([\d.]+)\)'
            }
            
            for rule_name, pattern in rules_map.items():
                match = re.search(pattern, setup_content)
                if match:
                    rules[rule_name] = float(match.group(1))
        
        return rules
    
    def _extract_tracks(self, content: str) -> List[Dict[str, Any]]:
        """Extract existing track information."""
        tracks = []
        
        # Find all segment definitions
        segment_pattern = r'\(segment\s+(.*?)\)'
        
        for match in re.finditer(segment_pattern, content, re.DOTALL):
            segment_content = match.group(1)
            
            track = {
                'start': {'x': 0.0, 'y': 0.0},
                'end': {'x': 0.0, 'y': 0.0},
                'width': 0.25,
                'layer': 'F.Cu',
                'net': 0
            }
            
            # Extract start point
            start_match = re.search(r'\(start\s+([\d.-]+)\s+([\d.-]+)\)', segment_content)
            if start_match:
                track['start'] = {
                    'x': float(start_match.group(1)),
                    'y': float(start_match.group(2))
                }
            
            # Extract end point
            end_match = re.search(r'\(end\s+([\d.-]+)\s+([\d.-]+)\)', segment_content)
            if end_match:
                track['end'] = {
                    'x': float(end_match.group(1)),
                    'y': float(end_match.group(2))
                }
            
            # Extract width
            width_match = re.search(r'\(width\s+([\d.]+)\)', segment_content)
            if width_match:
                track['width'] = float(width_match.group(1))
            
            # Extract layer
            layer_match = re.search(r'\(layer\s+"([^"]+)"\)', segment_content)
            if layer_match:
                track['layer'] = layer_match.group(1)
            
            # Extract net
            net_match = re.search(r'\(net\s+(\d+)\)', segment_content)
            if net_match:
                track['net'] = int(net_match.group(1))
            
            tracks.append(track)
        
        return tracks
    
    def _extract_vias(self, content: str) -> List[Dict[str, Any]]:
        """Extract existing via information."""
        vias = []
        
        # Find all via definitions
        via_pattern = r'\(via\s+(.*?)\)'
        
        for match in re.finditer(via_pattern, content, re.DOTALL):
            via_content = match.group(1)
            
            via = {
                'x': 0.0,
                'y': 0.0,
                'size': 0.6,
                'drill': 0.3,
                'layers': ['F.Cu', 'B.Cu'],
                'net': 0
            }
            
            # Extract position
            pos_match = re.search(r'\(at\s+([\d.-]+)\s+([\d.-]+)\)', via_content)
            if pos_match:
                via['x'] = float(pos_match.group(1))
                via['y'] = float(pos_match.group(2))
            
            # Extract size
            size_match = re.search(r'\(size\s+([\d.]+)\)', via_content)
            if size_match:
                via['size'] = float(size_match.group(1))
            
            # Extract drill
            drill_match = re.search(r'\(drill\s+([\d.]+)\)', via_content)
            if drill_match:
                via['drill'] = float(drill_match.group(1))
            
            # Extract layers
            layers_match = re.search(r'\(layers\s+"([^"]+)"\s+"([^"]+)"\)', via_content)
            if layers_match:
                via['layers'] = [layers_match.group(1), layers_match.group(2)]
            
            # Extract net
            net_match = re.search(r'\(net\s+(\d+)\)', via_content)
            if net_match:
                via['net'] = int(net_match.group(1))
            
            vias.append(via)
        
        return vias
    
    def create_board_from_data(self, board_data: Dict[str, Any]) -> Board:
        """Create Board domain object from parsed data."""
        # Create board
        board = Board(
            id='parsed_board',
            name=board_data.get('title', 'Parsed Board'),
            thickness=1.6,  # Default thickness
            layer_count=len([l for l in board_data.get('layers', []) if 'Cu' in l.get('name', '')])
        )
        
        # Add layers
        for layer_data in board_data.get('layers', []):
            layer = Layer(
                name=layer_data['name'],
                type='signal' if 'Cu' in layer_data['name'] else 'other',
                stackup_position=layer_data.get('stackup_position', 0)
            )
            board.add_layer(layer)
        
        # Add components
        for comp_data in board_data.get('components', []):
            component = Component(
                id=comp_data.get('reference', ''),
                reference=comp_data.get('reference', ''),
                value=comp_data.get('value', ''),
                footprint=comp_data.get('footprint', ''),
                position=Coordinate(comp_data.get('x', 0), comp_data.get('y', 0)),
                angle=comp_data.get('angle', 0),
                layer=comp_data.get('layer', 'F.Cu')
            )
            
            # Add pads
            for pad_data in comp_data.get('pads', []):
                pad = Pad(
                    id=pad_data['id'],
                    component_id=component.id,
                    net_id=pad_data.get('net_id'),
                    position=Coordinate(pad_data.get('x', 0), pad_data.get('y', 0)),
                    size=(pad_data.get('width', 1.0), pad_data.get('height', 1.0)),
                    drill_size=pad_data.get('drill_size'),
                    layer=pad_data.get('layer', 'F.Cu'),
                    shape=pad_data.get('shape', 'circle')
                )
                component.pads.append(pad)
            
            board.add_component(component)
        
        # Add nets
        for net_data in board_data.get('nets', []):
            # Find pads for this net
            net_pads = []
            for component in board.components:
                for pad in component.pads:
                    if pad.net_id == net_data['id']:
                        net_pads.append(pad)
            
            if net_pads:
                net = Net(
                    id=net_data['id'],
                    name=net_data['name'],
                    netclass=net_data.get('netclass', 'Default'),
                    pads=net_pads
                )
                board.add_net(net)
        
        return board
    
    def _convert_to_domain_board(self, board_data: Dict[str, Any], file_path: str) -> Board:
        """Convert parsed board data to domain Board object."""
        return self.create_board_from_data(board_data)