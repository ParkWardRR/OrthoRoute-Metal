"""Domain models for design rule constraints."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum


class ClearanceType(Enum):
    """Types of clearance rules."""
    TRACK_TO_TRACK = "track_to_track"
    TRACK_TO_PAD = "track_to_pad"
    TRACK_TO_VIA = "track_to_via"
    VIA_TO_VIA = "via_to_via"
    PAD_TO_PAD = "pad_to_pad"
    COPPER_POUR = "copper_pour"


@dataclass(frozen=True)
class NetClass:
    """Value object representing a netclass with its constraints."""
    name: str
    track_width: float
    via_diameter: float
    via_drill: float
    clearance: float = 0.2  # Default clearance in mm
    track_width_min: Optional[float] = None
    track_width_max: Optional[float] = None
    via_diameter_min: Optional[float] = None
    via_diameter_max: Optional[float] = None
    
    def __post_init__(self) -> None:
        # Set min/max defaults if not provided
        if self.track_width_min is None:
            object.__setattr__(self, 'track_width_min', self.track_width * 0.5)
        if self.track_width_max is None:
            object.__setattr__(self, 'track_width_max', self.track_width * 2.0)
        if self.via_diameter_min is None:
            object.__setattr__(self, 'via_diameter_min', self.via_diameter * 0.8)
        if self.via_diameter_max is None:
            object.__setattr__(self, 'via_diameter_max', self.via_diameter * 1.5)
    
    def is_track_width_valid(self, width: float) -> bool:
        """Check if track width is within netclass limits."""
        return self.track_width_min <= width <= self.track_width_max
    
    def is_via_size_valid(self, diameter: float) -> bool:
        """Check if via diameter is within netclass limits."""
        return self.via_diameter_min <= diameter <= self.via_diameter_max


@dataclass
class DRCConstraints:
    """Domain entity containing all design rule constraints."""
    
    # Global minimums (absolute limits)
    min_track_width: float = 0.1  # mm
    min_track_spacing: float = 0.1  # mm
    min_via_diameter: float = 0.2  # mm
    min_via_drill: float = 0.1  # mm
    
    # Default values
    default_track_width: float = 0.25  # mm
    default_clearance: float = 0.2  # mm
    default_via_diameter: float = 0.6  # mm
    default_via_drill: float = 0.3  # mm
    
    # Netclasses
    netclasses: Dict[str, NetClass] = field(default_factory=dict)
    
    # Advanced constraints
    blind_via_enabled: bool = True
    buried_via_enabled: bool = True
    micro_via_enabled: bool = False
    
    # Layer-specific constraints
    layer_constraints: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Clearance matrix
    clearance_matrix: Dict[ClearanceType, float] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Initialize default netclass and clearance matrix."""
        if "Default" not in self.netclasses:
            self.netclasses["Default"] = NetClass(
                name="Default",
                track_width=self.default_track_width,
                via_diameter=self.default_via_diameter,
                via_drill=self.default_via_drill,
                clearance=self.default_clearance
            )
        
        # Initialize clearance matrix with defaults
        if not self.clearance_matrix:
            self.clearance_matrix = {
                ClearanceType.TRACK_TO_TRACK: self.default_clearance,
                ClearanceType.TRACK_TO_PAD: self.default_clearance,
                ClearanceType.TRACK_TO_VIA: self.default_clearance,
                ClearanceType.VIA_TO_VIA: self.default_clearance,
                ClearanceType.PAD_TO_PAD: self.default_clearance,
                ClearanceType.COPPER_POUR: self.default_clearance * 1.5,
            }
    
    def add_netclass(self, netclass: NetClass) -> None:
        """Add or update a netclass."""
        self.netclasses[netclass.name] = netclass
    
    def get_netclass(self, name: str) -> NetClass:
        """Get netclass by name, fallback to Default."""
        return self.netclasses.get(name, self.netclasses["Default"])
    
    def get_clearance_for_nets(self, net1_class: str, net2_class: str) -> float:
        """Get clearance between two netclasses."""
        nc1 = self.get_netclass(net1_class)
        nc2 = self.get_netclass(net2_class)
        
        # Return the maximum clearance required by either netclass
        return max(nc1.clearance, nc2.clearance)
    
    def get_clearance(self, clearance_type: ClearanceType) -> float:
        """Get clearance for specific constraint type."""
        return self.clearance_matrix.get(clearance_type, self.default_clearance)
    
    def set_clearance(self, clearance_type: ClearanceType, value: float) -> None:
        """Set clearance for specific constraint type."""
        self.clearance_matrix[clearance_type] = value
    
    def get_layer_constraints(self, layer_name: str) -> Dict[str, float]:
        """Get layer-specific constraints."""
        return self.layer_constraints.get(layer_name, {
            'min_track_width': self.min_track_width,
            'min_spacing': self.min_track_spacing,
        })
    
    def set_layer_constraint(self, layer_name: str, constraint: str, value: float) -> None:
        """Set layer-specific constraint."""
        if layer_name not in self.layer_constraints:
            self.layer_constraints[layer_name] = {}
        self.layer_constraints[layer_name][constraint] = value
    
    def validate_track_width(self, width: float, netclass: str = "Default") -> bool:
        """Validate if track width meets DRC requirements."""
        if width < self.min_track_width:
            return False
        
        nc = self.get_netclass(netclass)
        return nc.is_track_width_valid(width)
    
    def validate_via_size(self, diameter: float, drill: float, 
                         netclass: str = "Default") -> bool:
        """Validate if via size meets DRC requirements."""
        if diameter < self.min_via_diameter or drill < self.min_via_drill:
            return False
        
        # Check aspect ratio (assume 1.6mm standard thickness)
        aspect_ratio = 1.6 / drill
        if aspect_ratio > 10:  # Standard manufacturing limit
            return False
        
        nc = self.get_netclass(netclass)
        return nc.is_via_size_valid(diameter)
    
    def validate_clearance(self, distance: float, clearance_type: ClearanceType) -> bool:
        """Validate if clearance distance meets requirements."""
        required_clearance = self.get_clearance(clearance_type)
        return distance >= required_clearance
    
    def get_via_types_allowed(self) -> List[str]:
        """Get list of allowed via types."""
        allowed = ["through"]  # Through vias always allowed
        
        if self.blind_via_enabled:
            allowed.append("blind")
        if self.buried_via_enabled:
            allowed.append("buried")
        if self.micro_via_enabled:
            allowed.append("micro")
        
        return allowed
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert constraints to dictionary for serialization."""
        return {
            'min_track_width': self.min_track_width,
            'min_track_spacing': self.min_track_spacing,
            'min_via_diameter': self.min_via_diameter,
            'min_via_drill': self.min_via_drill,
            'default_track_width': self.default_track_width,
            'default_clearance': self.default_clearance,
            'default_via_diameter': self.default_via_diameter,
            'default_via_drill': self.default_via_drill,
            'netclasses': {name: {
                'name': nc.name,
                'track_width': nc.track_width,
                'via_diameter': nc.via_diameter,
                'via_drill': nc.via_drill,
                'clearance': nc.clearance
            } for name, nc in self.netclasses.items()},
            'blind_via_enabled': self.blind_via_enabled,
            'buried_via_enabled': self.buried_via_enabled,
            'micro_via_enabled': self.micro_via_enabled,
            'layer_constraints': self.layer_constraints,
            'clearance_matrix': {ct.value: val for ct, val in self.clearance_matrix.items()}
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DRCConstraints:
        """Create constraints from dictionary."""
        constraints = cls(
            min_track_width=data.get('min_track_width', 0.1),
            min_track_spacing=data.get('min_track_spacing', 0.1),
            min_via_diameter=data.get('min_via_diameter', 0.2),
            min_via_drill=data.get('min_via_drill', 0.1),
            default_track_width=data.get('default_track_width', 0.25),
            default_clearance=data.get('default_clearance', 0.2),
            default_via_diameter=data.get('default_via_diameter', 0.6),
            default_via_drill=data.get('default_via_drill', 0.3),
            blind_via_enabled=data.get('blind_via_enabled', True),
            buried_via_enabled=data.get('buried_via_enabled', True),
            micro_via_enabled=data.get('micro_via_enabled', False),
        )
        
        # Load netclasses
        for name, nc_data in data.get('netclasses', {}).items():
            constraints.add_netclass(NetClass(
                name=nc_data['name'],
                track_width=nc_data['track_width'],
                via_diameter=nc_data['via_diameter'],
                via_drill=nc_data['via_drill'],
                clearance=nc_data['clearance']
            ))
        
        # Load clearance matrix
        clearance_data = data.get('clearance_matrix', {})
        for ct_name, value in clearance_data.items():
            try:
                clearance_type = ClearanceType(ct_name)
                constraints.set_clearance(clearance_type, value)
            except ValueError:
                pass  # Skip unknown clearance types
        
        return constraints