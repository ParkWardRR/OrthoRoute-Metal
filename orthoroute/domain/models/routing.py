"""Domain models for routing structures."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Set, Dict, Any
from enum import Enum
from uuid import uuid4
import math

from .board import Coordinate


class SegmentType(Enum):
    """Types of routing segments."""
    TRACK = "track"
    ARC = "arc"
    VIA = "via"


class ViaType(Enum):
    """Types of vias."""
    THROUGH = "through"
    BLIND = "blind"
    BURIED = "buried"
    MICRO = "micro"


@dataclass(frozen=True)
class Segment:
    """Value object representing a routing segment."""
    type: SegmentType
    start: Coordinate
    end: Coordinate
    width: float
    layer: str
    net_id: str
    
    @property
    def length(self) -> float:
        """Calculate segment length."""
        if self.type == SegmentType.TRACK:
            return self.start.distance_to(self.end)
        return 0.0  # Vias have no length
    
    @property
    def is_horizontal(self) -> bool:
        """Check if segment is horizontal (within tolerance)."""
        return abs(self.start.y - self.end.y) < 0.001
    
    @property
    def is_vertical(self) -> bool:
        """Check if segment is vertical (within tolerance)."""
        return abs(self.start.x - self.end.x) < 0.001
    
    @property
    def is_manhattan(self) -> bool:
        """Check if segment follows Manhattan routing (horizontal or vertical)."""
        return self.is_horizontal or self.is_vertical


@dataclass(frozen=True)
class Via:
    """Value object representing a via."""
    position: Coordinate
    diameter: float
    drill_size: float
    from_layer: str
    to_layer: str
    net_id: str
    via_type: ViaType = ViaType.THROUGH
    
    @property
    def aspect_ratio(self) -> float:
        """Calculate via aspect ratio (board thickness / drill size)."""
        # This would need board thickness information
        # For now, assume standard 1.6mm thickness
        return 1.6 / self.drill_size


@dataclass
class Route:
    """Domain entity representing a complete route for a net."""
    id: str
    net_id: str
    segments: List[Segment] = field(default_factory=list)
    vias: List[Via] = field(default_factory=list)
    
    def __post_init__(self) -> None:
        if not self.id:
            self.id = str(uuid4())
    
    @property
    def total_length(self) -> float:
        """Calculate total route length."""
        return sum(segment.length for segment in self.segments)
    
    @property
    def layers_used(self) -> Set[str]:
        """Get all layers used by this route."""
        layers = set()
        for segment in self.segments:
            layers.add(segment.layer)
        for via in self.vias:
            layers.add(via.from_layer)
            layers.add(via.to_layer)
        return layers
    
    @property
    def via_count(self) -> int:
        """Get number of vias in this route."""
        return len(self.vias)
    
    def add_segment(self, segment: Segment) -> None:
        """Add a segment to the route."""
        if segment.net_id != self.net_id:
            raise ValueError(f"Segment net_id {segment.net_id} doesn't match route net_id {self.net_id}")
        self.segments.append(segment)
    
    def add_via(self, via: Via) -> None:
        """Add a via to the route."""
        if via.net_id != self.net_id:
            raise ValueError(f"Via net_id {via.net_id} doesn't match route net_id {self.net_id}")
        self.vias.append(via)
    
    def is_manhattan_compliant(self) -> bool:
        """Check if route follows Manhattan routing rules."""
        return all(segment.is_manhattan for segment in self.segments)
    
    def validate_connectivity(self) -> List[str]:
        """Validate that route segments are properly connected."""
        issues = []
        
        if len(self.segments) < 1:
            return issues  # Empty route is valid
        
        # Check segment connectivity
        for i in range(len(self.segments) - 1):
            current = self.segments[i]
            next_seg = self.segments[i + 1]
            
            # Check if segments connect or if there's a via between them
            connects_directly = (
                current.end.distance_to(next_seg.start) < 0.001 or
                current.start.distance_to(next_seg.end) < 0.001
            )
            
            # Check for via at connection point
            via_at_connection = any(
                via.position.distance_to(current.end) < 0.001 and
                via.from_layer == current.layer and
                via.to_layer == next_seg.layer
                for via in self.vias
            )
            
            if not connects_directly and not via_at_connection:
                issues.append(f"Segments {i} and {i+1} are not connected")
        
        return issues
    
    def get_route_statistics(self) -> Dict[str, Any]:
        """Get comprehensive route statistics."""
        return {
            'total_length': self.total_length,
            'segment_count': len(self.segments),
            'via_count': self.via_count,
            'layers_used': list(self.layers_used),
            'is_manhattan': self.is_manhattan_compliant(),
            'connectivity_issues': len(self.validate_connectivity())
        }


@dataclass
class RoutingResult:
    """Value object representing the result of a routing operation."""
    success: bool
    route: Optional[Route] = None
    error_message: Optional[str] = None
    execution_time: float = 0.0
    memory_used: float = 0.0
    algorithm_used: str = ""
    
    @classmethod
    def success_result(cls, route: Route, execution_time: float = 0.0, 
                      algorithm: str = "") -> RoutingResult:
        """Create a successful routing result."""
        return cls(
            success=True,
            route=route,
            execution_time=execution_time,
            algorithm_used=algorithm
        )
    
    @classmethod
    def failure_result(cls, error_message: str, execution_time: float = 0.0,
                      algorithm: str = "") -> RoutingResult:
        """Create a failed routing result."""
        return cls(
            success=False,
            error_message=error_message,
            execution_time=execution_time,
            algorithm_used=algorithm
        )


@dataclass
class RoutingStatistics:
    """Value object containing routing session statistics."""
    nets_attempted: int = 0
    nets_routed: int = 0
    nets_failed: int = 0
    total_length: float = 0.0
    total_vias: int = 0
    total_time: float = 0.0
    memory_peak: float = 0.0
    algorithm_used: str = ""
    
    @property
    def success_rate(self) -> float:
        """Calculate routing success rate."""
        if self.nets_attempted == 0:
            return 0.0
        return self.nets_routed / self.nets_attempted
    
    @property
    def average_length(self) -> float:
        """Calculate average route length."""
        if self.nets_routed == 0:
            return 0.0
        return self.total_length / self.nets_routed
    
    @property
    def average_vias_per_net(self) -> float:
        """Calculate average vias per routed net."""
        if self.nets_routed == 0:
            return 0.0
        return self.total_vias / self.nets_routed
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert statistics to dictionary."""
        return {
            'nets_attempted': self.nets_attempted,
            'nets_routed': self.nets_routed,
            'nets_failed': self.nets_failed,
            'success_rate': self.success_rate,
            'total_length_mm': self.total_length,
            'average_length_mm': self.average_length,
            'total_vias': self.total_vias,
            'average_vias_per_net': self.average_vias_per_net,
            'total_time_seconds': self.total_time,
            'memory_peak_mb': self.memory_peak,
            'algorithm': self.algorithm_used
        }