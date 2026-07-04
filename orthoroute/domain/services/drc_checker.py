"""Domain service for design rule checking."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Dict, Tuple, Optional

from ..models.board import Board, Coordinate, Pad, Component
from ..models.routing import Route, Segment, Via
from ..models.constraints import DRCConstraints, ClearanceType


@dataclass
class DRCViolation:
    """Represents a design rule violation."""
    type: str
    severity: str  # 'error', 'warning', 'info'
    message: str
    location: Optional[Coordinate] = None
    net_id: Optional[str] = None
    component_id: Optional[str] = None
    
    def __str__(self) -> str:
        location_str = f" at ({self.location.x:.3f}, {self.location.y:.3f})" if self.location else ""
        return f"{self.severity.upper()}: {self.message}{location_str}"


class DRCChecker:
    """Domain service for design rule checking."""
    
    def __init__(self, constraints: DRCConstraints) -> None:
        """Initialize DRC checker with constraints."""
        self.constraints: DRCConstraints = constraints
    
    def check_board(self, board: Board) -> List[DRCViolation]:
        """Perform comprehensive DRC check on board."""
        violations = []
        
        # Check component placement
        violations.extend(self._check_component_spacing(board))
        
        # Check pad-to-pad clearances
        violations.extend(self._check_pad_clearances(board))
        
        # Check nets for basic issues
        violations.extend(self._check_net_integrity(board))
        
        return violations
    
    def check_route(self, route: Route, board: Board) -> List[DRCViolation]:
        """Check a specific route for DRC violations."""
        violations = []
        
        # Check track widths
        violations.extend(self._check_track_widths(route))
        
        # Check via sizes
        violations.extend(self._check_via_sizes(route))
        
        # Check track-to-track clearances within route
        violations.extend(self._check_route_self_clearance(route))
        
        # Check route connectivity
        violations.extend(self._check_route_connectivity(route))
        
        return violations
    
    def check_routes_clearance(self, routes: List[Route]) -> List[DRCViolation]:
        """Check clearances between different routes."""
        violations = []
        
        for i, route1 in enumerate(routes):
            for route2 in routes[i+1:]:
                violations.extend(self._check_inter_route_clearance(route1, route2))
        
        return violations
    
    def _check_component_spacing(self, board: Board) -> List[DRCViolation]:
        """Check spacing between components."""
        violations = []
        min_component_spacing = 0.5  # mm - typical minimum
        
        for i, comp1 in enumerate(board.components):
            for comp2 in board.components[i+1:]:
                if comp1.layer != comp2.layer:
                    continue  # Different layers don't need spacing check
                
                bounds1 = comp1.get_bounds()
                bounds2 = comp2.get_bounds()
                
                # Calculate minimum distance between component bounds
                distance = self._calculate_bounds_distance(bounds1, bounds2)
                
                if distance < min_component_spacing:
                    violations.append(DRCViolation(
                        type="component_spacing",
                        severity="warning",
                        message=f"Components {comp1.reference} and {comp2.reference} "
                               f"too close ({distance:.3f}mm < {min_component_spacing:.3f}mm)",
                        location=bounds1.center,
                        component_id=comp1.id
                    ))
        
        return violations
    
    def _check_pad_clearances(self, board: Board) -> List[DRCViolation]:
        """Check pad-to-pad clearances."""
        violations = []
        all_pads = board.get_all_pads()
        
        for i, pad1 in enumerate(all_pads):
            for pad2 in all_pads[i+1:]:
                if pad1.layer != pad2.layer:
                    continue  # Different layers
                
                if pad1.net_id == pad2.net_id and pad1.net_id is not None:
                    continue  # Same net, no clearance required
                
                distance = pad1.position.distance_to(pad2.position)
                pad_clearance = pad1.size[0]/2 + pad2.size[0]/2  # Simplified circular assumption
                
                required_clearance = self.constraints.get_clearance(ClearanceType.PAD_TO_PAD)
                total_required = pad_clearance + required_clearance
                
                if distance < total_required:
                    violations.append(DRCViolation(
                        type="pad_clearance",
                        severity="error",
                        message=f"Pad clearance violation: {distance:.3f}mm < {total_required:.3f}mm required",
                        location=pad1.position,
                        net_id=pad1.net_id
                    ))
        
        return violations
    
    def _check_net_integrity(self, board: Board) -> List[DRCViolation]:
        """Check net integrity issues."""
        violations = []
        
        for net in board.nets:
            # Check for nets with single pad
            if len(net.pads) == 1:
                violations.append(DRCViolation(
                    type="single_pad_net",
                    severity="warning",
                    message=f"Net {net.name} has only one pad",
                    net_id=net.id
                ))
            
            # Check for nets with no pads
            elif len(net.pads) == 0:
                violations.append(DRCViolation(
                    type="empty_net",
                    severity="error",
                    message=f"Net {net.name} has no pads",
                    net_id=net.id
                ))
        
        return violations
    
    def _check_track_widths(self, route: Route) -> List[DRCViolation]:
        """Check track width compliance."""
        violations = []
        
        for segment in route.segments:
            if segment.type.value != "track":
                continue
            
            # Get netclass for this route
            net = route.net_id  # This would need to be resolved to actual net
            netclass_name = "Default"  # Would get from net
            
            if not self.constraints.validate_track_width(segment.width, netclass_name):
                violations.append(DRCViolation(
                    type="track_width",
                    severity="error",
                    message=f"Track width {segment.width:.3f}mm violates DRC rules",
                    location=segment.start,
                    net_id=route.net_id
                ))
        
        return violations
    
    def _check_via_sizes(self, route: Route) -> List[DRCViolation]:
        """Check via size compliance."""
        violations = []
        
        for via in route.vias:
            netclass_name = "Default"  # Would get from net
            
            if not self.constraints.validate_via_size(via.diameter, via.drill_size, netclass_name):
                violations.append(DRCViolation(
                    type="via_size",
                    severity="error",
                    message=f"Via size {via.diameter:.3f}mm/{via.drill_size:.3f}mm violates DRC rules",
                    location=via.position,
                    net_id=route.net_id
                ))
        
        return violations
    
    def _check_route_self_clearance(self, route: Route) -> List[DRCViolation]:
        """Check clearances within a single route."""
        violations = []
        
        # Check segment-to-segment clearances (parallel tracks)
        for i, seg1 in enumerate(route.segments):
            for seg2 in route.segments[i+2:]:  # Skip adjacent segments
                if seg1.layer != seg2.layer:
                    continue
                
                distance = self._calculate_segment_distance(seg1, seg2)
                required_clearance = self.constraints.get_clearance(ClearanceType.TRACK_TO_TRACK)
                
                if distance < required_clearance:
                    violations.append(DRCViolation(
                        type="track_clearance",
                        severity="error",
                        message=f"Track-to-track clearance violation within route: "
                               f"{distance:.3f}mm < {required_clearance:.3f}mm",
                        location=seg1.start,
                        net_id=route.net_id
                    ))
        
        return violations
    
    def _check_route_connectivity(self, route: Route) -> List[DRCViolation]:
        """Check route connectivity."""
        violations = []
        
        connectivity_issues = route.validate_connectivity()
        for issue in connectivity_issues:
            violations.append(DRCViolation(
                type="connectivity",
                severity="error",
                message=issue,
                net_id=route.net_id
            ))
        
        return violations
    
    def _check_inter_route_clearance(self, route1: Route, route2: Route) -> List[DRCViolation]:
        """Check clearance between two different routes."""
        violations = []
        
        if route1.net_id == route2.net_id:
            return violations  # Same net, no clearance needed
        
        # Check track-to-track clearances
        for seg1 in route1.segments:
            for seg2 in route2.segments:
                if seg1.layer != seg2.layer:
                    continue
                
                distance = self._calculate_segment_distance(seg1, seg2)
                required_clearance = self.constraints.get_clearance(ClearanceType.TRACK_TO_TRACK)
                
                if distance < required_clearance:
                    violations.append(DRCViolation(
                        type="inter_route_clearance",
                        severity="error",
                        message=f"Clearance violation between nets {route1.net_id} and {route2.net_id}: "
                               f"{distance:.3f}mm < {required_clearance:.3f}mm",
                        location=seg1.start
                    ))
        
        return violations
    
    def _calculate_bounds_distance(self, bounds1: Any, bounds2: Any) -> float:
        """Calculate minimum distance between two rectangular bounds."""
        # Simplified distance calculation
        dx = max(0, max(bounds1.min_x - bounds2.max_x, bounds2.min_x - bounds1.max_x))
        dy = max(0, max(bounds1.min_y - bounds2.max_y, bounds2.min_y - bounds1.max_y))
        return (dx * dx + dy * dy) ** 0.5
    
    def _calculate_segment_distance(self, seg1: Segment, seg2: Segment) -> float:
        """Calculate minimum distance between two line segments."""
        # Simplified: use distance between start points
        # Real implementation would calculate true line-to-line distance
        return seg1.start.distance_to(seg2.start)
    
    def generate_drc_report(self, violations: List[DRCViolation]) -> Dict[str, Any]:
        """Generate a comprehensive DRC report."""
        report = {
            'total_violations': len(violations),
            'errors': len([v for v in violations if v.severity == 'error']),
            'warnings': len([v for v in violations if v.severity == 'warning']),
            'violations_by_type': {},
            'violations_by_net': {},
            'violations': [str(v) for v in violations]
        }
        
        # Group by type
        for violation in violations:
            vtype = violation.type
            if vtype not in report['violations_by_type']:
                report['violations_by_type'][vtype] = 0
            report['violations_by_type'][vtype] += 1
        
        # Group by net
        for violation in violations:
            if violation.net_id:
                if violation.net_id not in report['violations_by_net']:
                    report['violations_by_net'][violation.net_id] = 0
                report['violations_by_net'][violation.net_id] += 1
        
        return report