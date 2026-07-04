"""Lattice3D builder tests — validates grid topology, H/V discipline, and via connectivity."""
import pytest
import numpy as np

from orthoroute.algorithms.manhattan.unified_pathfinder import Lattice3D, CSRGraph


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_lattice():
    """5×5 grid, 4 layers, pitch=0.4mm. bounds chosen so x_steps=5, y_steps=5."""
    bounds = (0.0, 0.0, 1.6, 1.6)  # (1.6-0.0)/0.4+1 = 5 steps
    return Lattice3D(bounds, pitch=0.4, layers=4)


@pytest.fixture
def six_layer_lattice():
    """10×10 grid, 6 layers — the standard PCB stack."""
    bounds = (0.0, 0.0, 3.6, 3.6)  # 10 steps each axis
    return Lattice3D(bounds, pitch=0.4, layers=6)


@pytest.fixture
def small_graph(small_lattice):
    """Build the CSR graph for the 5×5×4 lattice."""
    return small_lattice.build_graph(via_cost=1.0, use_gpu=False)


# ---------------------------------------------------------------------------
# Node-count and GID round-trip
# ---------------------------------------------------------------------------

class TestNodeCount:
    def test_node_count_matches_dimensions(self, small_lattice):
        """Lattice3D.num_nodes must equal Nx * Ny * Nz."""
        expected = small_lattice.x_steps * small_lattice.y_steps * small_lattice.layers
        assert small_lattice.num_nodes == expected

    def test_six_layer_node_count(self, six_layer_lattice):
        """Six-layer lattice should have Nx*Ny*6 nodes."""
        lat = six_layer_lattice
        assert lat.num_nodes == lat.x_steps * lat.y_steps * 6


class TestGIDRoundTrip:
    def test_gid_roundtrip(self, small_lattice):
        """node_idx → idx_to_coord → node_idx must be an identity."""
        lat = small_lattice
        for z in range(lat.layers):
            for y in range(lat.y_steps):
                for x in range(lat.x_steps):
                    gid = lat.node_idx(x, y, z)
                    rx, ry, rz = lat.idx_to_coord(gid)
                    assert (rx, ry, rz) == (x, y, z), (
                        f"Round-trip failed for ({x},{y},{z}): "
                        f"got ({rx},{ry},{rz}) via gid={gid}"
                    )

    def test_gid_unique(self, small_lattice):
        """Every (x,y,z) triple must map to a unique global ID."""
        lat = small_lattice
        seen = set()
        for z in range(lat.layers):
            for y in range(lat.y_steps):
                for x in range(lat.x_steps):
                    gid = lat.node_idx(x, y, z)
                    assert gid not in seen, f"Duplicate gid {gid} for ({x},{y},{z})"
                    seen.add(gid)
        assert len(seen) == lat.num_nodes


# ---------------------------------------------------------------------------
# Manhattan adjacency & H/V discipline
# ---------------------------------------------------------------------------

class TestManhattanAdjacency:
    def test_manhattan_adjacency_no_diagonals(self, small_lattice, small_graph):
        """Every CSR edge must connect ±1 in exactly one axis (no diagonals)."""
        lat = small_lattice
        g = small_graph
        indptr = g.indptr if isinstance(g.indptr, np.ndarray) else g.indptr.get()
        indices = g.indices if isinstance(g.indices, np.ndarray) else g.indices.get()

        for u in range(lat.num_nodes):
            ux, uy, uz = lat.idx_to_coord(u)
            for ei in range(int(indptr[u]), int(indptr[u + 1])):
                v = int(indices[ei])
                vx, vy, vz = lat.idx_to_coord(v)
                dx = abs(vx - ux)
                dy = abs(vy - uy)
                dz = abs(vz - uz)
                # Must differ in exactly ONE axis
                changes = (dx > 0) + (dy > 0) + (dz > 0)
                assert changes == 1, (
                    f"Non-Manhattan edge: ({ux},{uy},{uz}) -> ({vx},{vy},{vz}), "
                    f"dx={dx} dy={dy} dz={dz}"
                )
                # Planar edges must be step-1; via edges check layer distance
                if dz == 0:
                    assert dx + dy == 1, f"Planar edge step > 1"

    def test_layer_discipline(self, small_lattice, small_graph):
        """H-layer edges move only in X; V-layer edges move only in Y.

        Layer direction assignment:
          z=0 (F.Cu) → 'v' (vertical / Y-only)
          z=1 (In1)  → 'h' (horizontal / X-only)
          z=2 (In2)  → 'v'
          z=3 (In3)  → 'h'
        Outer layers (0, layers-1) have NO lateral edges in this build.
        """
        lat = small_lattice
        g = small_graph
        indptr = g.indptr if isinstance(g.indptr, np.ndarray) else g.indptr.get()
        indices = g.indices if isinstance(g.indices, np.ndarray) else g.indices.get()

        for u in range(lat.num_nodes):
            ux, uy, uz = lat.idx_to_coord(u)
            for ei in range(int(indptr[u]), int(indptr[u + 1])):
                v = int(indices[ei])
                vx, vy, vz = lat.idx_to_coord(v)
                if uz != vz:
                    continue  # skip via edges
                direction = lat.get_legal_axis(uz)
                if direction == 'h':
                    assert vy == uy, (
                        f"H-layer {uz} edge has Y movement: "
                        f"({ux},{uy},{uz}) -> ({vx},{vy},{vz})"
                    )
                else:  # 'v'
                    assert vx == ux, (
                        f"V-layer {uz} edge has X movement: "
                        f"({ux},{uy},{uz}) -> ({vx},{vy},{vz})"
                    )


# ---------------------------------------------------------------------------
# Via edges
# ---------------------------------------------------------------------------

class TestViaEdges:
    def test_via_edges_connect_layers(self, small_lattice, small_graph):
        """Via edges must only exist between different layers at the same (x,y)."""
        lat = small_lattice
        g = small_graph
        indptr = g.indptr if isinstance(g.indptr, np.ndarray) else g.indptr.get()
        indices = g.indices if isinstance(g.indices, np.ndarray) else g.indices.get()

        for u in range(lat.num_nodes):
            ux, uy, uz = lat.idx_to_coord(u)
            for ei in range(int(indptr[u]), int(indptr[u + 1])):
                v = int(indices[ei])
                vx, vy, vz = lat.idx_to_coord(v)
                if uz == vz:
                    continue  # planar edge
                # Via edge: must have same (x,y)
                assert (vx, vy) == (ux, uy), (
                    f"Via edge changes (x,y): ({ux},{uy},{uz}) -> ({vx},{vy},{vz})"
                )

    def test_via_edges_legal_pairs(self, small_lattice, small_graph):
        """Every via edge must connect a pair in the legal_via_pairs set."""
        lat = small_lattice
        g = small_graph
        legal = lat.get_legal_via_pairs(lat.layers)
        indptr = g.indptr if isinstance(g.indptr, np.ndarray) else g.indptr.get()
        indices = g.indices if isinstance(g.indices, np.ndarray) else g.indices.get()

        for u in range(lat.num_nodes):
            ux, uy, uz = lat.idx_to_coord(u)
            for ei in range(int(indptr[u]), int(indptr[u + 1])):
                v = int(indices[ei])
                vx, vy, vz = lat.idx_to_coord(v)
                if uz == vz:
                    continue
                assert (int(uz), int(vz)) in legal or (int(vz), int(uz)) in legal, (
                    f"Illegal via pair ({uz},{vz}) at ({ux},{uy})"
                )


# ---------------------------------------------------------------------------
# Boundary nodes
# ---------------------------------------------------------------------------

class TestBoundaryNodes:
    def test_boundary_nodes_fewer_neighbors(self, small_lattice, small_graph):
        """Corner and edge nodes must have strictly fewer neighbors than interior nodes."""
        lat = small_lattice
        g = small_graph
        indptr = g.indptr if isinstance(g.indptr, np.ndarray) else g.indptr.get()

        # Pick an interior node on an internal layer that has lateral edges
        # Layers 1..layers-2 have lateral edges
        interior_z = 1
        interior_x = lat.x_steps // 2
        interior_y = lat.y_steps // 2
        interior_gid = lat.node_idx(interior_x, interior_y, interior_z)
        interior_deg = int(indptr[interior_gid + 1]) - int(indptr[interior_gid])

        # Corner node on same layer
        corner_gid = lat.node_idx(0, 0, interior_z)
        corner_deg = int(indptr[corner_gid + 1]) - int(indptr[corner_gid])

        assert corner_deg < interior_deg, (
            f"Corner degree {corner_deg} should be < interior degree {interior_deg}"
        )
