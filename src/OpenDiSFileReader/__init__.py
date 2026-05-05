### OpenDiS File Reader ####
# File reader for the OpenDiS data format

import copy
import functools
import operator
from collections.abc import Generator
from dataclasses import dataclass
from io import TextIOWrapper
from typing import Any, Callable

import numpy as np
from ovito.data import DataCollection, ParticleType, SimulationCell
from ovito.io import FileReaderInterface
from ovito.traits import OvitoObject
from ovito.vis import LinesVis


@dataclass
class Arm:
    arm_tag: int  # node_tag of the neighbor this arm connects to
    bvec: list[float]  # Burgers vector of this arm
    nvec: list[float]  # slip plane normal vector of this arm


@dataclass
class Node:
    node_tag: int
    pos: list[float]
    num_arms: int
    arms: list[Arm]
    constrain: int
    processed: bool = False  # set during line tracing to avoid revisiting


@dataclass
class Line:
    # Each segment is a list of nodes forming one arm-chain from a junction outward.
    segments: list[list[Node]]


class OpenDiSFileReader(FileReaderInterface):
    lines_vis = OvitoObject(
        LinesVis, shading=LinesVis.Shading.Normal, wrapped_lines=True
    )
    particle_type = OvitoObject(ParticleType, name="Node")

    @staticmethod
    def detect(filename: str) -> bool:
        # OpenDiS data files always start with "dataFileVersion = ..."
        try:
            with open(filename, "r") as f:
                line = f.readline()
                return line.strip().startswith("dataFileVersion =")
        except OSError:
            return False

    def scan(self, filename: str, register_frame: Callable[..., None]) -> None:
        # OpenDiS files contain a single static snapshot
        register_frame(frame_info=(0, 0))

    @staticmethod
    def skip_line(line: str) -> bool:
        return line.startswith("#") or not line.strip()

    @staticmethod
    def parse_number(number: str) -> int | float:
        try:
            return int(number)
        except ValueError:
            return float(number)

    @staticmethod
    def read_array(f: TextIOWrapper) -> list[int | float]:
        # Reads one value per line until a line containing "]" is encountered.
        # Called after the opening "[" has already been consumed by parse_header.
        array = []
        line = f.readline().strip()
        while "]" not in line:
            if line.startswith("#") or not line:
                continue
            array.append(__class__.parse_number(line))
            line = f.readline().strip()
        return array

    @staticmethod
    def parse_header(f: TextIOWrapper) -> dict[str, Any]:
        key = None
        header_end = "END OF DATA FILE PARAMETERS"
        header = {}
        while line := f.readline():
            if header_end in line:
                return header

            if __class__.skip_line(line):
                continue

            if "=" in line:
                key = line.split("=")[0].strip()

            if "[" in line:
                # Value is a multi-line bracketed array; read until closing "]"
                array = __class__.read_array(f)
                assert key is not None
                header[key] = array
            else:
                assert key is not None
                header[key] = __class__.parse_number(line.split("=")[-1].strip())
        return header

    @staticmethod
    def parse_domain_decomposition(
        f: TextIOWrapper, num_domains: int
    ) -> list[list[float | int]]:
        data = []
        while len(data) < num_domains:
            line = f.readline().strip()
            if __class__.skip_line(line):
                continue
            tokens = line.split()
            data.append([__class__.parse_number(t) for t in tokens])
        return data

    @staticmethod
    def parse_primary_line(line: str) -> Node:
        # Lines may be prefixed with a domain tag: "domain,tag  x y z ..."
        # Split on "," and take the last part to strip any domain prefix.
        line = line.strip().split(",")[-1]
        tokens = line.split()
        return Node(
            int(tokens[0]),
            [float(t) for t in tokens[1:4]],
            int(tokens[4]),
            [],
            int(tokens[5]),
        )

    @staticmethod
    def parse_secondary_line(f: TextIOWrapper, num_arms: int) -> list[Arm]:
        # Arms are stored in "secodary" lines, one per line arm and *num_arms* entries.
        # Each arm's data consists of: arm_tag, bvec (3 floats), nvec (3 floats).
        # This function reads all arms belonging to a node and returns them as a list
        arms = []
        current_arm = []
        while len(arms) < num_arms:
            line = f.readline()
            if __class__.skip_line(line):
                continue
            if "," in line:
                line = line.strip().split(",")[-1]

            tokens = line.split()

            if len(current_arm) == 0:
                # First token on the first line of an arm entry is the neighbor tag
                current_arm.append(int(tokens[0]))
                tokens = tokens[1:]
            assert len(tokens) == 3 or len(tokens) == 6
            while tokens:
                current_arm.append([float(t) for t in tokens[:3]])
                tokens = tokens[3:]

            assert len(current_arm) <= 3
            if len(current_arm) == 3:
                arms.append(Arm(*current_arm))
                current_arm = []
        return arms

    @staticmethod
    def parse_nodal_data(f: TextIOWrapper) -> list[Node]:
        # Node records alternate between a primary line (tag, position, num_arms, constraint)
        # and secondary lines (one entry per arm with bvec and nvec).
        data = []
        primary_line = True
        while line := f.readline():
            if __class__.skip_line(line):
                continue

            if primary_line:
                data.append(__class__.parse_primary_line(line))
                primary_line = False
            if not primary_line:
                node = data[-1]
                node.arms = __class__.parse_secondary_line(f, node.num_arms)
                primary_line = True
        return data

    @staticmethod
    def parse_body(f: TextIOWrapper, num_domains: int) -> dict[str, Any]:
        key = None
        body = {}
        while line := f.readline():
            if __class__.skip_line(line):
                continue

            if "=" in line:
                key = line.split("=")[0].strip()

            if key == "domainDecomposition":
                body[key] = __class__.parse_domain_decomposition(f, num_domains)

            if key == "nodalData":
                body[key] = __class__.parse_nodal_data(f)
        return body

    @staticmethod
    def point_in_cell(cell: SimulationCell, point: np.ndarray) -> bool:
        if np.any(point < cell[:, 3]):
            return False
        for i in range(3):
            if point[i] >= cell[i, 3] + cell[i, i]:
                return False
        return True

    @staticmethod
    def get_next_start_node(
        nodes: list[Node], start: int
    ) -> tuple[int, Node] | tuple[None, None]:
        while start < len(nodes):
            # Any node that is not in a chain, either junction or end point
            if nodes[start].num_arms != 2 and not nodes[start].processed:
                return start, nodes[start]
            start += 1
        return None, None

    @staticmethod
    def trace_path(start: Node, arm: Arm, node_dict: dict[int, Node]) -> list[Node]:
        # Follow a chain of degree-2 nodes from start through arm until reaching
        # a junction (num_arms != 2) or an already-processed node.
        segment = [start]
        prev_tag = start.node_tag
        curr = node_dict[arm.arm_tag]
        while curr.num_arms == 2 and not curr.processed:
            segment.append(curr)
            curr.processed = True
            for next_arm in curr.arms:
                if next_arm.arm_tag != prev_tag:
                    prev_tag = curr.node_tag
                    curr = node_dict[next_arm.arm_tag]
                    break
        segment.append(curr)
        return segment

    @staticmethod
    def walk_lines(nodes: list[Node]) -> Generator[float, None, list[Line]]:
        # Returns the collected Line objects; callers must use "yield from" to
        # receive the return value while forwarding progress yields upstream.
        node_dict = {node.node_tag: node for node in nodes}

        lines = []
        start, start_node = __class__.get_next_start_node(nodes, 0)
        while start_node is not None and start is not None:
            yield 0.0
            start_node.processed = True
            segments = []
            for arm in start_node.arms:
                if not node_dict[arm.arm_tag].processed:
                    segments.append(__class__.trace_path(start_node, arm, node_dict))
            if segments:
                lines.append(Line(segments))
            start, start_node = __class__.get_next_start_node(nodes, start)
        return lines

    @staticmethod
    def walk_line(
        segment: list[Node],
        ref_point: np.ndarray,
        cell: SimulationCell,
        positions: list[np.ndarray],
        sections: list[int],
        bvecs: list[np.ndarray],
        nvecs: list[np.ndarray],
        counter: int,
    ) -> Generator[float, None, np.ndarray]:
        # ref_point carries the last unwrapped position across calls so that
        # delta_vector can resolve PBC images consistently along the full path.
        for node_id in range(1, len(segment)):
            yield 0.0
            n0 = segment[node_id - 1]
            n1 = segment[node_id]

            # delta_vector returns the shortest-image displacement, respecting PBC
            p0 = ref_point + cell.delta_vector(ref_point, n0.pos)
            p1 = p0 + cell.delta_vector(p0, n1.pos)
            ref_point = p1

            # Avoid duplicating the shared start point when appending the next
            # segment from the same junction (sections[-1] already equals counter).
            new_segment = len(sections) == 0 or sections[-1] != counter

            if new_segment:
                positions.append(p0)
                sections.append(counter)
            positions.append(p1)
            sections.append(counter)

            # Burgers vector and normal come from the arm in n0 that points to n1
            matching_arm = None
            for arm in n0.arms:
                if arm.arm_tag == n1.node_tag:
                    matching_arm = arm
                    break
            if matching_arm is None:
                raise Exception(
                    "Could not find matching arm for node {}".format(n1.node_tag)
                )
            if new_segment:
                bvecs.append(matching_arm.bvec)
                nvecs.append(matching_arm.nvec)
            bvecs.append(matching_arm.bvec)
            nvecs.append(matching_arm.nvec)

        return ref_point

    def parse(self, data: DataCollection, filename: str, **kwargs: Any) -> Generator[str | float, None, None] | None:  # type: ignore[override]

        with open(filename, "r") as f:
            header = __class__.parse_header(f)
            num_domains = np.prod(header["dataDecompGeometry"])
            body = __class__.parse_body(f, num_domains)
            assert len(body["nodalData"]) == header["nodeCount"]

        for k, v in header.items():
            data.attributes[k] = v

        cell = np.zeros((3, 4))
        cell[:, 3] = header["minCoordinates"]
        for i in range(3):
            cell[i, i] = header["maxCoordinates"][i] - header["minCoordinates"][i]
        cell = data.create_cell(cell, pbc=(True, True, True))

        # Scale line/node width to ~0.5 % of the cell diagonal for visual clarity
        self.lines_vis.width = 5 * np.linalg.norm(cell[:3, :3].diagonal()) / 1000

        particles = data.create_particles(count=header["nodeCount"])
        identifier = particles.create_property("Particle Identifier")
        particle_type = particles.create_property("Particle Type")
        positions = particles.create_property("Position")
        num_arms = particles.create_property("Num Arms", dtype=int)
        constraint = particles.create_property("Constraint", dtype=int)

        self.particle_type.radius = self.lines_vis.width / 2
        particle_type[:] = self.particle_type.id
        particle_type.types_.append(copy.deepcopy(self.particle_type))

        for i, node in enumerate(body["nodalData"]):
            identifier[i] = node.node_tag
            positions[i] = node.pos
            num_arms[i] = node.num_arms
            constraint[i] = node.constrain
            yield i / len(body["nodalData"])

        positions = []
        sections = []
        bvecs = []
        nvecs = []
        counter = 0

        # Line objects returned by the generator via its StopIteration value.
        lines = yield from self.walk_lines(body["nodalData"])

        ref_point = np.asarray(data.cell[:, 3])
        for line in lines:
            ref_point = data.cell.wrap_point(ref_point)
            for segment in line.segments:
                # Reverse so the segment walks from the far end back to the junction,
                # keeping ref_point continuous across consecutive segments of the line.
                segment = list(reversed(segment))
                ref_point = yield from self.walk_line(
                    segment,
                    ref_point,
                    cell,
                    positions,
                    sections,
                    bvecs,
                    nvecs,
                    counter,
                )
            counter += 1

        self.lines_vis.color = self.particle_type.color

        lines = data.lines.create("Arms", vis=self.lines_vis)
        lines.create_property("Position", data=positions)
        lines.create_property("Section", data=sections)
        lines.create_property("Burgers vector", data=bvecs)
        lines.create_property(
            "Burgers vector magnitude", data=np.linalg.norm(bvecs, axis=1)
        )
        lines.create_property("Normal vector", data=nvecs)
