#### OpenDiS File Reader ####
# File reader for the OpenDiS data format

import functools
import operator
from dataclasses import dataclass
from io import TextIOWrapper
from typing import Callable

import numpy as np
from ovito.data import DataCollection, SimulationCell
from ovito.io import FileReaderInterface
from ovito.traits import OvitoObject
from ovito.vis import LinesVis


@dataclass
class Arm:
    arm_tag: int
    bvec: list[float]
    nvec: list[float]


@dataclass
class Node:
    node_tag: int
    pos: list[float]
    num_arms: int
    arms: list[Arm]
    constrain: int
    processed: bool = False


@dataclass
class Line:
    segments: list[list[Node], list[Node]]


class OpenDiSFileReader(FileReaderInterface):
    lines_vis = OvitoObject(
        LinesVis, shading=LinesVis.Shading.Normal, wrapped_lines=True
    )

    @staticmethod
    def detect(filename: str):
        try:
            with open(filename, "r") as f:
                line = f.readline()
                return line.strip().startswith("dataFileVersion =")
        except OSError:
            return False

    def scan(self, filename: str, register_frame: Callable[..., None]):
        register_frame(frame_info=(0, 0))

    @staticmethod
    def skip_line(line: str):
        return line.startswith("#") or not line.strip()

    @staticmethod
    def parse_number(number: str) -> int | float:
        try:
            return int(number)
        except ValueError:
            return float(number)

    @staticmethod
    def read_array(f: TextIOWrapper) -> list[int | float]:
        array = []
        line = f.readline().strip()
        while "]" not in line:
            if line.startswith("#") or not line:
                continue
            array.append(__class__.parse_number(line))
            line = f.readline().strip()
        return array

    @staticmethod
    def parse_header(f: TextIOWrapper) -> dict[str : int | float | list[int | float]]:
        key = None
        line = True
        header_end = "END OF DATA FILE PARAMETERS"
        header = {}
        while line:
            line = f.readline()
            if header_end in line:
                return header

            if __class__.skip_line(line):
                continue

            if "=" in line:
                key = line.split("=")[0].strip()

            if "[" in line:
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
    ) -> list[float | int]:
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
    def parse_nodal_data(f: TextIOWrapper):
        data = []
        line = True
        primary_line = True
        while line:
            line = f.readline()
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
    def parse_body(f: TextIOWrapper, num_domains: int):
        key = None
        line = True
        body = {}
        while line:
            line = f.readline()
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
    def point_in_cell(cell: SimulationCell, point: np.ndarray):
        if np.any(point < cell[:, 3]):
            return False
        for i in range(3):
            if point[i] >= cell[i, 3] + cell[i, i]:
                return False

    @staticmethod
    def get_next_start_node(nodes: list[Node], start: int):
        while start < len(nodes):
            if nodes[start].num_arms == 2 and not nodes[start].processed:
                return start, nodes[start]
            start += 1
        return None, None

    @staticmethod
    def walk_lines(nodes: list[Node]):
        node_dict = {node.node_tag: node for node in nodes}

        lines = []
        start, start_node = __class__.get_next_start_node(nodes, 0)
        while start_node is not None:
            toProcess = [start_node]
            fwd = 0
            lines.append(Line([[], [start_node]]))
            while toProcess:
                node = toProcess.pop()
                node.processed = True
                lines[-1].segments[fwd].append(node)
                for arm in node.arms:
                    neigh_node = node_dict[arm.arm_tag]
                    if not neigh_node.processed and neigh_node.num_arms < 3:
                        toProcess.append(neigh_node)
                    elif not neigh_node.processed:
                        lines[-1].segments[fwd].append(neigh_node)
                        fwd += 1
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
        rev: bool,
    ):
        for node_id in range(1, len(segment)):
            n0 = segment[node_id - 1]
            n1 = segment[node_id]

            p0 = ref_point + cell.delta_vector(ref_point, n0.pos)
            p1 = p0 + cell.delta_vector(p0, n1.pos)
            ref_point = p1

            positions.append(p0)
            positions.append(p1)
            sections.append(counter)
            sections.append(counter)

            matching_arm = None
            for arm in n0.arms:
                if arm.arm_tag == n1.node_tag:
                    matching_arm = arm
                    break
            assert matching_arm is not None
            bvecs.append(arm.bvec)
            bvecs.append(arm.bvec)
            nvecs.append(arm.nvec)
            nvecs.append(arm.nvec)

        return ref_point

    def parse(self, data: DataCollection, filename: str, **kwargs):

        with open(filename, "r") as f:
            header = __class__.parse_header(f)
            num_domains = functools.reduce(operator.mul, header["dataDecompGeometry"])
            body = __class__.parse_body(f, num_domains)
            assert len(body["nodalData"]) == header["nodeCount"]

        for k, v in header.items():
            data.attributes[k] = v

        cell = np.zeros((3, 4))
        cell[:, 3] = header["minCoordinates"]
        for i in range(3):
            cell[i, i] = header["maxCoordinates"][i] - header["minCoordinates"][i]
        cell = data.create_cell(cell, pbc=(1, 1, 1))

        line_width = 5 * np.linalg.norm(cell[:3, :3].diagonal()) / 1000
        color = (255 / 255, 167 / 255, 26 / 255)

        particles = data.create_particles(count=header["nodeCount"])
        identifier = particles.create_property("Particle Identifier")
        particle_type = particles.create_property("Particle Type")
        positions = particles.create_property("Position")
        num_arms = particles.create_property("Num Arms", dtype=int)
        constraint = particles.create_property("Constraint", dtype=int)

        type_1 = particle_type.add_type_id(1, particles, name="node")
        type_1.radius = line_width / 2
        type_1.color = color

        particle_type[:] = 1

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
        lines = self.walk_lines(body["nodalData"])
        for line in lines:
            segment = list(reversed(line.segments[0]))
            ref_point = np.asarray(segment[0].pos)
            ref_point = self.walk_line(
                segment, ref_point, cell, positions, sections, bvecs, nvecs, counter
            )
            segment = line.segments[1]
            ref_point = self.walk_line(
                segment, ref_point, cell, positions, sections, bvecs, nvecs, counter
            )
            counter += 1

        self.lines_vis.width = line_width
        self.lines_vis.color = color
        lines = data.lines.create("Arms", vis=self.lines_vis)

        lines.create_property("Position", data=positions)
        lines.create_property("Section", data=sections)
        lines.create_property("Burgers vector", data=bvecs)
        lines.create_property(
            "Burgers vector magnitude", data=np.linalg.norm(bvecs, axis=1)
        )
        lines.create_property("Normal vector", data=nvecs)
