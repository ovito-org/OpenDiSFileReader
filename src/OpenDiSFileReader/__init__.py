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

        node_tags = {}
        for i, node in enumerate(body["nodalData"]):
            assert node.node_tag not in node_tags
            node_tags[node.node_tag] = node
            yield i / len(body["nodalData"])

        self.lines_vis.width = line_width
        self.lines_vis.color = color
        lines = data.lines.create("Arms", vis=self.lines_vis)
        positions = []
        section = []
        bvecs = []
        nvecs = []
        counter = 0
        for i, node in enumerate(body["nodalData"]):
            p0 = np.asarray(cell.wrap_point(node.pos))
            for arm in node.arms:

                p1 = np.asarray(node_tags[arm.arm_tag].pos)
                vec = cell.wrap_vector(p1 - p0)
                p1 = p0 + vec

                positions.append(p0)
                positions.append(p1)

                section.append(counter)
                section.append(counter)
                bvecs.append(arm.bvec)
                bvecs.append(arm.bvec)

                nvecs.append(arm.nvec)
                nvecs.append(arm.nvec)
                counter += 1
            yield i / len(body["nodalData"])

        lines.create_property("Position", data=positions)
        lines.create_property("Section", data=section)
        lines.create_property("Burgers vector", data=bvecs)
        lines.create_property(
            "Burgers vector magnitude", data=np.linalg.norm(bvecs, axis=1)
        )
        lines.create_property("Normal vector", data=nvecs)
