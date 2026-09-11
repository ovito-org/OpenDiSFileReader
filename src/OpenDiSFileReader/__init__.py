### OpenDiS File Reader ###
# File reader for the OpenDiS data formats

from collections.abc import Callable
from io import TextIOWrapper
from typing import Any
import re

import numpy as np
from ovito.data import DataCollection
from ovito.io import FileReaderInterface
from ovito.traits import OvitoObject
from ovito.vis import LinesVis


from enum import Enum
class DDDFileFormat(Enum):
    DATAFILE = 1
    EXADIS_RESTART = 2
    UNKNOWN = 3


class OpenDiSFileReader(FileReaderInterface):
    lines_vis = OvitoObject(
        LinesVis, shading=LinesVis.Shading.Normal, wrapped_lines=True, title="Dislocations"
    )

    @staticmethod
    def detect_format(filename: str) -> DDDFileFormat:
        # OpenDiS data files always start with "dataFileVersion = ..."
        # ExaDiS restart files always start with "# ExaDiS restart file"
        try:
            with open(filename, "r") as f:
                line = f.readline().strip()
                if line.startswith("dataFileVersion ="):
                    return DDDFileFormat.DATAFILE
                elif line.startswith("# ExaDiS restart file"):
                    return DDDFileFormat.EXADIS_RESTART
                else:
                    return DDDFileFormat.UNKNOWN
        except OSError:
            return DDDFileFormat.UNKNOWN

    @staticmethod
    def detect(filename: str) -> bool:
        try:
            return __class__.detect_format(filename) != DDDFileFormat.UNKNOWN
        except OSError:
            return False

    def scan(self, filename: str, register_frame: Callable[..., None]) -> None:
        # OpenDiS files contain a single static snapshot
        file_format = __class__.detect_format(filename)
        if file_format == DDDFileFormat.EXADIS_RESTART:
            try:
                header, *_ = __class__.parse_exadis_header(filename)
                step_number = header["step"]
                register_frame(frame_info=(step_number), label=f"Timestep {step_number}")
            except OSError:
                register_frame(frame_info=(0))
        else:
            register_frame(frame_info=(0))
    
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
    def read_array(f: TextIOWrapper) -> list[int | float | list[int | float]]:
        # Reads scalar values or vector rows until a line containing "]" is encountered.
        # Called after the opening "[" has already been consumed by parse_header.
        array = []
        while line := f.readline():
            array_end = "]" in line
            line = line.split("#", 1)[0].replace("]", "").strip()
            if not line:
                if array_end:
                    break
                continue
            tokens = line.split()
            values = [__class__.parse_number(token) for token in tokens]
            array.append(values[0] if len(values) == 1 else values)
            if array_end:
                break
        return array

    @staticmethod
    def parse_header(f: TextIOWrapper) -> dict[str, Any]:
        key = None
        header_end = "nodalData"
        header = {}
        while line := f.readline():
            if header_end in line:
                return header

            if __class__.skip_line(line):
                continue

            if "=" in line:
                key = line.split("=")[0].strip()

            if key == "domainDecomposition":
                continue

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
    def parse_nodal_data(f: TextIOWrapper, node_count: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        # Node records alternate between a primary line (tag, position, num_arms, constraint)
        # and secondary lines (one entry per arm with bvec and nvec).
        nodes = np.empty((node_count, 6), dtype=float) if node_count else []
        segs = []
        nodes_map = {}
        pending_segs: dict[tuple[int, int], list[int]] = {}
        node_index = 0

        content = re.sub(r"(?m)#.*$", "", f.read()).replace(",", " ")
        values = np.fromstring(content, sep=" ")
        cursor = 0

        while cursor < values.size:
            domain = int(values[cursor])
            tag = int(values[cursor + 1])
            num_arms = int(values[cursor + 5])
            node = [
                domain, tag,
                values[cursor + 2], values[cursor + 3], values[cursor + 4],
                int(values[cursor + 6])
            ]
            cursor += 7

            if node_count:
                nodes[node_index] = node
            else:
                nodes.append(node)
            node_tag = (node[0], node[1])
            nodes_map[node_tag] = node_index

            for seg_index in pending_segs.pop(node_tag, []):
                segs[seg_index][1] = node_index

            for _ in range(num_arms):
                arm_tag = (int(values[cursor]), int(values[cursor + 1]))
                if arm_tag in nodes_map:
                    cursor += 8
                    continue

                pending_segs.setdefault(arm_tag, []).append(len(segs))
                segs.append([
                    node_index, -1,
                    *values[cursor + 2:cursor + 5],
                    *values[cursor + 5:cursor + 8]
                ])
                cursor += 8

            node_index += 1

        if pending_segs:
            missing_tag = next(iter(pending_segs))
            raise AssertionError(f"Nodal data references missing node tag {missing_tag}")

        if node_count:
            assert node_index == node_count

        return np.asarray(nodes), np.asarray(segs, dtype=float).reshape((-1, 8))

    @staticmethod
    def parse_body(f: TextIOWrapper, num_domains: int, node_count: int | None = None) -> dict[str, Any]:
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
                nodes, segs = __class__.parse_nodal_data(f, node_count)
                body["nodes"] = nodes
                body["segs"] = segs
        return body

    @staticmethod
    def parse_data_file(filename: str):
        with open(filename, "r") as f:
            header = __class__.parse_header(f)
            num_domains = int(np.prod(header["dataDecompGeometry"]))
        with open(filename, "r") as f:
            body = __class__.parse_body(f, num_domains, header.get("nodeCount"))

        header["segmentCount"] = len(body["segs"])
        pbc = (True, True, True)
        cell = np.zeros((3, 4))
        cell[:, 3] = header["minCoordinates"]
        for i in range(3):
            cell[i, i] = header["maxCoordinates"][i] - header["minCoordinates"][i]

        return header, num_domains, pbc, cell, body
    
    @staticmethod
    def parse_exadis_header(filename: str):
        header: dict[str, Any] = {"dataDecompGeometry": [1, 1, 1]}
        pbc = (True, True, True)
        cell = np.zeros((3, 4))

        def parse_values(tokens: list[str]) -> int | float | str | list[int | float]:
            values = []
            for token in tokens:
                try:
                    values.append(__class__.parse_number(token))
                except ValueError:
                    return " ".join(tokens)
            return values[0] if len(values) == 1 else values

        line_num = 0
        with open(filename, "r") as f:
            while line := f.readline():
                line_num += 1

                if __class__.skip_line(line):
                    continue
                tokens = line.split()
                if not tokens:
                    continue

                key = tokens[0]
                if key == "Nnodes":
                    header["nodeCount"] = int(tokens[1])
                    break
                elif key == "pbc":
                    pbc = tuple(bool(int(t)) for t in tokens[1:4])
                    header["pbc"] = [int(t) for t in tokens[1:4]]
                elif key == "H":
                    h = [float(t) for t in tokens[1:10]]
                    cell[:, :3] = np.asarray(h, dtype=float).reshape((3, 3))
                    header["cellVectors"] = [
                        h[0:3],
                        h[3:6],
                        h[6:9],
                    ]
                elif key == "origin":
                    origin = [float(t) for t in tokens[1:4]]
                    cell[:, 3] = origin
                    header["origin"] = origin
                    header["minCoordinates"] = origin
                elif key == "crystal" and len(tokens) > 2:
                    header[f"crystal_{tokens[1]}"] = parse_values(tokens[2:])
                else:
                    header[key] = parse_values(tokens[1:])

        return header, pbc, cell, line_num

    @staticmethod
    def parse_exadis_file(filename: str):

        header, pbc, cell, line_num = __class__.parse_exadis_header(filename)

        node_count = header["nodeCount"]
        nodes = np.loadtxt(filename, skiprows=line_num, max_rows=node_count)
        if nodes.shape[1] == 5:
            nodes = np.hstack((np.zeros((node_count, 1)), nodes))
        elif nodes.shape[1] == 7:
            nodes = nodes[:,1:]

        line_num = line_num + node_count + 2
        segs = np.loadtxt(filename, skiprows=line_num)
        header["segmentCount"] = segs.shape[0]

        if "minCoordinates" not in header:
            header["minCoordinates"] = [0.0, 0.0, 0.0]
        if "cellVectors" not in header:
            raise ValueError("ExaDiS restart file is missing H cell matrix")

        cell_vectors = np.asarray(header["cellVectors"], dtype=float)
        lower = np.asarray(header["minCoordinates"], dtype=float)
        upper = lower + np.sum(cell_vectors, axis=0)
        header["maxCoordinates"] = upper.tolist()

        num_domains = 1
        body = {"nodes": nodes, "segs": segs}

        return header, num_domains, pbc, cell, body

    @staticmethod
    def generate_connectivity(
        nodes_constraint: np.ndarray, segs_indices: np.ndarray
    ) -> list[list[tuple[int, int, int]]]:
        conn: list[list[tuple[int, int, int]]] = [[] for _ in nodes_constraint]
        for i, (n1, n2) in enumerate(segs_indices):
            conn[n1].append((n2, i, 1))
            conn[n2].append((n1, i, -1))
        return conn

    @staticmethod
    def build_links(
        cell,
        nodes_pos: np.ndarray, nodes_constraint: np.ndarray,
        segs_indices: np.ndarray, segs_bvec: np.ndarray, segs_nvec: np.ndarray,
    ):
        conn = __class__.generate_connectivity(nodes_constraint, segs_indices)
        is_discretization = [
            len(node_conn) == 2 and constr == 0
            for constr, node_conn in zip(nodes_constraint, conn)
        ]
        visited = [-1] * len(nodes_constraint)
        try:
            seg_deltas = np.asarray(
                cell.delta_vector(
                    nodes_pos[segs_indices[:,0]], nodes_pos[segs_indices[:,1]]
                )
            )
            if seg_deltas.shape != segs_bvec.shape:
                raise TypeError
        except (TypeError, ValueError):
            seg_deltas = None

        def next_connection(node_index: int, prev_index: int) -> tuple[int, int, int]:
            node_conn = conn[node_index]
            neighbor, seg_index, order = node_conn[0]
            if neighbor != prev_index:
                return neighbor, seg_index, order
            return node_conn[1]

        num_physical_nodes = 0
        nl = 0

        link_positions = []
        link_sections = []
        link_bvecs = []
        link_nvecs = []

        # Links connected to physical nodes: junctions, endpoints, and constrained nodes.
        for n, node_conn in enumerate(conn):
            if is_discretization[n]:
                continue
            if visited[n] == -1:
                visited[n] = num_physical_nodes
                num_physical_nodes += 1

            for nn, il, order in node_conn:
                if visited[nn] == -1:
                    position = nodes_pos[n]
                    link_positions.append(position)
                    if seg_deltas is None:
                        position = position + cell.delta_vector(position, nodes_pos[nn])
                    else:
                        position = position + order * seg_deltas[il]
                    link_positions.append(position)
                    link_sections.append(nl) # first node
                    link_sections.append(nl) # second node
                    link_bvecs.append(order * segs_bvec[il])
                    link_nvecs.append(order * segs_nvec[il])
                    prev = n

                    if not is_discretization[nn]:
                        visited[nn] = num_physical_nodes
                        num_physical_nodes += 1
                        nl += 1
                        continue

                    visited[nn] = 1
                    while is_discretization[nn]:
                        prev, (nn, ilp, order) = nn, next_connection(nn, prev)
                        if seg_deltas is None:
                            position = position + cell.delta_vector(position, nodes_pos[nn])
                        else:
                            position = position + order * seg_deltas[ilp]
                        link_positions.append(position)
                        link_sections.append(nl)

                        if not is_discretization[nn]:
                            if visited[nn] == -1:
                                visited[nn] = num_physical_nodes
                                num_physical_nodes += 1
                            nl += 1
                        else:
                            visited[nn] = 1
                elif not is_discretization[nn] and nn > n:
                    position = nodes_pos[n]
                    link_positions.append(position)
                    if seg_deltas is None:
                        position = position + cell.delta_vector(position, nodes_pos[nn])
                    else:
                        position = position + order * seg_deltas[il]
                    link_positions.append(position)
                    link_sections.append(nl) # first node
                    link_sections.append(nl) # second node
                    link_bvecs.append(order * segs_bvec[il])
                    link_nvecs.append(order * segs_nvec[il])
                    nl += 1

        # Closed loops made only of discretization nodes.
        for n, node_conn in enumerate(conn):
            if visited[n] != -1 or not is_discretization[n]:
                continue
            visited[n] = num_physical_nodes
            num_physical_nodes += 1
            prev = n
            nn, il, order = node_conn[0]
            position = nodes_pos[n]
            link_positions.append(position)
            if seg_deltas is None:
                position = position + cell.delta_vector(position, nodes_pos[nn])
            else:
                position = position + order * seg_deltas[il]
            link_positions.append(position)
            link_sections.append(nl) # first node
            link_sections.append(nl) # second node
            link_bvecs.append(order * segs_bvec[il])
            link_nvecs.append(order * segs_nvec[il])
            while nn != n:
                visited[nn] = 1
                prev, (nn, ilp, order) = nn, next_connection(nn, prev)
                if seg_deltas is None:
                    position = position + cell.delta_vector(position, nodes_pos[nn])
                else:
                    position = position + order * seg_deltas[ilp]
                link_positions.append(position)
                link_sections.append(nl)
            nl += 1

        link_positions = np.asarray(link_positions)
        link_sections = np.asarray(link_sections)
        link_bvecs = np.asarray(link_bvecs)
        link_bvecs = link_bvecs[link_sections]
        link_nvecs = np.asarray(link_nvecs)
        link_nvecs = link_nvecs[link_sections]

        return link_positions, link_sections, link_bvecs, link_nvecs

    def parse(self, data: DataCollection, filename: str, frame_info: Any, **kwargs: Any):

        # Read data
        file_format = __class__.detect_format(filename)
        if file_format == DDDFileFormat.DATAFILE:
            header, num_domains, pbc, cell, body = __class__.parse_data_file(filename)
        elif file_format == DDDFileFormat.EXADIS_RESTART:
            header, num_domains, pbc, cell, body = __class__.parse_exadis_file(filename)
        else:
            raise ValueError(f"OpenDiSFileReader does not support file type")

        for k, v in header.items():
            data.attributes[k] = v

        cell = data.create_cell(cell, pbc=pbc)

        # Scale line/node width to ~0.1 % of the cell diagonal for visual clarity
        self.lines_vis.width = 1 / 1000 * np.linalg.norm(cell[:3, :3].diagonal())

        # Create nodes
        nodes = body["nodes"]
        assert len(nodes) == header["nodeCount"]
        nodes_pos, nodes_constraint = nodes[:,2:5], nodes[:,5]
        nodes_pos = cell.wrap_point(nodes_pos)

        particles = data.create_particles(count=header["nodeCount"], vis_params={'title': 'Nodes'})
        tags = particles.create_property("Node Tag", dtype=int, components=('domain', 'index'), data=nodes[:,0:2].astype(int))
        particle_type = particles.create_property("Particle Type")
        positions = particles.create_property("Position", data=nodes_pos)
        constraints = particles.create_property("Constraint", dtype=int, data=nodes_constraint)

        node_type = particle_type.add_type_name("Node", data.particles)
        node_type.radius = self.lines_vis.width / 2
        particle_type[:] = node_type.id

        self.lines_vis.color = node_type.color

        # Create lines
        segs = body["segs"]
        assert len(segs) == header["segmentCount"]
        segs_indices = segs[:,0:2].astype(int)
        segs_bvec, segs_nvec = segs[:,2:5], segs[:,5:8]

        link_positions, link_sections, link_bvecs, link_nvecs = \
            __class__.build_links(cell, nodes_pos, nodes_constraint, segs_indices, segs_bvec, segs_nvec)

        lines = data.lines.create("Dislocations", count=len(link_positions), vis=self.lines_vis)
        lines.create_property("Position", data=link_positions)
        lines.create_property("Section", data=link_sections)
        lines.create_property("Burgers vector", data=link_bvecs, components=["X", "Y", "Z"])
        lines.create_property(
            "Burgers vector magnitude", data=np.linalg.norm(link_bvecs, axis=1)
        )
        lines.create_property("Normal vector", data=link_nvecs, components=["X", "Y", "Z"])

        print(f"Number of nodes: {header['nodeCount']}")
        print(f"Number of segments: {header['segmentCount']}")
        print(f"Number of links: {link_sections[-1]+1 if len(link_sections) > 0 else 0}")
