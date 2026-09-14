import numpy as np
import pytest
from ovito.io import import_file


@pytest.fixture(scope="module")
def restart_data():
    pipeline = import_file(
        "https://raw.githubusercontent.com/OpenDiS/OpenDiS/refs/heads/main/examples/11_restart_calculation/output_fcc_Cu_15um_1e3/restart.1000.exadis"
    )
    return pipeline.compute()


def test_restart_file_particles(restart_data):
    particles = restart_data.particles
    assert particles.count == 16587
    np.testing.assert_array_equal(particles["Node Tag"][1], np.array([0, 1]))
    np.testing.assert_array_almost_equal(
        particles["Position"][1],
        np.array((50.61472253415699, -5478.35496528705, -23268.62984057775)),
    )
    assert particles["Particle Type"][1] == 1
    assert particles["Constraint"][1] == 0


def test_restart_file_cell(restart_data):
    cell = restart_data.cell
    assert cell[0, 0] == 58824
    assert cell[1, 1] == 58824
    assert cell[2, 2] == 58824
    assert cell[0, 3] == -29412
    assert cell[1, 3] == -29412
    assert cell[2, 3] == -29412
    assert cell[0, 1] == 0
    assert cell[0, 2] == 0
    assert cell[1, 0] == 0
    assert cell[1, 2] == 0
    assert cell[2, 0] == 0
    assert cell[2, 1] == 0


def test_restart_file_lines(restart_data):
    lines = restart_data.lines["Dislocations"]
    assert lines.count == 21380

    np.testing.assert_array_almost_equal(
        lines["Position"][1],
        np.array((-22696.928511200043, -15144.029259799943, -25956.73443400004)),
    )
    assert lines["Section"][1] == 0
    np.testing.assert_array_almost_equal(
        lines["Burgers vector"][1], np.array((0.707107, 0, 0.707107))
    )
    np.testing.assert_almost_equal(lines["Burgers vector magnitude"][1], 1)
    np.testing.assert_array_almost_equal(
        lines["Normal vector"][1], np.array((0.57735, -0.57735, -0.57735))
    )


@pytest.fixture(scope="module")
def data_data():
    pipeline = import_file(
        "https://raw.githubusercontent.com/OpenDiS/OpenDiS/refs/heads/main/examples/10_strain_hardening/180chains_16.10e.data"
    )
    return pipeline.compute()


def test_data_file_particles(data_data):
    particles = data_data.particles
    assert particles.count == 5055
    np.testing.assert_array_equal(particles["Node Tag"][1], np.array([0, 1]))
    np.testing.assert_array_almost_equal(
        particles["Position"][1],
        np.array((-29411.99158503, 21757.39115484, -2123.39364935)),
    )
    assert particles["Particle Type"][1] == 1
    assert particles["Constraint"][1] == 0


def test_data_file_cell(data_data):
    cell = data_data.cell
    assert cell[0, 0] == 58824
    assert cell[1, 1] == 58824
    assert cell[2, 2] == 58824
    assert cell[0, 3] == -29412
    assert cell[1, 3] == -29412
    assert cell[2, 3] == -29412
    assert cell[0, 1] == 0
    assert cell[0, 2] == 0
    assert cell[1, 0] == 0
    assert cell[1, 2] == 0
    assert cell[2, 0] == 0
    assert cell[2, 1] == 0


def test_data_file_lines(data_data):
    lines = data_data.lines["Dislocations"]
    assert lines.count == 8221

    np.testing.assert_array_almost_equal(
        lines["Position"][1],
        np.array((7925.62874548, -20872.16185782, -1668.37119666)),
    )
    assert lines["Section"][1] == 0
    np.testing.assert_array_almost_equal(
        lines["Burgers vector"][1], np.array((0, 0.707107, -0.707107))
    )
    np.testing.assert_almost_equal(lines["Burgers vector magnitude"][1], 1)
    np.testing.assert_array_almost_equal(
        lines["Normal vector"][1], np.array((0.57735, 0.57735, 0.57735))
    )
