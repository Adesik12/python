import numpy as np
import MDAnalysis as mda
from ase import Atoms, io
from ase import units
from ase.units import fs
import nglview
from scipy.constants import k
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution


class BerendsenThermostat:


    def __init__(self, temperature, tau, time_step):

        self.target_temperature = temperature
        self.tau = tau
        self.time_step = time_step
        self.current_temperature = None

    def apply(self, atoms):


        velocities = atoms.get_velocities()
        current_temperature = np.sum(0.5 * atoms.get_masses() * np.sum(velocities ** 2, axis=1)) / (
                    1.5 * k * len(atoms))
        scaling_factor = np.sqrt(1 + (self.time_step / self.tau) * (self.target_temperature / current_temperature - 1))
        atoms.set_velocities(velocities * scaling_factor)
        self.current_temperature = current_temperature
        return atoms


def compute_forces_with_cutoff(universe, cutoff_radius):



    positions = universe.positions
    epsilon = 0.238  # Параметр энергии
    sigma = 3.405  # Параметр расстояния

    num_atoms = len(universe)
    forces = np.zeros((num_atoms, 3))

    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            rij = positions[i] - positions[j]
            r = np.linalg.norm(rij)


            if r < cutoff_radius:
                lj_force = 48 * epsilon * ((sigma ** 12 / r ** 13) - (sigma ** 6 / r ** 7)) * rij / r
                forces[i] += lj_force
                forces[j] -= lj_force

    return forces


def run_md_simulation_with_cutoff(universe, num_steps, time_step, cutoff_radius):


    total_energy_values = []

    for step in range(num_steps):

        forces = compute_forces_with_cutoff(universe, cutoff_radius)


        update_coordinates_and_velocities_verlet(universe, forces, time_step)


        total_energy = compute_total_energy(universe,cutoff_radius)
        total_energy_values.append(total_energy)

    return total_energy_values


def update_coordinates_and_velocities_verlet(universe, forces, time_step):


    positions = universe.positions
    velocities = universe.get_velocities()
    masses = universe.get_masses()
    accelerations = forces / masses[:, np.newaxis]


    new_positions = positions + velocities * time_step + 0.5 * accelerations * time_step ** 2


    new_forces = compute_forces_with_cutoff(universe,
                                            cutoff_radius)  # Предполагается, что cutoff_radius определен где-то ранее
    new_accelerations = new_forces / masses[:, np.newaxis]


    new_velocities = velocities + 0.5 * (accelerations + new_accelerations) * time_step

    universe.positions = new_positions
    universe.set_velocities(new_velocities)


def save_trajectory(step, atoms):

    with open(f"trajectory_step_{step}.xyz", 'w') as traj_file:
        traj_file.write(f"{len(atoms)}\n")
        traj_file.write("Atoms. Positions\n")
        for atom in atoms:
            traj_file.write(f"Ar {atom.x} {atom.y} {atom.z}\n")


def compute_potential_energy(atoms, cutoff_radius):

    positions = atoms.get_positions()
    epsilon = 0.238  # Параметр энергии
    sigma = 3.405  # Параметр расстояния
    total_potential_energy = 0.0
    num_atoms = len(atoms)

    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            rij = np.linalg.norm(positions[i] - positions[j])


            if rij < cutoff_radius:
                lj_potential = 4 * epsilon * ((sigma / rij) ** 12 - (sigma / rij) ** 6)
                total_potential_energy += lj_potential

    return total_potential_energy


def compute_total_energy(atoms,cutoff_radius ):


    velocities = atoms.get_velocities()
    kinetic_energy = 0.5 * np.sum(np.sum(velocities ** 2, axis=1) * atoms.get_masses())


    potential_energy = compute_potential_energy(atoms,cutoff_radius)


    total_energy = kinetic_energy + potential_energy

    return total_energy


def compute_radial_distribution(atoms, box_length, num_bins, max_distance):



    bin_edges = np.linspace(0, max_distance, num_bins + 1)
    bin_width = bin_edges[1] - bin_edges[0]

    distances = []


    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            dx = atoms[i].x - atoms[j].x
            dy = atoms[i].y - atoms[j].y
            dz = atoms[i].z - atoms[j].z


            dx -= box_length * round(dx / box_length)
            dy -= box_length * round(dy / box_length)
            dz -= box_length * round(dz / box_length)

            distance = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
            distances.append(distance)


    hist, _ = np.histogram(distances, bins=bin_edges, density=True)
    rdf = hist / (4 * np.pi * bin_edges[1:] ** 2 * bin_width * len(atoms))


    return bin_edges[1:], rdf


def create_argon_system(temperature, volume, num_atoms):


    def compute_radial_distribution(atoms, box_length, num_bins, max_distance):
        bin_edges = np.linspace(0, max_distance, num_bins + 1)
        bin_width = bin_edges[1] - bin_edges[0]

        distances = []
        for i in range(len(atoms)):
            for j in range(i + 1, len(atoms)):
                dx = atoms[i].x - atoms[j].x
                dy = atoms[i].y - atoms[j].y
                dz = atoms[i].z - atoms[j].z

                # Periodic boundary conditions
                dx -= box_length * round(dx / box_length)
                dy -= box_length * round(dy / box_length)
                dz -= box_length * round(dz / box_length)

                distance = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
                distances.append(distance)

        hist, _ = np.histogram(distances, bins=bin_edges, density=True)
        rdf = hist / (4 * np.pi * bin_edges[1:] ** 2 * bin_width * len(atoms))

        return bin_edges[1:], rdf


    coordinates = np.random.uniform(-5, 5, size=(num_atoms, 3))
    argon_atoms = Atoms(symbols='Ar' * num_atoms, positions=coordinates, pbc=True)


    argon_atoms.set_pbc([True, True, True])


    volume_ase = volume * (1e-30)
    cell = np.zeros((3, 3))
    np.fill_diagonal(cell, volume_ase ** (1 / 3))
    argon_atoms.set_cell(cell, scale_atoms=True)


    mean_velocity = np.sqrt(3 * k * temperature / argon_atoms.get_masses().sum())
    velocities = np.random.normal(loc=0, scale=mean_velocity, size=(num_atoms, 3))
    MaxwellBoltzmannDistribution(argon_atoms, temperature_K=temperature)
    argon_atoms.set_velocities(velocities)


    with open('argon.xyz', 'w') as xyz_file:
        xyz_file.write(f"{num_atoms}\n")
        xyz_file.write("Atoms. Positions and velocities\n")
        velocities = argon_atoms.get_velocities()
        for i, atom in enumerate(argon_atoms):
            xyz_file.write(f"Ar {atom.x} {atom.y} {atom.z} {velocities[i][0]} {velocities[i][1]} {velocities[i][2]}\n")

    return argon_atoms


# simulation parameters
num_steps = 1000
time_step = 0.1 * units.fs
cutoff_radius = 10.0


argon_universe = create_argon_system(300, 100, 1000)  # Создание системы атомов аргона с заданными параметрами
box_length = argon_universe.cell.cellpar()[0]


tau_berendsen = 1.0 * units.fs
thermostat = BerendsenThermostat(temperature=300.0, tau=tau_berendsen, time_step=time_step)


total_energy_values_with_cutoff = []

# Saving the initial coordinates
initial_atoms = Atoms(argon_universe.symbols, positions=argon_universe.positions)
initial_pdb_filename = "initial_config.pdb"
io.write(initial_pdb_filename, initial_atoms)

# Creating an initial interactive visualization
view = nglview.show_file(initial_pdb_filename)
view.add_representation('ball+stick')
view.center()
view.camera = "orthographic"
view.parameters = {"clipDist": 0}
view.parameters = {"clipNear": 0}
view
temperature_values = []
kinetic_energy_values = []
potential_energy_values = []
total_energy_values = []
#starting the simulation
for step in range(num_steps):
    argon_universe = thermostat.apply(argon_universe)


    forces = compute_forces_with_cutoff(argon_universe, cutoff_radius)


    update_coordinates_and_velocities_verlet(argon_universe, forces, time_step)


    current_temperature = thermostat.current_temperature


    kinetic_energy = 0.5 * np.sum(np.sum(argon_universe.get_velocities() ** 2, axis=1) * argon_universe.get_masses())
    potential_energy = compute_potential_energy(argon_universe,cutoff_radius)


    total_energy = kinetic_energy + potential_energy


    print(f"step {step + 1}: Current temperature = {current_temperature} K")
    print(f"Kinetic energy = {kinetic_energy} eV")
    print(f"potential energy = {potential_energy} eV")
    print(f"total energy = {total_energy} eV")


max_distance = 10.0
num_bins = 100

rdf_bin_edges, rdf_values = compute_radial_distribution(argon_universe, box_length, num_bins, max_distance)


with open("radial_distribution.csv", 'w') as rdf_file:
    rdf_file.write("Distance, RDF\n")
    for distance, rdf in zip(rdf_bin_edges, rdf_values):
        rdf_file.write(f"{distance}, {rdf}\n")


def compute_rmsd(atoms, initial_positions):

    squared_distances = [(a.x - initial.x)**2 + (a.y - initial.y)**2 + (a.z - initial.z)**2 for a, initial in zip(atoms, initial_positions)]
    rmsd = np.sqrt(np.mean(squared_distances))
    return rmsd


initial_positions = [Atom(initial.x, initial.y, initial.z) for initial in argon_universe]
final_positions = [Atom(final.x, final.y, final.z) for final in argon_universe]


final_rmsd = compute_rmsd(final_positions, initial_positions)
print(f"The radius of flexibility at the end of the simulation: {final_rmsd}")
