# Simulation d'une galaxie à n corps en utilisant une grille spatiale pour accélérer le calcul des forces gravitationnelles.
#     On crée une classe représentant le système de corps avec la méthode d'intégration basée sur une grille.
# Cette variante de Stage 01 ajoute un benchmark sans affichage et la parallélisation Numba sur les boucles sûres.
import argparse
import numpy as np
import visualizer3d
import time
from numba import get_num_threads, njit, prange

# Unités:
# - Distance: année-lumière (ly)
# - Masse: masse solaire (M_sun)
# - Vitesse: année-lumière par an (ly/an)
# - Temps: année

# Constante gravitationnelle en unités [ly^3 / (M_sun * an^2)]
G = 1.560339e-13

def generate_star_color(mass : float) -> tuple[int, int, int]:
    """
    Génère une couleur pour une étoile en fonction de sa masse.
    Les étoiles massives sont bleues, les moyennes sont jaunes, les petites sont rouges.
    
    Parameters:
    -----------
    mass : float
        Masse de l'étoile en masses solaires
    
    Returns:
    --------
    color : tuple
        Couleur RGB (R, G, B) avec des valeurs entre 0 et 255
    """
    if mass > 5.0:
        # Étoiles massives: bleu-blanc
        return (150, 180, 255)
    elif mass > 2.0:
        # Étoiles moyennes-massives: blanc
        return (255, 255, 255)
    elif mass >= 1.0:
        # Étoiles comme le Soleil: jaune
        return (255, 255, 200)
    else:
        # Étoiles de faible masse: rouge-orange
        return (255, 150, 100)

@njit(parallel=True)
def compute_cell_mass_and_com(cell_start_indices : np.ndarray, body_indices : np.ndarray,
                              cell_masses : np.ndarray, cell_com_positions : np.ndarray,
                              masses : np.ndarray, positions : np.ndarray):
    for i in prange(cell_masses.shape[0]):
        cell_mass = 0.0
        com_x = 0.0
        com_y = 0.0
        com_z = 0.0
        start_idx = cell_start_indices[i]
        end_idx = cell_start_indices[i + 1]
        for j in range(start_idx, end_idx):
            ibody = body_indices[j]
            m = masses[ibody]
            cell_mass += m
            com_x += positions[ibody, 0] * m
            com_y += positions[ibody, 1] * m
            com_z += positions[ibody, 2] * m
        cell_masses[i] = cell_mass
        if cell_mass > 0.0:
            inv_mass = 1.0 / cell_mass
            cell_com_positions[i, 0] = com_x * inv_mass
            cell_com_positions[i, 1] = com_y * inv_mass
            cell_com_positions[i, 2] = com_z * inv_mass
        else:
            cell_com_positions[i, 0] = 0.0
            cell_com_positions[i, 1] = 0.0
            cell_com_positions[i, 2] = 0.0

@njit
def update_stars_in_grid( cell_start_indices : np.ndarray, body_indices : np.ndarray,
                          cell_masses : np.ndarray, cell_com_positions : np.ndarray,
                          masses: np.ndarray,
                          positions : np.ndarray, grid_min : np.ndarray, grid_max : np.ndarray,
                          cell_size : np.ndarray, n_cells : np.ndarray):
    n_bodies = positions.shape[0]
    # Réinitialise les compteurs de début des cellules
    cell_start_indices.fill(-1)
    # Compte le nombre de corps dans chaque cellule
    cell_counts = np.zeros(shape=(np.prod(n_cells),), dtype=np.int64)
    for ibody in range(n_bodies):
        cell_idx = np.floor((positions[ibody] - grid_min) / cell_size).astype(np.int64)
        # Gère le cas où un corps est exactement sur la borne max   
        for i in range(3):
            if cell_idx[i] >= n_cells[i]:
                cell_idx[i] = n_cells[i] - 1
            elif cell_idx[i] < 0:
                cell_idx[i] = 0
        morse_idx = cell_idx[0] + cell_idx[1]*n_cells[0] + cell_idx[2]*n_cells[0]*n_cells[1]
        cell_counts[morse_idx] += 1
    # Calcule les indices de début des cellules
    running_index = 0
    for i in range(len(cell_counts)):
        cell_start_indices[i] = running_index
        running_index += cell_counts[i]
    cell_start_indices[len(cell_counts)] = running_index # Fin du dernier corps
    # Remplit les indices des corps dans les cellules
    current_counts = np.zeros(shape=(np.prod(n_cells),), dtype=np.int64)
    for ibody in range(n_bodies):
        cell_idx = np.floor((positions[ibody] - grid_min) / cell_size).astype(np.int64)
        for i in range(3):
            if cell_idx[i] >= n_cells[i]:
                cell_idx[i] = n_cells[i] - 1
            elif cell_idx[i] < 0:
                cell_idx[i] = 0
        morse_idx = cell_idx[0] + cell_idx[1]*n_cells[0] + cell_idx[2]*n_cells[0]*n_cells[1]
        index_in_cell = cell_start_indices[morse_idx] + current_counts[morse_idx]
        body_indices[index_in_cell] = ibody
        current_counts[morse_idx] += 1
    compute_cell_mass_and_com(cell_start_indices, body_indices, cell_masses, cell_com_positions, masses, positions)

@njit(parallel=True)
def compute_acceleration( positions : np.ndarray, masses : np.ndarray,
                          cell_start_indices : np.ndarray, body_indices : np.ndarray,
                          cell_masses : np.ndarray, cell_com_positions : np.ndarray,
                          grid_min : np.ndarray, grid_max : np.ndarray,
                          cell_size : np.ndarray, n_cells : np.ndarray):
    n_bodies = positions.shape[0]
    a = np.zeros_like(positions)
    for ibody in prange(n_bodies):
        pos = positions[ibody]
        cell_idx = np.floor((pos - grid_min) / cell_size).astype(np.int64)
        for i in range(3):
            if cell_idx[i] >= n_cells[i]:
                cell_idx[i] = n_cells[i] - 1
            elif cell_idx[i] < 0:
                cell_idx[i] = 0
        # Parcourt toutes les cellules pour calculer la contribution gravitationnelle
        for ix in range(n_cells[0]):
            for iy in range(n_cells[1]):
                for iz in range(n_cells[2]):
                    morse_idx = ix + iy*n_cells[0] + iz*n_cells[0]*n_cells[1]
                    if (abs(ix-cell_idx[0]) > 2) or (abs(iy-cell_idx[1]) > 2) or (abs(iz-cell_idx[2]) > 2):
                        cell_com = cell_com_positions[morse_idx]    
                        cell_mass = cell_masses[morse_idx]
                        if cell_mass > 0.0:
                            direction = cell_com - pos
                            distance = np.sqrt(direction[0]**2 + direction[1]**2 + direction[2]**2)
                            if distance > 1.E-10:
                                inv_dist3 = 1.0 / (distance ** 3)
                                a[ibody,:] += G * direction[:] * inv_dist3 * cell_mass
                    else:
                        # Parcourt les corps dans cette cellule
                        start_idx = cell_start_indices[morse_idx]
                        end_idx = cell_start_indices[morse_idx+1]
                        for j in range(start_idx, end_idx):
                            jbody = body_indices[j]
                            if jbody != ibody:
                                direction = positions[jbody] - pos
                                distance = np.sqrt(direction[0]**2 + direction[1]**2 + direction[2]**2)
                                if distance > 1.E-10:
                                    inv_dist3 = 1.0 / (distance ** 3)
                                    a[ibody,:] += G * direction[:] * inv_dist3 * masses[jbody]
    return a

@njit(parallel=True)
def integrate_positions(positions : np.ndarray, velocities : np.ndarray, accelerations : np.ndarray, dt : float):
    for ibody in prange(positions.shape[0]):
        positions[ibody, 0] += velocities[ibody, 0] * dt + 0.5 * accelerations[ibody, 0] * dt * dt
        positions[ibody, 1] += velocities[ibody, 1] * dt + 0.5 * accelerations[ibody, 1] * dt * dt
        positions[ibody, 2] += velocities[ibody, 2] * dt + 0.5 * accelerations[ibody, 2] * dt * dt

@njit(parallel=True)
def integrate_velocities(velocities : np.ndarray, accelerations : np.ndarray, accelerations_new : np.ndarray, dt : float):
    for ibody in prange(velocities.shape[0]):
        velocities[ibody, 0] += 0.5 * (accelerations[ibody, 0] + accelerations_new[ibody, 0]) * dt
        velocities[ibody, 1] += 0.5 * (accelerations[ibody, 1] + accelerations_new[ibody, 1]) * dt
        velocities[ibody, 2] += 0.5 * (accelerations[ibody, 2] + accelerations_new[ibody, 2]) * dt

# On crée une grille cartésienne régulière pour diviser l'espace englobant la galaxie en cellules
class SpatialGrid:
    """_summary_
    """
    def __init__(self, positions : np.ndarray, nb_cells_per_dim : tuple[int, int, int]):
        self.min_bounds = np.min(positions, axis=0) - 1.E-6
        self.max_bounds = np.max(positions, axis=0) + 1.E-6
        self.n_cells = np.array(nb_cells_per_dim)
        self.cell_size = (self.max_bounds - self.min_bounds) / self.n_cells
        # On va stocker les indices des corps dans chaque cellule adéquate
        # Les cellules seront stockées sous une forme morse : indice de la cellule = ix + iy*n_cells_x + iz*n_cells_x*n_cells_y
        # et on gère deux tableaux : un pour le début des indices de chaque cellule, un autre pour les indices des corps
        self.cell_start_indices = np.full(np.prod(self.n_cells) + 1, -1, dtype=np.int64)
        self.body_indices = np.empty(shape=(positions.shape[0],), dtype=np.int64)
        # Stockage du centre de masse de chaque cellule et de la masse totale contenue dans chaque cellule
        self.cell_masses = np.zeros(shape=(np.prod(self.n_cells),), dtype=np.float32)
        self.cell_com_positions = np.zeros(shape=(np.prod(self.n_cells), 3), dtype=np.float32)
        
    def update_bounds(self, positions : np.ndarray):
        self.min_bounds = np.min(positions, axis=0) - 1.E-6
        self.max_bounds = np.max(positions, axis=0) + 1.E-6
        self.cell_size = (self.max_bounds - self.min_bounds) / self.n_cells
        
    def update(self, positions : np.ndarray, masses : np.ndarray):
        #self.update_bounds(positions)
        update_stars_in_grid( self.cell_start_indices, self.body_indices,                             
                              self.cell_masses, self.cell_com_positions,
                              masses,
                              positions, self.min_bounds, self.max_bounds,
                              self.cell_size, self.n_cells)

class NBodySystem:
    def __init__(self, filename, ncells_per_dir : tuple[int, int, int] = (10,10,10)):
        positions = []
        velocities = []
        masses    = []
        
        self.max_mass = 0.
        self.box = np.array([[-1.E-6,-1.E-6,-1.E-6],[1.E-6,1.E-6,1.E-6]], dtype=np.float64) # Contient les coins min et max du système
        with open(filename, "r") as fich:
            line = fich.readline() # Récupère la masse, la position et la vitesse sous forme de chaîne
            # Récupère les données numériques pour instancier un corps qu'on rajoute aux corps déjà présents :
            while line:
                data = line.split()
                masses.append(float(data[0]))
                positions.append([float(data[1]), float(data[2]), float(data[3])])
                velocities.append([float(data[4]), float(data[5]), float(data[6])])
                self.max_mass = max(self.max_mass, masses[-1])
                
                for i in range(3):
                    self.box[0][i] = min(self.box[0][i], positions[-1][i]-1.E-6)
                    self.box[1][i] = max(self.box[1][i], positions[-1][i]+1.E-6)
                    
                line = fich.readline()
        
        self.positions  = np.array(positions, dtype=np.float32)
        self.velocities = np.array(velocities, dtype=np.float32)
        self.masses     = np.array(masses, dtype=np.float32)
        self.colors = [generate_star_color(m) for m in masses]
        self.grid = SpatialGrid(self.positions, ncells_per_dir)
        self.grid.update(self.positions, self.masses)
        
    def update_positions(self, dt):
        """Applique la méthode de Verlet vectorisée pour mettre à jour les positions et vitesses des corps."""
        a = compute_acceleration( self.positions, self.masses,
                                  self.grid.cell_start_indices, self.grid.body_indices,
                                  self.grid.cell_masses, self.grid.cell_com_positions,
                                  self.grid.min_bounds, self.grid.max_bounds,
                                  self.grid.cell_size, self.grid.n_cells)
        integrate_positions(self.positions, self.velocities, a, dt)
        self.grid.update(self.positions, self.masses)
        a_new = compute_acceleration( self.positions, self.masses,
                                      self.grid.cell_start_indices, self.grid.body_indices,
                                      self.grid.cell_masses, self.grid.cell_com_positions,
                                      self.grid.min_bounds, self.grid.max_bounds,
                                      self.grid.cell_size, self.grid.n_cells)
        integrate_velocities(self.velocities, a, a_new, dt)

system : NBodySystem

def update_positions(dt : float):
    global system
    system.update_positions(dt)
    return system.positions

def run_simulation(filename, geometry=(800,600), ncells_per_dir : tuple[int, int, int] = (10,10,10), dt=0.001):
    # Initialise le système de corps :
    global system
    system = NBodySystem(filename, ncells_per_dir=ncells_per_dir)
    # Initialise l'affichage graphique :
    pos = system.positions
    col = system.colors
    intensity = np.clip(system.masses / system.max_mass, 0.5, 1.0)
    visu = visualizer3d.Visualizer3D(pos, col, intensity,  [[system.box[0][0], system.box[1][0]], [system.box[0][1], system.box[1][1]], [system.box[0][2], system.box[1][2]]])
    visu.run(updater=update_positions, dt = dt)

def run_benchmark(filename : str, ncells_per_dir : tuple[int, int, int], dt : float, steps : int, warmup : int):
    local_system = NBodySystem(filename, ncells_per_dir=ncells_per_dir)

    for _ in range(warmup):
        local_system.update_positions(dt)

    step_times = np.empty(steps, dtype=np.float64)
    for istep in range(steps):
        t0 = time.perf_counter()
        local_system.update_positions(dt)
        step_times[istep] = time.perf_counter() - t0

    total_time = float(np.sum(step_times))
    mean_time = float(np.mean(step_times))
    std_time = float(np.std(step_times))
    print(
        "BENCHMARK "
        f"threads={get_num_threads()} "
        f"warmup={warmup} "
        f"steps={steps} "
        f"total_s={total_time:.6f} "
        f"mean_step_ms={mean_time * 1000.0:.3f} "
        f"std_step_ms={std_time * 1000.0:.3f}"
    )

def parse_args():
    parser = argparse.ArgumentParser(description="Simulation N-corps Stage 01 avec Numba et benchmark compute-only.")
    parser.add_argument("filename", nargs="?", default="data/galaxy_1000")
    parser.add_argument("dt", nargs="?", type=float, default=0.001)
    parser.add_argument("ni", nargs="?", type=int, default=20)
    parser.add_argument("nj", nargs="?", type=int, default=20)
    parser.add_argument("nk", nargs="?", type=int, default=1)
    parser.add_argument("--benchmark", action="store_true", help="Exécute uniquement le calcul, sans SDL/OpenGL.")
    parser.add_argument("--steps", type=int, default=30, help="Nombre de pas mesurés en mode benchmark.")
    parser.add_argument("--warmup", type=int, default=1, help="Nombre de pas non mesurés pour amortir la compilation JIT.")
    return parser.parse_args()

def main():
    args = parse_args()
    n_cells_per_dir = (args.ni, args.nj, args.nk)

    if args.benchmark:
        print(
            f"Benchmark de {args.filename} avec dt = {args.dt}, grille {n_cells_per_dir}, "
            f"warmup = {args.warmup}, steps = {args.steps}"
        )
        run_benchmark(args.filename, ncells_per_dir=n_cells_per_dir, dt=args.dt, steps=args.steps, warmup=args.warmup)
    else:
        print(f"Simulation de {args.filename} avec dt = {args.dt} et grille {n_cells_per_dir}")
        run_simulation(args.filename, ncells_per_dir=n_cells_per_dir, dt=args.dt)

if __name__ == "__main__":
    main()
