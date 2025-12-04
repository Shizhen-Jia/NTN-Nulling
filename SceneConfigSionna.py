# import os
# if os.getenv("CUDA_VISIBLE_DEVICES") is None:
#     gpu_num = 0 # Use "" to use the CPU
#     os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Or "" for CPU only
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("✅ Using GPU:", gpus[0].name)
    except RuntimeError as e:
        print("❌ RuntimeError:", e)
else:
    print("❌ No GPU available. Using CPU.")

import sionna
import sionna.rt
import numpy as np
import tensorflow as tf
# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#     try:
#         tf.config.experimental.set_memory_growth(gpus[0], True)
#     except RuntimeError as e:
#         print(e)
# # Avoid warnings from TensorFlow
# tf.get_logger().setLevel('ERROR')
# from antenna_vsat import v60g_tx_pattern_x_axis
from satellite_projection import satellite_projection
import sionna
import sionnautils
from sionnautils.miutils import CoverageMapPlanner 
from sionna.rt import Scene,load_scene, PlanarArray, Transmitter, Receiver, Camera,\
                      PathSolver, RadioMapSolver, subcarrier_frequencies, AntennaPattern
# from sionna.channel import cir_to_time_channel
# from sionna.rt.antenna import visualize
tf.keras.backend.clear_session()
from scipy.special import jv
import mitsuba as mi
from sionna.rt.antenna_pattern import register_antenna_pattern,create_factory,PolarizedAntennaPattern
from sionna.phy.channel import  cir_to_time_channel



# Define constants (adjust these as needed)
dish_diameter = 0.6        # meters
tx_gain_dB = 38.1           # dBi
c = 3e8                     # speed of light (m/s)
# tx_frequency_mid = np.mean([13.75, 14.5]) * 1e9  # mid frequency in Hz
tx_frequency_mid = np.mean([9.999, 9.999]) * 1e9
def v_vsat_pattern(theta: mi.Float, phi: mi.Float) -> mi.Complex2f:
    """
    Custom vertically-polarized VSAT antenna pattern.
    
    This function uses the (2*J₁(u)/u)² model, where
    u = π*dish_diameter*sin(angle_from_boresight)/(λ)
    and applies gain normalization and back-lobe suppression.
    """
    # Convert theta and phi to numpy arrays (if not already)
    theta_np = theta.numpy() if hasattr(theta, "numpy") else np.array(theta)
    phi_np = phi.numpy() if hasattr(phi, "numpy") else np.array(phi)
    
    # Compute the angle from the boresight.
    # Here, boresight is assumed to point along the positive x-axis.
    ray_x = np.sin(theta_np) * np.cos(phi_np)
    angle_from_x = np.arccos(np.clip(ray_x, -1.0, 1.0))
    
    # Compute u parameter using the mid-frequency wavelength λ = c/tx_frequency_mid
    u = np.pi * dish_diameter * np.sin(angle_from_x) / (c / tx_frequency_mid)
    epsilon = 1e-10
    u_safe = np.where(np.abs(u) < epsilon, epsilon, u)
    
    # Compute the aperture pattern using the Bessel function J₁
    j1_u = jv(1, u_safe)
    pattern = (2.0 * j1_u / u_safe)**2
    # Handle boresight (when u is zero)
    pattern = np.where(np.abs(u) < epsilon, 1.0, pattern)
    
    # Apply gain normalization (convert from dBi to linear scale)
    max_gain_linear = 10**(tx_gain_dB/10)
    normalized_pattern = pattern * max_gain_linear
    
    # Suppress the back-lobe (for angles beyond 90° from boresight)
    back_lobe_mask = angle_from_x > (np.pi/2)
    normalized_pattern = np.where(back_lobe_mask, normalized_pattern * 0.001, normalized_pattern)
    
    # Convert from power to field amplitude (E = √power)
    field_amplitude = np.sqrt(normalized_pattern)
    
    # Convert the result into the expected complex type (imaginary part is zero)
    field_amplitude_mi = mi.Float(field_amplitude)
    return mi.Complex2f(field_amplitude_mi, mi.Float(0))

from sionna.rt.antenna_pattern import register_antenna_pattern


def create_vsat_factory(name: str):
    def f(*, polarization, polarization_model="tr38901_2"):
        from sionna.rt.antenna_pattern import PolarizedAntennaPattern
        return PolarizedAntennaPattern(
            v_pattern=globals()["v_" + name + "_pattern"],
            polarization=polarization,
            polarization_model=polarization_model
        )
    return f

# Ensure your custom pattern function is available in globals
globals()["v_vsat_pattern"] = v_vsat_pattern

# Register the antenna pattern with the name "vsat"
register_antenna_pattern("vsat", create_vsat_factory("vsat"))

class SceneConfigSionna:
    def __init__(self, scene, nbs, nsect, fc = 10e9):
        """
        scene : A sionna.rt.Scene object with loaded geometry (XML or otherwise).
        """
        self.scene = scene
        
        self.fc = fc

        # Some default parameters you can modify:
        self.grid_size = 1.0
        self.nbs = nbs               # Number of gNB/base stations
        self.nsect = nsect              # Usually 3, for sector-based coverage
        self.BS_height_above_roof = 35  # For base station on building
        self.BS_height_above_ground = 45
        self.tn_height_above_roof = 1  # For base station on building
        self.tn_height_above_ground = 1.6
        self.ntn_height_above_roof = 1  # For base station on building
        self.ntn_height_above_ground = 1.6
        self.sat_distance = 500e3   # Satellite distance from region (m)

        # Coverage map placeholders
        self.cm = None
        self.L_NS = None
        self.W_WE = None
        self.bbox = None
        self.extent = None
        self.point_type = None
        self.paths_tn = None
        self.paths_ntn = None

        # Position arrays
        self.tx_pos = None
        self.rx_ntn_pos = None
        self.bs_tn_pos = None
        self.ntn_look_pos = None

        # Path results
        self.a_tn = None
        self.tau_tn = None
        self.a_ntn = None
        self.taps_ntn = None
        self.tau_ntn = None
        self.ntn_rx = None
        
        self.toff = None
        self.time = None

    def compute_positions(self,
                        ntn_rx,
                        tn_rx,
                        azimuth,
                        elevation,
                        centerBS=True,
                        bs_dist_min=35,
                        bs_dist_max=1000):
        """
        1) Build coverage map and determine bounding box
        2) Select random positions for TX, TN/NTN receivers
        3) Optionally place TX at (x=0, y=0) if centerBS=True
        4) Filter TN receivers by distance constraints
        5) Compute random satellite direction & project
        """

        # 1) Create coverage map
        self.ntn_rx = ntn_rx
        self.tn_rx = tn_rx
        self.cm = CoverageMapPlanner(self.scene._scene, grid_size=self.grid_size)
        self.cm.set_grid()
        self.cm.compute_grid_attributes()
        self.hm = None

        # Determine bounding box from coverage map
        x_min, x_max = self.cm.x[0], self.cm.x[-1]
        y_min, y_max = self.cm.y[0], self.cm.y[-1]

        self.W_WE = x_max - x_min   # width (East-West)
        self.L_NS = y_max - y_min   # length (North-South)
        self.bbox = [-self.W_WE/2, self.W_WE/2, -self.L_NS/2, self.L_NS/2]
        self.extent = [self.cm.x[0], self.cm.x[-1], self.cm.y[0], self.cm.y[-1]]

        # 2) Place a single gNB TX position on building roof
        # locations_building = np.argwhere(self.cm.bldg_grid & self.cm.in_region)        
        locations_building = np.argwhere(self.cm.bldg_grid)
        locations_outdoor = np.argwhere(~self.cm.bldg_grid)
        
        outdoor = (self.cm.bldg_grid==False) 
        building = (self.cm.bldg_grid) 
        self.point_type = outdoor + 2*building
        self.point_type = self.point_type.astype(int)
        self.point_type = np.flipud(self.point_type)
        
        
        
        # if len(locations_building) < self.nbs:
        #     raise ValueError("Not enough building points to place the TX.")
        
        x_limit_min = x_min + bs_dist_max
        x_limit_max = x_max - bs_dist_max
        y_limit_min = y_min + bs_dist_max
        y_limit_max = y_max - bs_dist_max
        x_coords = self.cm.x[locations_outdoor[:, 1]]
        y_coords = self.cm.y[locations_outdoor[:, 0]]
        mask = (
            (x_coords >= x_limit_min) & (x_coords <= x_limit_max) &
            (y_coords >= y_limit_min) & (y_coords <= y_limit_max)
        )
        locations_outdoor_limited = locations_outdoor[mask]
        
        tx_ind = locations_outdoor_limited[
            np.random.choice(locations_outdoor_limited.shape[0], self.nbs, replace=False)
        ]

        # tx_ind = locations_outdoor[
        #     np.random.choice(locations_outdoor.shape[0], self.nbs, replace=False)
        # ]
        

        # If centerBS = True, force TX at (x=0, y=0). Otherwise random building coords.
        if centerBS:
            tx_x = 0
            tx_y = 0
            tx_z = np.where(
                self.cm.bldg_grid[tx_x, tx_y],
                self.cm.zmax_grid[tx_x, tx_y] + self.BS_height_above_roof,
                self.cm.zmin_grid[tx_x, tx_y] + self.BS_height_above_ground
            )
        else:
            tx_x = self.cm.x[tx_ind[:,1]]
            tx_y = self.cm.y[tx_ind[:,0]]
            tx_z = np.where(
                self.cm.bldg_grid[tx_ind[:, 0], tx_ind[:, 1]],
                self.cm.zmax_grid[tx_ind[:, 0], tx_ind[:, 1]] + self.BS_height_above_roof,
                self.cm.zmin_grid[tx_ind[:, 0], tx_ind[:, 1]] + self.BS_height_above_ground
            )
        self.tx_pos = np.column_stack((tx_x, tx_y, tx_z))

        # 3) Place NTN receivers (70% building, 30% outdoor)
        num_ntn_building = int(0.8 * ntn_rx)
        num_ntn_outdoor  = ntn_rx - num_ntn_building


        rx_ntn_building_ind = locations_building[
            np.random.choice(locations_building.shape[0], num_ntn_building, replace=False)]
        rx_ntn_outdoor_ind = locations_outdoor[
            np.random.choice(locations_outdoor.shape[0], num_ntn_outdoor, replace=False)]

        rx_ntn_ind = np.vstack((rx_ntn_building_ind, rx_ntn_outdoor_ind))

        rx_ntn_x = self.cm.x[rx_ntn_ind[:, 1]]
        rx_ntn_y = self.cm.y[rx_ntn_ind[:, 0]]
        rx_ntn_z = np.where(
            self.cm.bldg_grid[rx_ntn_ind[:, 0], rx_ntn_ind[:, 1]],
            self.cm.zmax_grid[rx_ntn_ind[:, 0], rx_ntn_ind[:, 1]] + self.ntn_height_above_roof,
            self.cm.zmin_grid[rx_ntn_ind[:, 0], rx_ntn_ind[:, 1]] + self.ntn_height_above_ground
        )
        self.rx_ntn_pos = np.column_stack((rx_ntn_x, rx_ntn_y, rx_ntn_z))

        # p1 = [-438.86474609, 954.68139648, 1.88]
        # p2 = [-832.86474609, 53.68139648, 1.88]

        # # sec2
        # p3 = [-310.86474609, -926.31860352, 1.88]
        # p4 = [-71.86474609, -636.31860352, 1.88]
        # p5 = [-677.86474609, -1027.31860352, 1.88]
        # p6 = [-729.86474609, -1069.31860352, 1.88]
        # p7 = [-268.86474609, -1116.31860352, 1.88]

        # # sec3
        # p8 = [1317.13525391, -1151.31860352, 1.88]
        # p9 = [786.13525391, 345.68139648, 1.88]
        # p10 = [1392.13525391, -369.31860352, 1.88]
        # p11 = [1253.13525391, -213.31860352, 1.88]
        # p12 = [1434.13525391, 762.68139648, 1.88]
        # p13 = [1116.13525391, -910.31860352, 1.88]
        # p14 = [1181.13525391, 91.68139648, 1.88]

        
        # self.rx_ntn_pos = np.array([p1,  p3, p11])
        # 4) Place TN receivers (outdoor), then filter by distance constraints
        
        
        num_tn_building = int(0.8 * tn_rx)
        num_tn_outdoor  = tn_rx - num_tn_building


        rx_tn_building_ind = locations_building[
            np.random.choice(locations_building.shape[0], num_tn_building, replace=False)]
        rx_tn_outdoor_ind = locations_outdoor[
            np.random.choice(locations_outdoor.shape[0], num_tn_outdoor, replace=False)]

        rx_tn_ind = np.vstack((rx_tn_building_ind, rx_tn_outdoor_ind))
        
        # rx_tn_ind = locations_outdoor[np.random.choice(locations_outdoor.shape[0], tn_rx, replace=False)]

        rx_tn_x = self.cm.x[rx_tn_ind[:, 1]]
        rx_tn_y = self.cm.y[rx_tn_ind[:, 0]]
        rx_tn_z = np.where(
            self.cm.bldg_grid[rx_tn_ind[:, 0], rx_tn_ind[:, 1]],
            self.cm.zmin_grid[rx_tn_ind[:, 0], rx_tn_ind[:, 1]] + self.tn_height_above_ground,
            self.cm.zmin_grid[rx_tn_ind[:, 0], rx_tn_ind[:, 1]] + self.tn_height_above_ground
        )
        rx_tn_pos = np.column_stack((rx_tn_x, rx_tn_y, rx_tn_z))

        # Filter out points not within [bs_dist_min, bs_dist_max] from TX
        bs_tn_dist = np.linalg.norm(rx_tn_pos - self.tx_pos, axis=1)
        indices = np.argwhere(
            (bs_tn_dist >= bs_dist_min) & (bs_tn_dist <= bs_dist_max)
        ).flatten()
        self.bs_tn_pos = rx_tn_pos[indices]
        
        # p1 = [1.18135254e+02, 5.34681396e+02, 1.79999995e+00]
        # # p2 = [1.16135254e+03, -3.76813965e+02, 1.79999995e+00]
        # p2 = [1.16135254e+03, 7.76813965e+02, 1.79999995e+00]

        # p3 = [-4.78864746e+02, 5.97681396e+02, 1.79999995e+00]
        # p4 = [-4.93864746e+02, 2.03681396e+02, 1.79999995e+00]

        # p5 = [-1.65486475e+03, -1.35131860e+03, 1.79999995e+00]
        # p5 = [-5e+02, -2e+03, 1.79999995e+00]
        
        # self.bs_tn_pos = np.array([p2,  p3, p5])

        # 5) Compute random satellite direction & project it to bounding box
        # azimuth = np.random.uniform(0, 360)
        # elevation = np.random.uniform(25, 90)
        x_proj, y_proj, z_proj = satellite_projection(
            azimuth,
            elevation,
            self.sat_distance,
            self.L_NS,
            self.W_WE
        )
        self.ntn_look_pos = np.array([x_proj, y_proj, z_proj])

    def compute_paths(self, tx_rows = 8, tx_cols = 8, tn_rx_rows = 4, tn_rx_cols = 4,max_depth=3,
                      bandwidth=100e6,l_min=-64, l_max=575 , pathstaps = False):
        """
        1) Configure scene frequency and remove old TX/RX
        2) Add TX, add TN array and receivers => compute TN CIR
        3) Remove TN, switch RX array to single-element custom => compute NTN CIR
        """
        self.scene.frequency = self.fc
        self.scene.synthetic_array = True
        # self.scene.bandwidth =  bandwidth
        # Remove existing TX and RX
        for rx_name in self.scene.receivers:
            self.scene.remove(rx_name)
        for tx_name in self.scene.transmitters:
            self.scene.remove(tx_name)

        # A. Set up the TX array
        self.scene.tx_array = PlanarArray(
            num_rows = tx_rows,
            num_cols = tx_cols,
            vertical_spacing=0.5,
            horizontal_spacing=0.5,
            polarization="V",
            pattern="tr38901"
        )

        # (1) Add Transmitters
        #    Use multiple sector approach for the single base station
        for i in range(self.nbs):
            for s in range(self.nsect):
                yaw = 2.0 * np.pi * s / self.nsect
                tx = sionna.rt.Transmitter(
                    name=f"tx-{s}",
                    position=self.tx_pos[i],
                    power_dbm=30,
                    # orientation=[yaw, 0, -0.1745329252] # headdown 10 degree or 5 degree -0.0873
                    # orientation=[yaw, 0,  -0.0873]
                    orientation=[yaw, -0.1745329252,  0]
                )
                self.scene.add(tx)
                    
        if self.tn_rx > 0:        
            # B. Set up the multi-element array for the TN side
            self.scene.rx_array = PlanarArray(
                num_rows = tn_rx_rows,
                num_cols = tn_rx_cols,
                vertical_spacing=0.5,
                horizontal_spacing=0.5,
                polarization="cross",
                # pattern="tr38901"
                pattern="dipole"
            )



            # (2) Add TN Receivers
            for i in range(self.bs_tn_pos.shape[0]):
                rx = sionna.rt.Receiver(
                    name=f"tn-{i}",
                    color=[0.0, 1.0, 0.0],
                    position=self.bs_tn_pos[i]
                )
                self.scene.add(rx)
                # let the receiver "look at" the BS center position
                rx.look_at(self.tx_pos.reshape(-1))
                
            # Compute paths for TN
            p_solver  = PathSolver()
            self.paths_tn = p_solver(scene=self.scene,
                                    max_depth=max_depth,
                                    los=True,
                                    specular_reflection=True,
                                    diffuse_reflection=False,
                                    refraction=True,
                                    synthetic_array=True)
                                    # seed=41)
        
            # Compute paths for TN
            self.a_tn, self.tau_tn = self.paths_tn.cir(normalize_delays=False, out_type="numpy")

        if self.ntn_rx > 0:
            for rx_name in self.scene.receivers:
                self.scene.remove(rx_name)

            self.scene.rx_array = PlanarArray(
                num_rows=1,
                num_cols=1,
                vertical_spacing=0.5,
                horizontal_spacing=0.5,
                pattern="vsat",
                polarization="V"
            )

            for i in range(self.rx_ntn_pos.shape[0]):
                rx = sionna.rt.Receiver(
                    name=f"ntn-{i}",
                    color=[1.0, 0.0, 0.0],
                    position=self.rx_ntn_pos[i]
                )
                self.scene.add(rx)
                rx.look_at(self.ntn_look_pos+self.rx_ntn_pos[i])

            p_solver  = PathSolver()
            self.paths_ntn = p_solver(scene=self.scene,
                                    max_depth=max_depth,
                                    los=True,
                                    specular_reflection=True,
                                    diffuse_reflection=False,
                                    refraction=True,
                                    synthetic_array=True)
                                    # seed=41)
            
            # Compute paths for TN
            
            if pathstaps == True:
                self.a_ntn, self.tau_ntn = self.paths_ntn.cir(normalize_delays=False, reverse_direction = True, out_type="numpy")
                self.taps_ntn = self.paths_ntn.taps(bandwidth=bandwidth, # Bandwidth to which the channel is low-pass filtered
                  l_min=l_min,        # Smallest time lag
                  l_max=l_max,       # Largest time lag
                  sampling_frequency=None, # Sampling at Nyquist rate, i.e., 1/bandwidth
                  normalize=False,  # Normalize energy
                  normalize_delays=False,
                  reverse_direction = True,
                  out_type="numpy")
                # Convert to tf.Tensor
                a = tf.convert_to_tensor(self.a_ntn, dtype=tf.complex64)
                tau = tf.convert_to_tensor(self.tau_ntn, dtype=tf.float32)
                a = tf.expand_dims(a, axis=0)      # shape += 1 at axis=0
                tau = tf.expand_dims(tau, axis=0)                
                self.h_time = cir_to_time_channel( bandwidth, a, tau, 
                                                    l_min=l_min, l_max=l_max, normalize=False).numpy()
                

            else:
                self.a_ntn, self.tau_ntn = self.paths_ntn.cir(
                    normalize_delays=False,
                    out_type="numpy"
                )

        