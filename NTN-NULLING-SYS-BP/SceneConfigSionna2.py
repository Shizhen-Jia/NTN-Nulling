import os
os.environ.pop("TF_USE_LEGACY_KERAS", None)   # 或 os.environ["TF_USE_LEGACY_KERAS"]="0"

if os.getenv("CUDA_VISIBLE_DEVICES") is None:
    gpu_num = 0 # Use "" to use the CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)
# Avoid warnings from TensorFlow
tf.get_logger().setLevel('ERROR')
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




# Define constants (adjust these as needed)
dish_diameter = 0.30        # meters
tx_gain_dB = 38.1           # dBi
c = 3e8                     # speed of light (m/s)
tx_frequency_mid = np.mean([13.75, 14.5]) * 1e9  # mid frequency in Hz
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
    def __init__(self, scene):
        """
        scene : A sionna.rt.Scene object with loaded geometry (XML or otherwise).
        """
        self.scene = scene
        
        self.fc = 10e9

        # Some default parameters you can modify:
        self.grid_size = 1.0
        self.nbs = None               # Number of gNB/base stations
        self.nsect = None              # Usually 3, for sector-based coverage
        self.BS_height_above_roof = 35  # For base station on building
        self.BS_height_above_ground = 45
        self.tn_height_above_roof = 1.2  # For base station on building
        self.tn_height_above_ground = 1.8
        self.ntn_height_above_roof = 1.2  # For base station on building
        self.ntn_height_above_ground = 1.88
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
        self.tn_pos = None
        self.ntn_look_pos = None

        # Path results
        self.a_tn = None
        self.tau_tn = None
        self.a_ntn = None
        self.tau_ntn = None
        self.ntn_rx = None
        
        self.toff = None
        self.h_tf = None
        self.tn_bs_index = None
        self.tn_sector_index = None

    def build_coverage_map(self, grid_size=None, show_xy=False, plot=False):
        """
        Build coverage map, compute bbox/extent/point_type.
        Optional: print x/y ranges and plot building/outdoor map.
        """
        if grid_size is not None:
            self.grid_size = grid_size

        self.cm = CoverageMapPlanner(self.scene._scene, grid_size=self.grid_size)
        self.cm.set_grid()
        self.cm.compute_grid_attributes()
        self.hm = None

        x_min, x_max = self.cm.x[0], self.cm.x[-1]
        y_min, y_max = self.cm.y[0], self.cm.y[-1]

        self.W_WE = x_max - x_min   # width (East-West)
        self.L_NS = y_max - y_min   # length (North-South)
        self.bbox = [-self.W_WE/2, self.W_WE/2, -self.L_NS/2, self.L_NS/2]
        # self.extent = [self.cm.x[0], self.cm.x[-1], self.cm.y[0], self.cm.y[-1]]
        self.extent = [x_min, x_max, y_min, y_max]

        building = (self.cm.bldg_grid)
        self.point_type = np.where(building, 2, 1).astype(np.int8)
        self.point_type = np.flipud(self.point_type)
        # outdoor = (self.cm.bldg_grid == False)
        # self.point_type = outdoor + 2 * building
        # self.point_type = self.point_type.astype(int)
        # self.point_type = np.flipud(self.point_type)

        if show_xy:
            print(f"x range: [{x_min:.3f}, {x_max:.3f}]")
            print(f"y range: [{y_min:.3f}, {y_max:.3f}]")

        if plot:
            self._plot_grid()

    def _plot_grid(self, show_bs=False, show_tn=False, show_ntn=False):
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        colors = ['lightgray', 'brown']
        cmap = ListedColormap(colors)
        plt.figure()
        plt.imshow(self.point_type, cmap=cmap, interpolation='nearest', extent=self.extent)
        if show_bs and self.tx_pos is not None:
            bs_marker = (3, 0, -30)  # triangle rotated 30 degrees clockwise
            plt.scatter(self.tx_pos[:, 0], self.tx_pos[:, 1], c="red", s=60, marker=bs_marker, label="BS")
        if show_tn and self.tn_pos is not None:
            plt.scatter(self.tn_pos[:, 0], self.tn_pos[:, 1], c="green", s=20, marker="o", label="TN")
        if show_ntn and self.rx_ntn_pos is not None:
            plt.scatter(self.rx_ntn_pos[:, 0], self.rx_ntn_pos[:, 1], c="blue", s=20, marker="x", label="NTN")
        has_overlays = (
            (show_bs and self.tx_pos is not None)
            or (show_tn and self.tn_pos is not None)
            or (show_ntn and self.rx_ntn_pos is not None)
        )
        if has_overlays:
            plt.legend(loc="upper right")
        plt.xlabel("x")
        plt.ylabel("y")
        if has_overlays:
            plt.title("Spatial Distribution of TN/NTN UEs and BSs")
        else:
            plt.title("Building/Outdoor Map")
        plt.show()

    def _snap_to_grid(self, x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        ix = np.searchsorted(self.cm.x, x)
        iy = np.searchsorted(self.cm.y, y)
        ix = np.clip(ix, 0, len(self.cm.x) - 1)
        iy = np.clip(iy, 0, len(self.cm.y) - 1)

        xg = self.cm.x[ix]
        yg = self.cm.y[iy]
        z = np.where(
            self.cm.bldg_grid[iy, ix],
            self.cm.zmax_grid[iy, ix] + self.BS_height_above_roof,
            self.cm.zmin_grid[iy, ix] + self.BS_height_above_ground
        )
        return xg, yg, z

    def _positions_from_indices(self, indices, height_roof, height_ground):
        x = self.cm.x[indices[:, 1]]
        y = self.cm.y[indices[:, 0]]
        z = np.where(
            self.cm.bldg_grid[indices[:, 0], indices[:, 1]],
            self.cm.zmax_grid[indices[:, 0], indices[:, 1]] + height_roof,
            self.cm.zmin_grid[indices[:, 0], indices[:, 1]] + height_ground
        )
        return np.column_stack((x, y, z))

    def compute_positions(self,
                        ntn_rx,
                        tn_rx,
                        azimuth,
                        elevation,
                        centerBS=True,
                        bs_dist_min=35,
                        bs_dist_max=1000,
                        bs_boundary=0.0,
                        bs_layout="random",
                        bs_grid=None,
                        nbs=None,
                        tn_distance=1500.0,
                        tn_building_ratio="sector",
                        ntn_building_ratio=None,
                        show_xy=False,
                        plot_grid=False,
                        plot_bs=False,
                        plot_tn=False,
                        plot_ntn=False):
        """
        1) Build coverage map and determine bounding box
        2) Select random positions for TX, TN/NTN receivers
        3) Optionally place TX at (x=0, y=0) if centerBS=True
        4) Filter TN receivers by distance constraints
        5) Compute random satellite direction & project
        """

        # 1) Create/reuse coverage map
        self.ntn_rx = ntn_rx
        if self.cm is None or self.point_type is None or self.extent is None:
            self.build_coverage_map(show_xy=show_xy, plot=False)
        else:
            if show_xy:
                x_min, x_max = self.cm.x[0], self.cm.x[-1]
                y_min, y_max = self.cm.y[0], self.cm.y[-1]
                print(f"x range: [{x_min:.3f}, {x_max:.3f}]")
                print(f"y range: [{y_min:.3f}, {y_max:.3f}]")
        x_min, x_max = self.cm.x[0], self.cm.x[-1]
        y_min, y_max = self.cm.y[0], self.cm.y[-1]

        # 2) Place a single gNB TX position on building roof
        # locations_building = np.argwhere(self.cm.bldg_grid & self.cm.in_region)        
        locations_building = np.argwhere(self.cm.bldg_grid)
        locations_outdoor = np.argwhere(~self.cm.bldg_grid)
        
        # if len(locations_building) < self.nbs:
        #     raise ValueError("Not enough building points to place the TX.")

        # If centerBS=True, force a single BS at (0,0) regardless of bs_grid
        if centerBS:
            self.nbs = 1
        elif bs_grid is not None:
            nx, ny = bs_grid
            if nx <= 0 or ny <= 0:
                raise ValueError("bs_grid must be positive, e.g. (2, 2).")
            self.nbs = int(nx * ny)
        elif nbs is not None:
            self.nbs = int(nbs)
        elif self.nbs is None:
            raise ValueError("nbs is required when bs_grid is not set.")

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
        

        # If centerBS = True and only 1 BS, force TX at (0, 0).
        if centerBS and self.nbs == 1:
            tx_x = 0
            tx_y = 0
            tx_z = np.where(
                self.cm.bldg_grid[tx_x, tx_y],
                self.cm.zmax_grid[tx_x, tx_y] + self.BS_height_above_roof,
                self.cm.zmin_grid[tx_x, tx_y] + self.BS_height_above_ground
            )
        elif bs_grid is not None:
            x_start = x_min + bs_boundary
            x_end = x_max - bs_boundary
            y_start = y_min + bs_boundary
            y_end = y_max - bs_boundary
            if x_end <= x_start or y_end <= y_start:
                raise ValueError("bs_boundary too large: no valid x/y range.")
            xs = np.linspace(x_start, x_end, nx)
            ys = np.linspace(y_start, y_end, ny)
            xx, yy = np.meshgrid(xs, ys)
            tx_x = xx.ravel()
            tx_y = yy.ravel()
            tx_x, tx_y, tx_z = self._snap_to_grid(tx_x, tx_y)
        elif bs_layout == "line":
            # Place BSs evenly on x-axis within boundary, y=0
            x_start = x_min + bs_boundary
            x_end = x_max - bs_boundary
            if x_end <= x_start:
                raise ValueError("bs_boundary too large: no valid x range.")
            tx_x = np.linspace(x_start, x_end, self.nbs)
            tx_y = np.zeros_like(tx_x)
            tx_x, tx_y, tx_z = self._snap_to_grid(tx_x, tx_y)
        else:
            tx_x = self.cm.x[tx_ind[:,1]]
            tx_y = self.cm.y[tx_ind[:,0]]
            tx_z = np.where(
                self.cm.bldg_grid[tx_ind[:, 0], tx_ind[:, 1]],
                self.cm.zmax_grid[tx_ind[:, 0], tx_ind[:, 1]] + self.BS_height_above_roof,
                self.cm.zmin_grid[tx_ind[:, 0], tx_ind[:, 1]] + self.BS_height_above_ground
            )
        self.tx_pos = np.column_stack((tx_x, tx_y, tx_z))


        # 3) Place NTN receivers
        if ntn_building_ratio is None:
            num_ntn_building = None
            num_ntn_outdoor = None
        else:
            num_ntn_building = int(round(ntn_building_ratio * ntn_rx))
            num_ntn_outdoor = ntn_rx - num_ntn_building

        if ntn_building_ratio is None:
            # Pure random from all points
            all_idx = np.argwhere(self.cm.bldg_grid | (~self.cm.bldg_grid))
            if ntn_rx > all_idx.shape[0]:
                candidate_indices = all_idx[
                    np.random.choice(all_idx.shape[0], ntn_rx, replace=True)
                ]
            else:
                candidate_indices = all_idx[
                    np.random.choice(all_idx.shape[0], ntn_rx, replace=False)
                ]
        else:
            # Random with building/outdoor ratio
            rx_ntn_building_ind = locations_building[
                np.random.choice(
                    locations_building.shape[0],
                    num_ntn_building,
                    replace=num_ntn_building > locations_building.shape[0]
                )
            ] if num_ntn_building > 0 else np.empty((0, 2), dtype=int)
            rx_ntn_outdoor_ind = locations_outdoor[
                np.random.choice(
                    locations_outdoor.shape[0],
                    num_ntn_outdoor,
                    replace=num_ntn_outdoor > locations_outdoor.shape[0]
                )
            ] if num_ntn_outdoor > 0 else np.empty((0, 2), dtype=int)
            candidate_indices = np.vstack((rx_ntn_building_ind, rx_ntn_outdoor_ind))

        def filter_indices(indices):
            # 根据 cm 坐标映射获得 x, y
            rx_ntn_x = self.cm.x[indices[:, 1]]
            rx_ntn_y = self.cm.y[indices[:, 0]]
            # 过滤条件：要求不在中心区域内
            # 比如，中心 500x500 米区域：x 和 y 同时满足 |x|<250 且 |y|<250
            mask = ~((np.abs(rx_ntn_x) < 250) & (np.abs(rx_ntn_y) < 800))
            return indices[mask]

        # 初步过滤
        filtered_indices = filter_indices(candidate_indices)

        # 如果数量不足 ntn_rx，则不断补充
        while filtered_indices.shape[0] < ntn_rx:
            # 为补充，按照比例重新采样一些候选点
            # 注意：为了防止 replace=False 时候候选点不足，可以设置 replace=True，
            # 但这可能会有重复，最后再使用 np.unique 去重
            extra_building = locations_building[
                np.random.choice(locations_building.shape[0], max(1, int(0.8 * (ntn_rx - filtered_indices.shape[0])),), replace=True)
            ]
            extra_outdoor = locations_outdoor[
                np.random.choice(locations_outdoor.shape[0], max(1, int(0.2 * (ntn_rx - filtered_indices.shape[0])),), replace=True)
            ]
            extra_candidates = np.vstack((extra_building, extra_outdoor))
            extra_filtered = filter_indices(extra_candidates)
            # 合并，去重
            filtered_indices = np.vstack((filtered_indices, extra_filtered))
            # 去重（因为索引是整数数组，这里可以使用 np.unique ）
            filtered_indices = np.unique(filtered_indices, axis=0)

        # 最终只保留前 ntn_rx 个（如果多于 ntn_rx 个）
        if filtered_indices.shape[0] > ntn_rx:
            filtered_indices = filtered_indices[:ntn_rx]

        # 更新映射后的 x, y 坐标
        rx_ntn_x = self.cm.x[filtered_indices[:, 1]]
        rx_ntn_y = self.cm.y[filtered_indices[:, 0]]
        rx_ntn_z = np.where(
            self.cm.bldg_grid[filtered_indices[:, 0], filtered_indices[:, 1]],
            self.cm.zmax_grid[filtered_indices[:, 0], filtered_indices[:, 1]] + self.ntn_height_above_roof,
            self.cm.zmin_grid[filtered_indices[:, 0], filtered_indices[:, 1]] + self.ntn_height_above_ground
        )
        self.rx_ntn_pos = np.column_stack((rx_ntn_x, rx_ntn_y, rx_ntn_z))
        
        
        # 4) Place TN receivers
        if tn_building_ratio == "sector":
            # For each BS, place 3 TN at 0/120/240 deg, distance tn_distance
            angles_deg = np.array([0.0, 120.0, 240.0])
            angles_rad = np.deg2rad(angles_deg)
            offsets = np.stack([np.cos(angles_rad), np.sin(angles_rad)], axis=1) * tn_distance
            bs_xy = self.tx_pos[:, :2]
            tn_xy = (bs_xy[:, None, :] + offsets[None, :, :]).reshape(-1, 2)
            tn_x = np.clip(tn_xy[:, 0], x_min, x_max)
            tn_y = np.clip(tn_xy[:, 1], y_min, y_max)
            tn_x, tn_y, tn_z = self._snap_to_grid(tn_x, tn_y)
            self.tn_pos = np.column_stack((tn_x, tn_y, tn_z))
        elif tn_building_ratio is None:
            # Random TN from all grid points
            if tn_rx <= 0:
                self.tn_pos = np.empty((0, 3))
            else:
                all_idx = np.argwhere(self.cm.bldg_grid | (~self.cm.bldg_grid))
                if tn_rx > all_idx.shape[0]:
                    chosen = all_idx[np.random.choice(all_idx.shape[0], tn_rx, replace=True)]
                else:
                    chosen = all_idx[np.random.choice(all_idx.shape[0], tn_rx, replace=False)]
                self.tn_pos = self._positions_from_indices(
                    chosen, self.tn_height_above_roof, self.tn_height_above_ground
                )
        else:
            # Random TN with building ratio
            if tn_rx <= 0:
                self.tn_pos = np.empty((0, 3))
            else:
                tn_building = int(round(float(tn_building_ratio) * tn_rx))
                tn_outdoor = tn_rx - tn_building
                bldg_idx = np.argwhere(self.cm.bldg_grid)
                out_idx = np.argwhere(~self.cm.bldg_grid)
                bldg_sel = bldg_idx[
                    np.random.choice(bldg_idx.shape[0], tn_building, replace=tn_building > bldg_idx.shape[0])
                ] if tn_building > 0 else np.empty((0, 2), dtype=int)
                out_sel = out_idx[
                    np.random.choice(out_idx.shape[0], tn_outdoor, replace=tn_outdoor > out_idx.shape[0])
                ] if tn_outdoor > 0 else np.empty((0, 2), dtype=int)
                chosen = np.vstack((bldg_sel, out_sel)) if tn_rx > 0 else np.empty((0, 2), dtype=int)
                self.tn_pos = self._positions_from_indices(
                    chosen, self.tn_height_above_roof, self.tn_height_above_ground
                )

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

        if plot_grid or plot_bs or plot_tn or plot_ntn:
            self._plot_grid(show_bs=plot_bs, show_tn=plot_tn, show_ntn=plot_ntn)
        
        
        
        
        

    def compute_paths(self, nsect, fc, tx_rows = 8, tx_cols = 8, tn_rx_rows = 1, tn_rx_cols = 1, max_depth=3,
                      bandwidth=100e6, tx_power_dbm=30):
        """
        1) Configure scene frequency and remove old TX/RX
        2) Add TX, add TN array and receivers => compute TN CIR
        3) Remove TN, switch RX array to single-element custom => compute NTN CIR
        """
        self.fc = fc
        self.nsect = nsect
        self.scene.frequency = self.fc
        self.scene.bandwidth = bandwidth
        self.scene.synthetic_array = True

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

        # B. Set up the multi-element array for the TN side
        self.scene.rx_array = PlanarArray(
            num_rows = tn_rx_rows,
            num_cols = tn_rx_cols,
            vertical_spacing=0.5,
            horizontal_spacing=0.5,
            polarization="V",
            # pattern="tr38901"
            pattern="dipole"
        )

        # (1) Add Transmitters
        #    Use multiple sector approach for the single base station
        for i in range(self.nbs):
            for s in range(self.nsect):
                yaw = 2.0 * np.pi * s / self.nsect
                tx = sionna.rt.Transmitter(
                    name=f"tx-{i}-{s}",
                    position=self.tx_pos[i],
                    power_dbm=tx_power_dbm,
                    orientation=[yaw, -0.0873 , 0] # headdown 10 degree,  -0.0873
                )
                self.scene.add(tx)

        # (2) Add TN Receivers
        # Pair each TN with nearest BS and store index + sector index
        if self.tx_pos is None or self.tx_pos.shape[0] == 0:
            raise ValueError("tx_pos is empty; run compute_positions before compute_paths.")
        diffs = self.tn_pos[:, None, :2] - self.tx_pos[None, :, :2]
        d2 = np.sum(diffs**2, axis=2)
        self.tn_bs_index = np.argmin(d2, axis=1)
        # sector index based on angle from BS to TN
        bs_xy = self.tx_pos[:, :2]
        tn_xy = self.tn_pos[:, :2]
        rel = tn_xy - bs_xy[self.tn_bs_index]
        ang = np.mod(np.arctan2(rel[:, 1], rel[:, 0]), 2*np.pi)
        sector_yaw = 2.0 * np.pi * np.arange(self.nsect) / self.nsect
        # nearest sector center (circular distance)
        dtheta = np.abs(ang[:, None] - sector_yaw[None, :])
        dtheta = np.minimum(dtheta, 2*np.pi - dtheta)
        self.tn_sector_index = np.argmin(dtheta, axis=1)
        for i in range(self.tn_pos.shape[0]):
            rx = sionna.rt.Receiver(
                name=f"tn-{i}",
                color=[0.0, 1.0, 0.0],
                position=self.tn_pos[i]
            )
            self.scene.add(rx)
            # Optional: look at nearest BS position (not strictly required)
            rx.look_at(self.tx_pos[self.tn_bs_index[i]])
            
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
        # [num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps]
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
                rx.look_at([0,0,20000])
                # rx.look_at(self.ntn_look_pos+self.rx_ntn_pos[i])

            
            self.paths_ntn = p_solver(scene=self.scene,
                                    max_depth=max_depth,
                                    los=True,
                                    specular_reflection=True,
                                    diffuse_reflection=False,
                                    refraction=True,
                                    synthetic_array=True)
                                    # seed=41)
            
            # Compute paths for TN

            self.a_ntn, self.tau_ntn = self.paths_ntn.cir(normalize_delays=False, out_type="numpy")
