# NTN-Nulling

This repository contains simulation tools and notebooks for evaluating interference nulling from terrestrial base stations to Non-Terrestrial Network User Equipments (NTN-UEs). The focus is on analyzing nulling performance under different lambda values, antenna configurations, and detection strategies.

## 📁 Project Structure

### 📊 Nulling CDF Evaluation

- **Lambda_CDF.ipynb**  
  Computes the CDF of nulling gain under different lambda values using perfect channel knowledge (threshold set to the true channel gain).

- **Lambda_CDF_det.ipynb**  
  Computes the CDF of nulling gain using estimated channels (`h_hat`) and thresholds derived from a false alarm probability.

### 📡 Antenna Configuration

- **Large_Antenna_Array.ipynb**  
  Demonstrates a 32×32 antenna array setup for nulling and beamforming evaluation.

### 🛰️ Satellite Geometry and Scene Setup

- **satellite_projection.py**  
  Computes the satellite "look" direction and position for NTN-UEs.

- **SceneConfigSionna.py**  
  Used by `Lambda_CDF.ipynb` to configure the ray tracing scene for nulling evaluations.

- **SceneConfigSionna2.py**  
  Used by `Pos_for_RM.ipynb` to configure scenarios with selected user positions.

- **sionnautils/**  
  Utility functions used by `SceneConfigSionna.py` for generating grids and locating TN/NTN UEs.

### 🗺️ Mapping and Visualization

- **map_and_pics.ipynb**  
  Plots the radio map, ray tracing map, and satellite visibility map for the simulated area. 

- **Pos_for_RM.ipynb**  
  Computes user positions and nulling vectors, serving as input to `map_and_pics.ipynb`.(In `SceneConfigSionna2.py` manually picked three TN-UEs for three sectors ), saved TN-UE postions, NTN-UE postions and Beamforming Vectors.

### 🔍 Signal Detection

- **detection_use_time_channel.ipynb**  
  Implements signal detection using time-domain CIR (`cir_to_time`) and analyzes paths and tap delay profiles.

## 📌 Notes

- All simulations assume a known or estimated channel model.
- Nulling vectors are computed using SVD-based techniques.
- Visualizations require matplotlib and ray-tracing output from the Sionna simulator.

## 🔧 Requirements

- Python ≥ 3.8
- [Sionna RT](https://nvlabs.github.io/sionna/)
- NumPy, Matplotlib, Jupyter

```
python -m venv sionna_env
source sionna_env/bin/activate
pip install --upgrade pip tensorflow tqdm numpy matplotlib importlib_metadata sionna sionna-rt
```  

## 🔁 Replace Sionna `radio_map.py`

This project uses a customized version of Sionna's `radio_map.py`.
After installing Sionna, replace the original file in the Sionna source tree:

- Replace `sionna/rt/radio_map.py`
- With this repository's `radio_map.py`

The modified `radio_map.py` adds the `inr_ntn` metric for NTN interference analysis.
Unlike the default `sinr`, `inr_ntn` treats the total received power from all base stations as interference and computes:

```python
inr_ntn = interference / noise
```

This metric can be used in radio map related evaluation and visualization workflows that accept `metric`, together with `path_gain`, `rss`, and `sinr`.

If you are using a virtual environment, the Sionna file is typically located at:

```bash
<your_venv>/lib/python3.x/site-packages/sionna/rt/radio_map.py
```

You can also locate it with:

```bash
python -c "import sionna, pathlib; print(pathlib.Path(sionna.__file__).resolve().parent / 'rt' / 'radio_map.py')"
```

---

