# satellite_projection.py

import numpy as np

def satellite_projection(azimuth, elevation, distance, L, W):
    """
    Calculate the satellite coordinates above the region and project them 
    onto the boundary if the satellite is outside that region.

    :param azimuth: Azimuth angle (degrees)
    :param elevation: Elevation angle (degrees)
    :param distance: Distance from the satellite to the origin (m)
    :param L: North-South length of the region (m)
    :param W: East-West width of the region (m)
    :return: (x_proj, y_proj, z_proj) 
             Adjusted satellite coordinates within the bounding box.
    """
    # Convert angles to radians
    azimuth_rad = np.radians(azimuth)
    elevation_rad = np.radians(elevation)
    
    # Calculate satellite Cartesian coordinates
    x = distance * np.cos(elevation_rad) * np.sin(azimuth_rad)
    y = distance * np.cos(elevation_rad) * np.cos(azimuth_rad)
    z = distance * np.sin(elevation_rad)
    
    # If z > 100, project to z=100
    if z > 100:
        alpha = 100.0 / z
        x *= alpha
        y *= alpha
        z = 100.0

    # Clamp x, y
    x_min, x_max = -W / 2.0, W / 2.0
    y_min, y_max = -L / 2.0, L / 2.0
    x = np.clip(x, x_min, x_max)
    y = np.clip(y, y_min, y_max)

    return x, y, z
    
    # # Define boundary limits
    # x_min, x_max = -W / 2, W / 2
    # y_min, y_max = -L / 2, L / 2
    
    # # Adjust coordinates based on boundary conditions
    # x_proj, y_proj, z_proj = x, y, z
    # if x < x_min:
    #     x_proj = x_min
    # elif x > x_max:
    #     x_proj = x_max
    
    # if y < y_min:
    #     y_proj = y_min
    # elif y > y_max:
    #     y_proj = y_max
    
    # # Calculate the projected height on the boundary
    # if x != 0 or y != 0:
    #     # handle edge case for zero denominator
    #     ratio_x = (x_proj - 0) / x if x != 0 else float('inf')
    #     ratio_y = (y_proj - 0) / y if y != 0 else float('inf')
    #     ratio   = min(ratio_x, ratio_y)
        
    #     if 0 < ratio <= 1:
    #         z_proj = ratio * z
    
    return x_proj, y_proj, z_proj

def steering_vector(theta, phi, num_rows, num_cols, fc=10e9):
    """
    Compute the beamforming steering vector for a recntn_look_postangular antenna array.
    
    :param theta: Elevation angle in degrees
    :param phi: Azimuth angle in degrees
    :param num_rows: Number of rows in the array
    :param num_cols: Number of columns in the array
    :param d: Element spacing, default is 0.5 wavelength
    :param fc: Carrier frequency in Hz, default is 3 GHz
    :return: Normalized steering vector of shape (num_rows*num_cols, 1)
    """
    d=0.5
    c = 3e8  # Speed of light in m/s
    wavelength = c / fc  # Compute wavelength from frequency
    k = 2 * np.pi / wavelength  # Wavenumber
    
    # Convert angles to radians
    theta_rad = np.radians(theta)
    phi_rad = np.radians(phi)
    
    # Compute direction cosines
    ux = np.sin(theta_rad) * np.cos(phi_rad)
    uy = np.sin(theta_rad) * np.sin(phi_rad)
    
    # Generate rectangular antenna coordinate indices
    x = np.arange(num_cols)
    y = np.arange(num_rows)
    X, Y = np.meshgrid(x, y)
    
    # Compute array response
    phase_shift = k * d * (X * ux + Y * uy)
    SV = np.exp(1j * phase_shift)
    
    return SV.flatten().reshape(-1, 1)  # Return as a column vector