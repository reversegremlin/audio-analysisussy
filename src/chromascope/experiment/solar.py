"""
Solar visualizer.
"""

import numpy as np
import noise
from PIL import Image, ImageDraw, ImageFilter
import math
import random

# Colormaps
COLORMAPS = {
    "default": [
        (0, (0, 0, 0)),
        (0.5, (255, 0, 0)),
        (0.8, (255, 255, 0)),
        (0.95, (255, 255, 255)),
    ],
    "blue": [
        (0, (0, 0, 0)),
        (0.5, (0, 0, 255)),
        (0.8, (0, 255, 255)),
        (0.95, (255, 255, 255)),
    ],
    "green": [
        (0, (0, 0, 0)),
        (0.5, (0, 255, 0)),
        (0.8, (255, 255, 0)),
        (0.95, (255, 255, 255)),
    ],
    "flame": [ # A more fiery colormap
        (0, (10, 0, 0)),
        (0.3, (150, 0, 0)),
        (0.6, (255, 100, 0)),
        (0.8, (255, 255, 50)),
        (1.0, (255, 255, 255)),
    ],
    "solar_flare": [ # A colormap designed for flares, with more white and intense yellows
        (0, (30, 0, 0)),
        (0.3, (220, 60, 0)),
        (0.6, (255, 180, 0)),
        (0.8, (255, 255, 50)),
        (1.0, (255, 255, 200)),
    ]
}


class SolarRenderer:
    def __init__(self, config):
        self.config = config
        self.width = config.width
        self.height = config.height
        self.time = 0.0
        self.camera_x = 0.0
        self.camera_y = 0.0
        self.camera_zoom = 1.0
        self.colormap_name = config.colormap
        self.base_colormap = COLORMAPS.get(config.colormap, COLORMAPS["default"])

        # For localized hot spots
        self.hot_spot_center = (0, 0)
        self.hot_spot_intensity = 0.0
        self.hot_spot_decay_rate = 0.85 # Faster decay for flares

        # For solar flares
        self.flares = [] # List to hold active flares

        # Sun properties (dynamically updated per frame)
        self.sun_center = (self.width // 2, self.height // 2)
        self.sun_radius = min(self.width, self.height) // 4


    def _update_camera(self, frame_index, global_energy, high_energy):
        """
        Updates camera position and zoom for a cinematic effect, influenced by audio energy.
        """
        # Base pan using sine waves
        base_pan_x = math.sin(frame_index * self.config.pan_speed_x * (math.pi / 180)) * 100
        base_pan_y = math.cos(frame_index * self.config.pan_speed_y * (math.pi / 180)) * 100

        # Add audio-reactive shake/jitter, more sensitive to high_energy
        shake_intensity = (global_energy * 75) + (high_energy * 150)
        self.camera_x = base_pan_x + (np.random.rand() - 0.5) * shake_intensity
        self.camera_y = base_pan_y + (np.random.rand() - 0.5) * shake_intensity

        # Base zoom using a sine wave, further influenced by global energy and high_energy
        base_zoom = 1.0 + math.sin(frame_index * self.config.zoom_speed * (math.pi / 180)) * 0.5
        self.camera_zoom = base_zoom + (global_energy * 0.8) + (high_energy * 1.5) # Global and high energy add more zoom

    def _generate_noise_layer(self, t, scale, octaves, persistence, lacunarity, offset_x=0, offset_y=0, angle=0.0):
        """Generates a single Perlin noise layer with optional rotation."""
        shape = (self.height, self.width)
        layer = np.zeros(shape)
        cos_angle = math.cos(angle)
        sin_angle = math.sin(angle)

        center_x, center_y = self.width / 2, self.height / 2

        for i in range(shape[0]):
            for j in range(shape[1]):
                # Translate to origin, rotate, then translate back
                translated_x = j - center_x
                translated_y = i - center_y

                rotated_x = translated_x * cos_angle - translated_y * sin_angle
                rotated_y = translated_x * sin_angle + translated_y * cos_angle

                # Apply camera zoom to the coordinates
                nx = ((rotated_x + center_x + offset_x) * scale / self.width) * self.camera_zoom
                ny = ((rotated_y + center_y + offset_y) * scale / self.height) * self.camera_zoom
                
                layer[i][j] = noise.pnoise3(
                    ny,
                    nx,
                    t,
                    octaves=octaves,
                    persistence=persistence,
                    lacunarity=lacunarity,
                    repeatx=self.width,
                    repeaty=self.height,
                    base=0,
                )
        return layer

    def _generate_hot_spot(self):
        """Generates a localized radial gradient for a hot spot."""
        hot_spot_img = np.zeros((self.height, self.width))
        if self.hot_spot_intensity > 0.01: # Only generate if visible
            cx, cy = self.hot_spot_center # Use cx, cy for consistency with PIL
            y, x = np.ogrid[-cy:self.height-cy, -cx:self.width-cx]
            
            distance = np.sqrt(x*x + y*y)
            max_dist = min(self.width, self.height) / 8 # Smaller radius for hot spots
            
            gradient = 1.0 - np.clip(distance / max_dist, 0, 1)
            
            hot_spot_img = gradient * self.hot_spot_intensity
            
            self.hot_spot_intensity *= self.hot_spot_decay_rate
        return hot_spot_img
    
    def _quadratic_bezier(self, p0, p1, p2, t):
        """Calculates a point on a quadratic Bezier curve."""
        return (1 - t)**2 * p0 + 2 * (1 - t) * t * p1 + t**2 * p2

    def _generate_arcing_flare_image(self, flare_data):
        """
        Generates a single arcing flare as a PIL Image with transparency.
        """
        flare_image = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(flare_image)

        start_point = flare_data["start_point"]
        control_point = flare_data["control_point"]
        end_point = flare_data["end_point"]
        max_width = flare_data["max_width"]
        current_intensity = flare_data["current_intensity"]
        color = flare_data["color"]

        num_segments = 20 # Number of segments to approximate the Bezier curve
        points = []
        for i in range(num_segments + 1):
            t = i / num_segments
            x = self._quadratic_bezier(start_point[0], control_point[0], end_point[0], t)
            y = self._quadratic_bezier(start_point[1], control_point[1], end_point[1], t)
            points.append((x, y))

        # Create glow layer
        glow_image = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        draw_glow = ImageDraw.Draw(glow_image)

        for i in range(len(points) - 1):
            p1 = points[i]
            p2 = points[i+1]
            
            width_interp = 1 - (i / (len(points) - 1))
            
            # Glow is wider and more transparent
            glow_width = max(1, int(max_width * width_interp * 2.0)) # Double the width for glow
            glow_alpha = int(255 * current_intensity * width_interp * 0.4) # Lower opacity for glow

            segment_color_glow = color + (glow_alpha,)
            draw_glow.line([p1, p2], fill=segment_color_glow, width=glow_width, joint="curve")
        
        # Apply blur to the glow layer
        glow_image = glow_image.filter(ImageFilter.GaussianBlur(radius=max_width * 1.5))

        # Draw the main flare with varying width and opacity
        for i in range(len(points) - 1):
            p1 = points[i]
            p2 = points[i+1]

            width_interp = 1 - (i / (len(points) - 1))
            segment_width = max(1, int(max_width * width_interp))
            
            segment_alpha = int(255 * current_intensity * width_interp)

            segment_color = color + (segment_alpha,)
            draw.line([p1, p2], fill=segment_color, width=segment_width, joint="curve")
        
        # Apply blur to the main flare (less than glow)
        flare_image = flare_image.filter(ImageFilter.GaussianBlur(radius=max_width * 0.5))

        # Composite glow and main flare
        final_flare_image = Image.alpha_composite(glow_image, flare_image)
        
        return final_flare_image


    def _get_sun_params(self, low_energy, percussive_impact):
        """Calculates current sun center and radius."""
        center_x, center_y = self.width // 2, self.height // 2
        base_radius = min(self.width, self.height) // 4
        radius_pulse = (low_energy * 75) + (percussive_impact * 150)
        radius = int(base_radius * self.camera_zoom + radius_pulse)
        return (center_x, center_y), radius

    def _generate_sun_texture(self, t, global_energy, harmonic_energy, percussive_impact, is_beat, is_onset, high_energy):
        """
        Generates a dynamic sun texture using multiple Perlin noise layers,
        influenced by various audio features, and adds localized hot spots/flares.
        """
        # Determine a general swirling angle for the main noise layers
        # This angle will slowly rotate and be influenced by global_energy
        swirl_angle = t * 0.01 + global_energy * math.pi * 0.5

        # Layer 1: Large-scale turbulence, influenced by global energy
        scale1 = 0.5 + global_energy * 2.5 # Increased sensitivity
        octaves1 = 6
        persistence1 = 0.5
        lacunarity1 = 2.0
        layer1 = self._generate_noise_layer(t, scale1, octaves1, persistence1, lacunarity1,
                                            offset_x=self.camera_x, offset_y=self.camera_y,
                                            angle=swirl_angle) # Apply swirl to layer 1

        # Layer 2: Medium-scale plasma flow, influenced by harmonic energy
        # Offset time or coordinates to simulate flow direction based on harmonic energy
        flow_speed = 1.0 + harmonic_energy * 2.0 # Increased flow speed
        flow_offset_x = math.sin(t * 0.5 * flow_speed) * harmonic_energy * 75
        flow_offset_y = math.cos(t * 0.5 * flow_speed) * harmonic_energy * 75

        scale2 = 0.2 + harmonic_energy * 2.0 # Increased sensitivity
        octaves2 = 8
        persistence2 = 0.6
        lacunarity2 = 2.5
        layer2_time = t * flow_speed 
        layer2 = self._generate_noise_layer(layer2_time, scale2, octaves2, persistence2, lacunarity2,
                                            offset_x=self.camera_x * 0.5 + flow_offset_x,
                                            offset_y=self.camera_y * 0.5 + flow_offset_y,
                                            angle=swirl_angle * 1.5) # Apply a slightly faster swirl to layer 2

        # Layer 3: Small-scale flickering hot spots, influenced by percussive impact
        scale3 = 0.1 + percussive_impact * 3.0 # Smaller scale, more reactive
        octaves3 = 4
        persistence3 = 0.3
        lacunarity3 = 3.0
        layer3_time = t * (1.0 + percussive_impact * 4.0) # Speed up with impact
        layer3 = self._generate_noise_layer(layer3_time, scale3, octaves3, persistence3, lacunarity3,
                                            offset_x=self.camera_x * 0.2, offset_y=self.camera_y * 0.2)

        # Layer 4: Plasma streamers, influenced by high_energy
        # This layer will be more directional, emanating from the center, and its intensity will depend on high_energy
        streamer_scale = 0.1 + high_energy * 0.2
        streamer_octaves = 2
        streamer_persistence = 0.5
        streamer_lacunarity = 1.0
        # The angle can change over time to give a swirling effect to the streamers
        streamer_angle = t * 0.05 + high_energy * math.pi * 2 # Rotate slowly and add more rotation with high_energy
        
        layer4 = self._generate_noise_layer(t * 1.5, # Faster time evolution for streamers
                                            streamer_scale,
                                            streamer_octaves,
                                            streamer_persistence,
                                            streamer_lacunarity,
                                            offset_x=self.camera_x * 0.1,
                                            offset_y=self.camera_y * 0.1,
                                            angle=streamer_angle)

        # Combine layers with some weighting
        # Increased influence of smaller layers for more detail and reactivity
        combined_noise = (layer1 * 0.5) + (layer2 * 0.4) + (layer3 * percussive_impact * 0.8) + (layer4 * high_energy * 0.7)
        
        # Add localized hot spots
        if is_beat or is_onset:
            self.hot_spot_center = (np.random.randint(self.width // 4, 3 * self.width // 4),
                                    np.random.randint(self.height // 4, 3 * self.height // 4))
            self.hot_spot_intensity = 1.0 # Max intensity on beat/onset

        hot_spot = self._generate_hot_spot()
        combined_noise = np.clip(combined_noise + hot_spot, 0, 1.5) # Allow some values > 1 for extra brightness
        
        # Normalize to 0-1
        world = (combined_noise - np.min(combined_noise)) / (np.max(combined_noise) - np.min(combined_noise))
        return world

    def _apply_colormap(self, texture, high_energy, spectral_brightness):
        """
        Applies a colormap to the texture, with dynamic color shifts based on audio features.
        """
        colormap_influence = np.clip((high_energy + spectral_brightness) / 2.0, 0, 1)
        
        interpolated_colormap = []
        base_cm = self.base_colormap
        flare_cm = COLORMAPS["solar_flare"]

        # Ensure colormaps have the same number of entries for simple interpolation
        for i in range(len(base_cm)):
            val = base_cm[i][0]
            r1, g1, b1 = base_cm[i][1]
            r2, g2, b2 = flare_cm[i][1] 
            
            r_interp = int(r1 + (r2 - r1) * colormap_influence)
            g_interp = int(g1 + (g2 - g1) * colormap_influence)
            b_interp = int(b1 + (b2 - b1) * colormap_influence)
            
            interpolated_colormap.append((val, (r_interp, g_interp, b_interp)))

        active_colormap = interpolated_colormap

        colored_texture = np.zeros((texture.shape[0], texture.shape[1], 3), dtype=np.uint8)

        for i in range(len(active_colormap) - 1):
            val_start, color_start = active_colormap[i]
            val_end, color_end = active_colormap[i+1]

            mask = (texture >= val_start) & (texture < val_end)
            if np.any(mask): 
                interp = (texture[mask] - val_start) / (val_end - val_start)

                for c in range(3):
                    colored_texture[mask, c] = (1 - interp) * color_start[c] + interp * color_end[c]

        val_last, color_last = active_colormap[-1]
        mask = texture >= val_last
        colored_texture[mask] = color_last
        
        return colored_texture


    def render_manifest(self, manifest, progress_callback=None):
        """
        Render the solar visualization for the given audio manifest.
        """
        n_frames = len(manifest["frames"])
        for i, frame_data in enumerate(manifest["frames"]):
            if progress_callback:
                progress_callback(i, n_frames)

            global_energy = frame_data["global_energy"]
            low_energy = frame_data["low_energy"]
            harmonic_energy = frame_data["harmonic_energy"]
            percussive_impact = frame_data["percussive_impact"]
            high_energy = frame_data["high_energy"]
            is_beat = frame_data["is_beat"]
            is_onset = frame_data["is_onset"]
            spectral_brightness = frame_data["spectral_brightness"]

            # Update camera
            self._update_camera(i, global_energy, high_energy)

            # Use global_energy to control the speed of the animation
            self.time += global_energy * 0.1

            # Update sun parameters
            self.sun_center, self.sun_radius = self._get_sun_params(low_energy, percussive_impact)

            # Generate sun texture using multiple layers and audio features
            texture = self._generate_sun_texture(self.time, global_energy, harmonic_energy, percussive_impact, is_beat, is_onset, high_energy)
            
            # Apply color map with dynamic shifts
            colored_texture_np = self._apply_colormap(texture, high_energy, spectral_brightness)

            # Create a circular mask for the sun body
            sun_mask_image = Image.new("L", (self.width, self.height), 0)
            draw_mask = ImageDraw.Draw(sun_mask_image)
            draw_mask.ellipse(
                (self.sun_center[0] - self.sun_radius, self.sun_center[1] - self.sun_radius,
                 self.sun_center[0] + self.sun_radius, self.sun_center[1] + self.sun_radius),
                fill=255,
            )
            sun_mask_np = np.array(sun_mask_image) / 255.0
            
            # Apply sun mask to the colored texture
            final_frame_np = (colored_texture_np * sun_mask_np[:, :, np.newaxis]).astype(np.uint8)
            
            # Flare generation and rendering
            if percussive_impact > 0.7: # High impact triggers a flare
                # Determine start point on the sun's circumference
                angle_start = random.uniform(0, 2 * math.pi)
                fx_start = int(self.sun_center[0] + self.sun_radius * math.cos(angle_start))
                fy_start = int(self.sun_center[1] + self.sun_radius * math.sin(angle_start))
                
                # Determine end point further out, with a random angle offset from start
                angle_offset = random.uniform(-math.pi / 4, math.pi / 4) # Spread the flares
                angle_end = angle_start + angle_offset
                # Extend flare length based on percussive_impact
                flare_length = self.sun_radius * (1.0 + percussive_impact * 1.5)
                fx_end = int(self.sun_center[0] + flare_length * math.cos(angle_end))
                fy_end = int(self.sun_center[1] + flare_length * math.sin(angle_end))

                # Control point for quadratic Bezier (influences arc height)
                # Offset from midpoint towards a random direction
                mid_x = (fx_start + fx_end) / 2
                mid_y = (fy_start + fy_end) / 2
                
                # Perpendicular direction for control point
                perp_angle = angle_start + math.pi / 2 # Rotate 90 degrees
                
                # Arc height influenced by high_energy
                arc_height = self.sun_radius * (0.2 + high_energy * 0.5)
                fcx = int(mid_x + arc_height * math.cos(perp_angle + random.uniform(-0.5, 0.5)))
                fcy = int(mid_y + arc_height * math.sin(perp_angle + random.uniform(-0.5, 0.5)))

                self.flares.append({
                    "start_point": (fx_start, fy_start),
                    "control_point": (fcx, fcy),
                    "end_point": (fx_end, fy_end),
                    "max_width": max(1, int(percussive_impact * self.sun_radius * (0.1 + high_energy * 0.15))),
                    "current_intensity": min(1.0, 0.5 + percussive_impact * 0.5 + high_energy * 0.3),
                    "color": (255, 200, 0), # Bright yellow/orange for flares
                    "decay_rate": 0.85 + (1 - 0.85) * global_energy * 0.5 # Slower decay for higher global energy
                })
            
            # Render existing flares and update their state
            flare_composite_image = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
            new_flares = []
            for flare in self.flares:
                flare_image_single = self._generate_arcing_flare_image(flare)
                flare_composite_image = Image.alpha_composite(flare_composite_image, flare_image_single)
                
                flare["current_intensity"] *= flare["decay_rate"]
                flare["max_width"] *= 0.95 # Flares also shrink
                if flare["current_intensity"] > 0.01 and flare["max_width"] > 1:
                    new_flares.append(flare)
            self.flares = new_flares

            # Convert main frame to PIL Image for compositing
            main_frame_image = Image.fromarray(final_frame_np, 'RGB').convert('RGBA')

            # Composite flares onto the main frame
            final_image = Image.alpha_composite(main_frame_image, flare_composite_image)
            
            frame = np.array(final_image.convert('RGB'))

            yield frame
