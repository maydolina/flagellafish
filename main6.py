"""
Enhanced Fish & Flagella Physics Simulator
Rewritten to:
 - Fix missing method (switch_propulsion)
 - Remove duplicated/stray code
 - Fix overlapping UI text by computing panel size from rendered text
 - Remove "Key Physics Concept" heading and make the insight panel auto-fit the text
 - Improve alpha panels and help overlay
 - Clean up scaling, presets, and controls

Run: requires pygame (pip install pygame)
"""

import pygame
import math
import sys
import textwrap
from dataclasses import dataclass
from typing import Tuple
from enum import Enum


# --------------------
# Configuration and Constants
# --------------------
@dataclass
class Config:
    WIDTH: int = 1200
    HEIGHT: int = 700
    FPS: int = 60
    FONT_SIZE: int = 18
    VISUAL_SPEED_SCALER: float = 200.0

    COLOR_BACKGROUND: Tuple[int, int, int] = (15, 25, 35)
    COLOR_FIN: Tuple[int, int, int] = (0, 180, 255)
    COLOR_FLAGELLA: Tuple[int, int, int] = (255, 180, 0)
    COLOR_TEXT: Tuple[int, int, int] = (220, 220, 220)
    COLOR_HIGHLIGHT: Tuple[int, int, int] = (255, 255, 100)
    COLOR_TRAIL: Tuple[int, int, int] = (100, 150, 200)
    COLOR_GRID: Tuple[int, int, int] = (40, 50, 60)

    MIN_SIZE: float = 0.000001
    MAX_SIZE: float = 2.0
    MIN_FREQ: float = 0.1
    MAX_FREQ: float = 1000.0
    MIN_VISCOSITY: float = 0.00001
    MAX_VISCOSITY: float = 1.0

    TRAIL_LENGTH: int = 100


class PropulsionType(Enum):
    FIN = "fin"
    FLAGELLA = "flagella"


# --------------------
# Physics functions
# --------------------

def reynolds_number(density: float, velocity: float, length: float, viscosity: float) -> float:
    if viscosity <= 0 or length <= 0:
        return 0.0
    return density * velocity * length / viscosity


def drag_coefficient(reynolds: float, propulsion_type: PropulsionType) -> float:
    if reynolds < 1:
        return 24 / max(reynolds, 0.01)
    elif reynolds < 1000:
        base_cd = 24 / reynolds + 6 / (1 + math.sqrt(reynolds)) + 0.4
        return base_cd * (0.7 if propulsion_type == PropulsionType.FLAGELLA else 1.0)
    else:
        return 0.44 if propulsion_type == PropulsionType.FIN else 0.25


def fin_propulsion_force(length: float, frequency: float, amplitude: float,
                         reynolds: float, density: float) -> float:
    if reynolds < 0.1:
        efficiency = 0.001
    elif reynolds < 1:
        efficiency = 0.01 * reynolds
    elif reynolds < 10:
        efficiency = 0.1 + 0.3 * (reynolds / 10)
    else:
        efficiency = 0.4 + 0.4 / (1 + math.exp(-(reynolds - 100) / 50))

    swept_area = length * amplitude
    force = 0.5 * density * (frequency * amplitude) ** 2 * swept_area * efficiency
    return force


def flagella_propulsion_force(length: float, frequency: float, pitch: float,
                              reynolds: float, density: float, viscosity: float) -> float:
    if reynolds < 0.01:
        efficiency = 3.0
    elif reynolds < 1:
        efficiency = 2.5 - 1.5 * reynolds
    elif reynolds < 10:
        efficiency = 1.0 - 0.8 * (reynolds / 10)
    else:
        efficiency = 0.2 * math.exp(-(reynolds - 10) / 50)

    helical_velocity = frequency * pitch
    force = 6 * math.pi * viscosity * length * helical_velocity * efficiency
    return max(0.0, force)


# --------------------
# Visual helpers
# --------------------
class Trail:
    def __init__(self, max_length: int):
        self.positions = []
        self.max_length = max_length

    def add_position(self, pos: pygame.Vector2):
        self.positions.append(pos.copy())
        if len(self.positions) > self.max_length:
            self.positions.pop(0)

    def draw(self, screen: pygame.Surface, color: Tuple[int, int, int]):
        if len(self.positions) < 2:
            return

        for i in range(1, len(self.positions)):
            alpha = i / len(self.positions)
            fade_color = tuple(int(c * alpha * 0.5) for c in color)
            start_pos = self.positions[i - 1]
            end_pos = self.positions[i]
            if start_pos.distance_to(end_pos) < 50:
                pygame.draw.line(screen, fade_color, start_pos, end_pos, max(1, int(alpha * 3)))


# --------------------
# Creature class
# --------------------
class Creature:
    def __init__(self, x: float, y: float, config: Config):
        self.pos = pygame.Vector2(x, y)
        self.velocity = pygame.Vector2(0.0, 0.0)
        self.size = 0.1
        self.frequency = 5.0
        self.amplitude = 0.02
        self.pitch = 0.001
        self.propulsion_type = PropulsionType.FIN

        self.animation_time = 0.0
        self.config = config
        self.trail = Trail(config.TRAIL_LENGTH)

        self.reynolds_number = 0.0
        self.drag_force = 0.0
        self.propulsion_force = 0.0
        self.efficiency = 0.0

        self.presets = {
            "bacterium": {"size": 0.000005, "freq": 100.0, "propulsion": PropulsionType.FLAGELLA},
            "small_fish": {"size": 0.08, "freq": 3.0, "propulsion": PropulsionType.FIN},
            "large_fish": {"size": 0.5, "freq": 1.5, "propulsion": PropulsionType.FIN},
        }

    def load_preset(self, preset_name: str):
        if preset_name in self.presets:
            p = self.presets[preset_name]
            self.size = p["size"]
            self.frequency = p["freq"]
            self.propulsion_type = p["propulsion"]
            self.velocity = pygame.Vector2(0.0, 0.0)
            self.trail.positions.clear()
            self.pos.x = int(self.config.WIDTH * 0.25)
            self.pos.y = int(self.config.HEIGHT * 0.5)
            self.animation_time = 0.0

    def switch_propulsion(self):
        """Toggle between fin and flagella."""
        self.propulsion_type = (PropulsionType.FLAGELLA
                                if self.propulsion_type == PropulsionType.FIN
                                else PropulsionType.FIN)

    def get_visual_speed_multiplier(self) -> float:
        if self.size < 0.00001:
            return 0.1
        elif self.size < 0.01:
            return 0.5
        else:
            return 1.0

    def get_display_size(self) -> int:
        # Use logarithmic-ish scaling with minimum sizes for visibility
        if self.size < 0.00001:
            # bacteria range
            bacteria_log = math.log10(max(self.size, 1e-9))
            bacteria_normalized = max(0.0, min(1.0, (bacteria_log + 9) / 4))
            display_size = 12 + bacteria_normalized * 12
            return max(8, int(display_size))
        else:
            log_size = math.log10(max(self.size, 1e-9))
            min_log, max_log = -5.0, 1.0
            normalized = max(0.0, min(1.0, (log_size - min_log) / (max_log - min_log)))
            display_size = 24 + normalized * 36
            return max(12, int(display_size))

    def get_propulsion_effectiveness(self) -> str:
        re = self.reynolds_number
        if self.propulsion_type == PropulsionType.FIN:
            if re < 0.1:
                return "TERRIBLE - Fins useless in viscous regime"
            elif re < 1:
                return "VERY POOR - Scallop theorem limits fins"
            elif re < 10:
                return "POOR - Fins starting to work"
            else:
                return "GOOD - Fins work well at high Re"
        else:
            if re < 0.01:
                return "EXCELLENT - Flagella perfect here"
            elif re < 1:
                return "VERY GOOD - Flagella optimal"
            elif re < 10:
                return "FAIR - Flagella losing efficiency"
            else:
                return "POOR - High drag hurts flagella"

    def get_organism_type(self) -> str:
        if self.size < 0.00001:
            return "Bacterium"
        elif self.size < 0.01:
            return "Microorganism"
        elif self.size < 0.1:
            return "Small Fish"
        else:
            return "Large Fish"

    def adjust_size(self, factor: float):
        self.size = max(self.config.MIN_SIZE, min(self.config.MAX_SIZE, self.size * factor))

    def adjust_frequency(self, factor: float):
        self.frequency = max(self.config.MIN_FREQ, min(self.config.MAX_FREQ, self.frequency * factor))

    def update(self, dt: float, density: float, viscosity: float):
        speed = self.velocity.magnitude()
        self.reynolds_number = reynolds_number(density, speed, self.size, viscosity)

        if self.propulsion_type == PropulsionType.FIN:
            self.propulsion_force = fin_propulsion_force(self.size, self.frequency, self.amplitude,
                                                         self.reynolds_number, density)
        else:
            self.propulsion_force = flagella_propulsion_force(self.size, self.frequency, self.pitch,
                                                              self.reynolds_number, density, viscosity)

        drag_coeff = drag_coefficient(self.reynolds_number, self.propulsion_type)
        cross_sectional_area = math.pi * (self.size / 2) ** 2
        self.drag_force = 0.5 * density * speed ** 2 * drag_coeff * cross_sectional_area

        mass = density * (4.0 / 3.0) * math.pi * (self.size / 2) ** 3
        net_force = self.propulsion_force - self.drag_force
        acceleration = net_force / mass if mass > 0 else 0.0

        # Only horizontal motion in this simplified sim
        self.velocity.x += acceleration * dt
        self.velocity.x = max(0.0, self.velocity.x)

        visual_speed_multiplier = self.get_visual_speed_multiplier()
        visual_velocity = self.velocity * visual_speed_multiplier
        self.pos += visual_velocity * dt * self.config.VISUAL_SPEED_SCALER

        if self.pos.x > self.config.WIDTH + 50:
            self.pos.x = -50
            self.trail.positions.clear()

        self.animation_time += dt
        self.trail.add_position(self.pos)

        theoretical_max = self.propulsion_force / max(1e-6, self.size)
        actual_performance = self.velocity.magnitude()
        self.efficiency = min(1.0, actual_performance / max(1e-6, theoretical_max))

    def draw(self, screen: pygame.Surface):
        trail_color = (self.config.COLOR_FIN if self.propulsion_type == PropulsionType.FIN
                       else self.config.COLOR_FLAGELLA)
        self.trail.draw(screen, trail_color)

        display_radius = self.get_display_size()
        body_color = (self.config.COLOR_FIN if self.propulsion_type == PropulsionType.FIN
                      else self.config.COLOR_FLAGELLA)

        if self.propulsion_type == PropulsionType.FIN:
            self._draw_enhanced_fin(screen, display_radius, body_color)
        else:
            self._draw_enhanced_flagella(screen, display_radius, body_color)

        pygame.draw.circle(screen, body_color, (int(self.pos.x), int(self.pos.y)), display_radius)

        efficiency_color = (0, int(255 * self.efficiency), int(255 * (1 - self.efficiency)))
        pygame.draw.circle(screen, efficiency_color, (int(self.pos.x), int(self.pos.y)), display_radius + 2, 2)

    def _draw_enhanced_fin(self, screen: pygame.Surface, radius: int, color: Tuple[int, int, int]):
        segments = max(4, radius // 3)
        tail_length = int(radius * 2.5)
        int_pos = (int(self.pos.x), int(self.pos.y))
        points = [int_pos]
        for i in range(1, segments + 1):
            progress = i / segments
            x_offset = -int(radius * 0.5) - int(progress * tail_length)
            primary_wave = self.amplitude * 1000 * math.sin(self.animation_time * self.frequency * 2 * math.pi)
            secondary_wave = self.amplitude * 300 * math.sin(self.animation_time * self.frequency * 4 * math.pi + progress * 3)
            y_offset = (primary_wave + secondary_wave) * (1 - progress * 0.5) * (radius / 10)
            point_x = int(self.pos.x + x_offset)
            point_y = int(self.pos.y + y_offset)
            points.append((point_x, point_y))

        if len(points) > 2:
            pygame.draw.polygon(screen, color, points)
            pygame.draw.polygon(screen, tuple(min(255, c + 50) for c in color), points, max(1, radius // 8))

    def _draw_enhanced_flagella(self, screen: pygame.Surface, radius: int, color: Tuple[int, int, int]):
        segments = max(10, radius * 2)
        length = radius * 3
        points = []
        for i in range(int(segments) + 1):
            progress = i / segments
            x_offset = -int(radius * 0.8) - int(progress * length)
            rotation_angle = progress * 12 + self.animation_time * self.frequency * 2 * math.pi
            helix_radius = radius * 0.3 * (1 - progress * 0.2)
            y_offset = helix_radius * math.sin(rotation_angle)
            z_offset = helix_radius * math.cos(rotation_angle)
            final_y = y_offset + z_offset * 0.2
            point_x = int(self.pos.x + x_offset)
            point_y = int(self.pos.y + final_y)
            points.append((point_x, point_y))

        if len(points) > 1:
            pygame.draw.lines(screen, color, False, points, max(1, radius // 6))
            if radius > 5:
                highlight_points = [(p[0], p[1] - 1) for p in points]
                pygame.draw.lines(screen, tuple(min(255, c + 80) for c in color), False, highlight_points, max(1, radius // 10))


# --------------------
# Game class
# --------------------
class EnhancedGame:
    def __init__(self):
        pygame.init()
        self.config = Config()
        self.screen = pygame.display.set_mode((self.config.WIDTH, self.config.HEIGHT))
        pygame.display.set_caption("Enhanced Fish & Flagella Physics Simulator")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", self.config.FONT_SIZE, bold=True)
        self.small_font = pygame.font.SysFont("Arial", max(12, self.config.FONT_SIZE - 3))

        self.is_running = True
        self.active_control = None
        self.show_help = False

        self.creature = Creature(self.config.WIDTH // 4, self.config.HEIGHT // 2, self.config)
        self.viscosity = 0.001
        self.density = 1000

        self.environments = {
            "Water": {"viscosity": 0.001, "density": 1000},
            "Honey": {"viscosity": 10.0, "density": 1400},
            "Air": {"viscosity": 0.000018, "density": 1.2},
            "Oil": {"viscosity": 0.1, "density": 900},
            "Cytoplasm": {"viscosity": 0.01, "density": 1100},
        }
        self.current_environment = "Water"

    def run(self):
        while self.is_running:
            dt = self.clock.tick(self.config.FPS) / 1000.0
            self.handle_events()
            self.update(dt)
            self.draw()
        pygame.quit()
        sys.exit()

    def handle_events(self):
        self.active_control = None
        keys = pygame.key.get_pressed()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.is_running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_f:
                    self.creature.switch_propulsion()
                elif event.key == pygame.K_h:
                    self.show_help = not self.show_help
                elif event.key == pygame.K_r:
                    self._reset_creature()
                elif event.key in [pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5]:
                    self._switch_environment(event.key - pygame.K_1)
                elif event.key == pygame.K_q:
                    self.creature.load_preset("bacterium")
                elif event.key == pygame.K_w:
                    self.creature.load_preset("small_fish")
                elif event.key == pygame.K_e:
                    self.creature.load_preset("large_fish")

        # Add size and frequency adjustment from second code
        if keys[pygame.K_UP]:
            self.creature.adjust_size(1.01)
            self.active_control = "Size"
        if keys[pygame.K_DOWN]:
            self.creature.adjust_size(0.99)
            self.active_control = "Size"
        if keys[pygame.K_RIGHT]:
            self.creature.adjust_frequency(1.01)
            self.active_control = "Frequency"
        if keys[pygame.K_LEFT]:
            self.creature.adjust_frequency(0.99)
            self.active_control = "Frequency"
        if keys[pygame.K_v]:
            self.viscosity = min(self.config.MAX_VISCOSITY, self.viscosity * 1.01)
            self.active_control = "Viscosity"
        if keys[pygame.K_b]:
            self.viscosity = max(self.config.MIN_VISCOSITY, self.viscosity / 1.01)
            self.active_control = "Viscosity"

    def _reset_creature(self):
        self.creature.pos.x = self.config.WIDTH // 4
        self.creature.velocity = pygame.Vector2(0.0, 0.0)
        self.creature.size = 0.1
        self.creature.frequency = 5.0
        self.creature.trail.positions.clear()

    def _switch_environment(self, env_index: int):
        env_names = list(self.environments.keys())
        if 0 <= env_index < len(env_names):
            self.current_environment = env_names[env_index]
            env_data = self.environments[self.current_environment]
            self.viscosity = env_data["viscosity"]
            self.density = env_data["density"]

    def update(self, dt: float):
        self.creature.update(dt, self.density, self.viscosity)

    def draw(self):
        self.screen.fill(self.config.COLOR_BACKGROUND)
        self._draw_background_grid()
        self.creature.draw(self.screen)
        self._draw_enhanced_ui()
        if self.show_help:
            self._draw_help()
        pygame.display.flip()

    def _draw_background_grid(self):
        grid_spacing = 50
        for x in range(0, self.config.WIDTH, grid_spacing):
            pygame.draw.line(self.screen, self.config.COLOR_GRID, (x, 0), (x, self.config.HEIGHT), 1)
        for y in range(0, self.config.HEIGHT, grid_spacing):
            pygame.draw.line(self.screen, self.config.COLOR_GRID, (0, y), (self.config.WIDTH, y), 1)

    def _draw_enhanced_ui(self):
        regime = "Inertial" if self.creature.reynolds_number > 10 else "Viscous"
        organism_type = self.creature.get_organism_type()
        effectiveness = self.creature.get_propulsion_effectiveness()

        info_data = [
            ("Organism", organism_type),
            ("Environment", f"{self.current_environment} (1-5)"),
            ("Propulsion", f"{self.creature.propulsion_type.value.capitalize()} (F)"),
            ("Effectiveness", effectiveness),
            ("Size", f"{self.creature.size:.1e} m (↑/↓)"),
            ("Frequency", f"{self.creature.frequency:.2f} Hz (←/→)"),
            ("Viscosity", f"{self.viscosity:.1e} Pa·s (V/B)"),
            ("Reynolds", f"{self.creature.reynolds_number:.2f} ({regime})"),
            ("Speed", f"{self.creature.velocity.magnitude():.2e} m/s"),
            ("Efficiency", f"{self.creature.efficiency * 100:.1f}%"),
        ]

        panel_x = self.config.WIDTH - 320
        self._draw_info_panel(info_data, panel_x, 15, width=300)

        # Insights panel (auto-fit, no title)
        insights = self._get_physics_insights()
        insight_x = self.config.WIDTH - 320
        insight_y = self.config.HEIGHT - 120
        # Wrap the insight text to a readable width and draw a small auto-sized panel
        wrapped = textwrap.fill(insights, width=36)
        self._draw_info_panel([(None, wrapped)], insight_x, insight_y, width=None)

        preset_text = "Presets: Q=Bacterium, W=Small Fish, E=Large Fish"
        preset_surface = self.small_font.render(preset_text, True, self.config.COLOR_TEXT)
        self.screen.blit(preset_surface, (15, self.config.HEIGHT - 50))

        controls_text = "Press H for help, R to reset, F to switch propulsion"
        text_surface = self.small_font.render(controls_text, True, self.config.COLOR_TEXT)
        self.screen.blit(text_surface, (15, self.config.HEIGHT - 30))

    def _draw_info_panel(self, data: list, x: int, y: int, width: int = None):
        # Data: list of (key, value). If key is None, value is treated as multi-line body text.
        padding = 12
        line_height = self.config.FONT_SIZE + 4

        # Prepare surfaces and determine required width
        key_surfaces = []
        value_surfaces = []
        max_key_w = 0
        max_value_w = 0
        total_lines = 0

        for key, value in data:
            if key is None:
                # multi-line value
                lines = str(value).split('\n')
                vs = [self.small_font.render(line, True, self.config.COLOR_TEXT) for line in lines]
                key_surfaces.append(None)
                value_surfaces.append(vs)
                max_value_w = max(max_value_w, max((s.get_width() for s in vs), default=0))
                total_lines += len(vs) + 1
            else:
                ks = self.font.render(f"{key}:", True, self.config.COLOR_HIGHLIGHT if key == self.active_control else self.config.COLOR_TEXT)
                vs = [self.font.render(str(value), True, self.config.COLOR_TEXT)]
                key_surfaces.append(ks)
                value_surfaces.append(vs)
                max_key_w = max(max_key_w, ks.get_width())
                max_value_w = max(max_value_w, vs[0].get_width())
                total_lines += 1

        # If width not provided, compute a snug width
        if width is None:
            width = max_key_w + max_value_w + padding * 3
            width = max(width, 160)

        panel_height = total_lines * (self.small_font.get_height() + 2) + padding
        panel_surface = pygame.Surface((width, panel_height), pygame.SRCALPHA)
        panel_surface.fill((10, 15, 25, 210))

        # Draw border on the screen directly (so alpha blends correctly)
        screen_rect = pygame.Rect(x - 5, y - 5, width + 10, panel_height + 10)
        border_surf = pygame.Surface((screen_rect.width, screen_rect.height), pygame.SRCALPHA)
        border_surf.fill((0, 0, 0, 0))
        pygame.draw.rect(border_surf, (255, 255, 255, 20), border_surf.get_rect(), 1)
        self.screen.blit(border_surf, (screen_rect.x, screen_rect.y))

        # Blit the panel itself
        self.screen.blit(panel_surface, (x, y))

        # Render text
        cur_y = y + padding // 2
        for ks, vs in zip(key_surfaces, value_surfaces):
            if ks is None:
                for surf in vs:
                    self.screen.blit(surf, (x + padding, cur_y))
                    cur_y += surf.get_height() + 2
                cur_y += 4
            else:
                # Key at left, value at right side of panel
                self.screen.blit(ks, (x + padding, cur_y))
                # right align value within panel
                val_surf = vs[0]
                val_x = x + width - padding - val_surf.get_width()
                self.screen.blit(val_surf, (val_x, cur_y))
                cur_y += max(ks.get_height(), val_surf.get_height()) + 6

    def _get_physics_insights(self) -> str:
        re = self.creature.reynolds_number
        organism = self.creature.get_organism_type()
        prop_type = self.creature.propulsion_type.value

        if organism == "Bacterium":
            if prop_type == "flagella":
                return ("Bacteria evolved flagella because they work well in viscous fluids.\n"
                        "Try switching to fins to see why reciprocal strokes fail at low Re.")
            else:
                return ("Fins are ineffective at bacterial scales due to the scallop theorem.\n"
                        "Reciprocal motion produces almost no net displacement in viscous regimes.")
        else:
            if prop_type == "fin":
                return (f"{organism} with fins: good choice. At higher Re, inertia helps fins produce thrust.\n"
                        "Flagella would generate excessive drag at this scale.")
            else:
                return (f"{organism} with flagella: inefficient at this scale.\n"
                        "Flagella create high drag when inertial effects dominate.")

    def _draw_help(self):
        # Semi-transparent overlay
        overlay = pygame.Surface((self.config.WIDTH // 2, self.config.HEIGHT // 2), pygame.SRCALPHA)
        overlay.fill((5, 10, 15, 230))
        x = self.config.WIDTH // 4
        y = self.config.HEIGHT // 4
        self.screen.blit(overlay, (x, y))
        pygame.draw.rect(self.screen, self.config.COLOR_HIGHLIGHT, (x, y, overlay.get_width(), overlay.get_height()), 3)

        help_text = [
            "BIOLOGICAL LOCOMOTION SIMULATOR",
            "",
            "ORGANISM PRESETS:",
            "Q - Bacterium (µm scale, flagella, high freq)",
            "W - Small Fish (cm scale, fins, low freq)",
            "E - Large Fish (dm scale, fins, very low freq)",
            "",
            "CONTROLS:",
            "F - Switch propulsion type",
            "↑/↓ - Adjust creature size",
            "←/→ - Adjust movement frequency",
            "V/B - Adjust fluid viscosity",
            "1-5 - Switch environments",
            "R - Reset creature",
        ]

        for i, line in enumerate(help_text):
            color = self.config.COLOR_HIGHLIGHT if line.endswith(":") or i == 0 else self.config.COLOR_TEXT
            surf = self.font.render(line, True, color)
            self.screen.blit(surf, (x + 20, y + 20 + i * (self.config.FONT_SIZE + 2)))


# --------------------
# Main
# --------------------
if __name__ == "__main__":
    game = EnhancedGame()
    game.run()