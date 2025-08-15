import pygame
import math
import sys

# --------------------
# Constants
# --------------------
# Screen and Display
WIDTH, HEIGHT = 1000, 600
FPS = 60
FONT_SIZE = 22
VISUAL_SPEED_SCALER = 500.0
# Colors
COLOR_BACKGROUND = (20, 30, 40)
COLOR_FIN = (0, 200, 255)
COLOR_FLAGELLA = (255, 200, 0)
COLOR_TEXT = (230, 230, 230)
COLOR_HIGHLIGHT = (255, 255, 0)


# --------------------
# Physics Functions
# --------------------

def reynolds_number(rho, v, L, mu):
    """Calculate the Reynolds number."""
    if mu == 0: return float('inf')
    return rho * v * L / mu


def fin_velocity(L, freq, amp, regime):
    """Calculate forward velocity from fin motion."""
    # Purcell's scallop theorem states reciprocal motion (like a simple hinge)
    # at low Reynolds numbers results in no net movement. Fins are more complex,
    # but their efficiency is still drastically reduced.
    efficiency = 0.8 if regime == "high" else 0.05
    return amp * freq * L * efficiency


def flagella_velocity(L, freq, pitch, regime):
    """Calculate forward velocity from a rotating corkscrew."""
    # Flagella are incredibly efficient at low Reynolds numbers but less so in
    # high-Re regimes where turbulence and drag dominate. The size (L) is
    # not directly used as velocity is mainly from pitch*frequency.
    efficiency = 0.3 if regime == "high" else 2.0
    return freq * pitch * efficiency


# --------------------
# Game Object Classes
# --------------------

class Creature:
    """Represents the organism being simulated."""

    def __init__(self, x, y):
        self.pos = pygame.Vector2(x, y)
        self.size = 1.0  # meters for fish, microns for bacteria
        self.freq = 2.0  # Hz (movement frequency)
        self.amp = 0.1  # meters (fin amplitude)
        self.pitch = 0.002  # meters per turn (flagella pitch)
        self.propulsion = "fin"
        self.speed = 0.0
        self.animation_time = 0.0

    def switch_propulsion(self):
        """Toggle between fin and flagella."""
        self.propulsion = "flagella" if self.propulsion == "fin" else "fin"

    def update(self, dt, rho, viscosity):
        """Update creature's state and position."""
        # 1. Calculate Reynolds number based on CURRENT speed
        re_val = reynolds_number(rho, self.speed, self.size, viscosity)
        regime = "high" if re_val > 10 else "low"

        # 2. Calculate speed based on propulsion and physics regime
        if self.propulsion == "fin":
            self.speed = fin_velocity(self.size, self.freq, self.amp, regime)
        else:  # flagella
            self.speed = flagella_velocity(self.size, self.freq, self.pitch, regime)

        # 3. Update position
        self.pos.x += self.speed * dt * VISUAL_SPEED_SCALER
        if self.pos.x > WIDTH + 50:
            self.pos.x = -50

        # 4. Update animation timer
        self.animation_time += dt

    def draw(self, screen):
        """Draw the creature and its propulsion system."""
        body_radius = max(5, int(self.size * 10))
        body_color = COLOR_FIN if self.propulsion == "fin" else COLOR_FLAGELLA

        if self.propulsion == "fin":
            self.draw_fin(screen, body_radius, body_color)
        else:  # flagella
            self.draw_flagella(screen, body_radius, body_color)

        # Draw main body on top
        pygame.draw.circle(screen, body_color, (int(self.pos.x), int(self.pos.y)), body_radius)

    def draw_fin(self, screen, radius, color):
        """Draw an animated, oscillating fin."""
        num_segments = 5
        tail_length = radius * 2.5
        points = [self.pos]
        for i in range(1, num_segments + 1):
            progress = i / num_segments
            x_offset = -radius * 0.8 - progress * tail_length
            # Sine wave based on time and position along the tail
            y_offset = self.amp * 200 * (1 - progress) * math.sin(self.animation_time * self.freq * 5 + progress * 4)
            points.append(self.pos + pygame.Vector2(x_offset, y_offset))
        pygame.draw.polygon(screen, color, points, width=0)

    def draw_flagella(self, screen, radius, color):
        """Draw an animated, rotating flagellum."""
        num_segments = 20
        length = radius * 3
        width = radius * 0.6
        points = []
        for i in range(num_segments + 1):
            progress = i / num_segments
            x_offset = -radius - progress * length
            # Corkscrew motion
            y_offset = width * math.sin(progress * 10 + self.animation_time * self.freq * 10)
            points.append(self.pos + pygame.Vector2(x_offset, y_offset))
        if len(points) > 1:
            pygame.draw.lines(screen, color, False, points, width=max(2, int(radius * 0.2)))


class Game:
    """Manages the main game loop, state, and rendering."""

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Advanced Fish & Flagella Simulator")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", FONT_SIZE)
        self.is_running = True
        self.active_control = None  # For highlighting controlled parameter

        # Simulation state
        self.creature = Creature(WIDTH / 4, HEIGHT / 2)
        self.viscosity = 0.001  # Pa·s (water)
        self.rho = 1000  # kg/m³ (density of water)

    def run(self):
        """The main game loop."""
        while self.is_running:
            dt = self.clock.tick(FPS) / 1000.0  # Delta time in seconds
            self.handle_events()
            self.update(dt)
            self.draw()
        pygame.quit()
        sys.exit()

    def handle_events(self):
        """Process user input."""
        self.active_control = None
        keys = pygame.key.get_pressed()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.is_running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_f or event.key == pygame.K_l:
                    self.creature.switch_propulsion()

        # Handle continuous key presses for smooth adjustments
        if keys[pygame.K_UP]:
            self.creature.size *= 1.01
            self.active_control = "Size"
        if keys[pygame.K_DOWN]:
            self.creature.size /= 1.01
            self.active_control = "Size"
        if keys[pygame.K_RIGHT]:
            self.creature.freq *= 1.01
            self.active_control = "Frequency"
        if keys[pygame.K_LEFT]:
            self.creature.freq /= 1.01
            self.active_control = "Frequency"
        if keys[pygame.K_v]:
            self.viscosity *= 1.02
            self.active_control = "Viscosity"
        if keys[pygame.K_b]:
            self.viscosity /= 1.02
            self.active_control = "Viscosity"

    def update(self, dt):
        """Update all game objects."""
        self.creature.update(dt, self.rho, self.viscosity)

    def draw(self):
        """Render everything to the screen."""
        self.screen.fill(COLOR_BACKGROUND)
        self.creature.draw(self.screen)
        self.draw_info_panel()
        pygame.display.flip()

    def draw_info_panel(self):
        """Display simulation parameters and controls."""
        re_val = reynolds_number(self.rho, self.creature.speed, self.creature.size, self.viscosity)
        regime = "High (Inertia)" if re_val > 10 else "Low (Viscosity)"

        info = {
            "Propulsion": f"{self.creature.propulsion.capitalize()} (F/L to switch)",
            "Size": f"{self.creature.size:.3f} m (UP/DOWN)",
            "Frequency": f"{self.creature.freq:.2f} Hz (LEFT/RIGHT)",
            "Viscosity": f"{self.viscosity:.5f} Pa·s (V/B)",
            "---": "---",
            "Reynolds number": f"{re_val:.2f} ({regime})",
            "Speed": f"{self.creature.speed:.4f} m/s",
        }

        for i, (key, value) in enumerate(info.items()):
            if key == "---": continue  # Skip separator

            # Determine text color (highlight if active)
            text_color = COLOR_HIGHLIGHT if key == self.active_control else COLOR_TEXT

            # Create and render key text
            key_surface = self.font.render(f"{key}:", True, text_color)
            self.screen.blit(key_surface, (15, 15 + i * (FONT_SIZE + 4)))

            # Create and render value text
            value_surface = self.font.render(value, True, text_color)
            self.screen.blit(value_surface, (200, 15 + i * (FONT_SIZE + 4)))


# --------------------
# Main Execution
# --------------------
if __name__ == "__main__":
    game = Game()
    game.run()