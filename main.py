import pygame
import math
import sys

# --------------------
# Physics Functions
# --------------------

def reynolds_number(rho, v, L, mu):
    """Calculate Reynolds number."""
    return rho * v * L / mu

def fin_velocity(L, freq, amp, regime):
    """Calculate forward velocity from fin motion."""
    if regime == "high":  # inertia-dominated
        return amp * freq * L * 0.8
    else:  # low Reynolds, Purcell’s scallop theorem hurts fins
        return amp * freq * L * 0.05  # massively reduced efficiency

def flagella_velocity(L, freq, pitch, regime):
    """Calculate forward velocity from rotating corkscrew."""
    if regime == "high":  # not great for big animals, lots of drag
        return freq * pitch * 0.3
    else:  # perfect for bacteria
        return freq * pitch * 2.0

# --------------------
# Pygame Setup
# --------------------

pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Fish with Flagella Simulator")
clock = pygame.time.Clock()

# --------------------
# Simulation Parameters
# --------------------
size = 1.0           # meters for fish, microns for bacteria
viscosity = 0.001    # Pa·s (water)
rho = 1000           # kg/m³ (density of water)
propulsion = "fin"   # "fin" or "flagella"
freq = 2.0           # Hz (movement frequency)
amp = 0.1            # meters (fin amplitude)
pitch = 0.002        # meters per turn (flagella pitch)
position = 50
speed = 0

# --------------------
# Main Loop
# --------------------
running = True
time_elapsed = 0

while running:
    screen.fill((30, 30, 40))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        # Keyboard controls
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_f:
                propulsion = "fin"
            if event.key == pygame.K_l:
                propulsion = "flagella"
            if event.key == pygame.K_UP:
                size *= 1.2
            if event.key == pygame.K_DOWN:
                size /= 1.2
            if event.key == pygame.K_RIGHT:
                freq *= 1.1
            if event.key == pygame.K_LEFT:
                freq /= 1.1
            if event.key == pygame.K_v:
                viscosity *= 2
            if event.key == pygame.K_b:
                viscosity /= 2

    # Determine physics regime
    Re_test = reynolds_number(rho, 0.1, size, viscosity)
    regime = "high" if Re_test > 10 else "low"

    # Calculate velocity
    if propulsion == "fin":
        speed = fin_velocity(size, freq, amp, regime)
    else:
        speed = flagella_velocity(size, freq, pitch, regime)

    # Update position
    position += speed * 0.1  # scaled down for display
    if position > WIDTH:
        position = 50
        time_elapsed = 0

    # Draw creature
    color = (0, 200, 255) if propulsion == "fin" else (255, 200, 0)
    pygame.draw.circle(screen, color, (int(position), HEIGHT // 2), int(size * 10))

    # Draw text info
    font = pygame.font.SysFont("Arial", 20)
    text_lines = [
        f"Propulsion: {propulsion} (press F or L to switch)",
        f"Size: {size:.4f} m (UP/DOWN to change)",
        f"Frequency: {freq:.2f} Hz (LEFT/RIGHT to change)",
        f"Viscosity: {viscosity} Pa·s (V/B to change)",
        f"Reynolds number: {Re_test:.2f} ({regime} regime)",
        f"Speed: {speed:.4f} m/s"
    ]
    for i, line in enumerate(text_lines):
        text_surface = font.render(line, True, (255, 255, 255))
        screen.blit(text_surface, (10, 10 + i * 22))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()
