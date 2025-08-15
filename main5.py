import pygame
import math
import sys

pygame.init()

# Screen
WIDTH, HEIGHT = 900, 700
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Flagella vs Fins - Vertical Race")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREY = (40, 40, 40)
RED = (200, 50, 50)
BLUE = (50, 100, 200)

FONT = pygame.font.SysFont(None, 20)

# Physics constants
ENVIRONMENTS = {
    "Water": 0.001,
    "Air": 0.0000181,
    "Honey": 10,
    "Oil": 0.1,
    "Cytoplasm": 0.002
}

class PropulsionType:
    FLAGELLA = "Flagella"
    FIN = "Fins"

class Creature:
    def __init__(self, propulsion, start_pos, size, frequency, viscosity):
        self.propulsion = propulsion
        self.size = size
        self.frequency = frequency
        self.viscosity = viscosity
        self.pos = pygame.Vector2(start_pos)
        self.speed = 0.0

    def drag_force(self):
        # Quadratic + linear drag (simplified)
        area = math.pi * (self.size/2)**2
        linear_drag = 6 * math.pi * self.viscosity * self.size * self.speed
        quad_drag = 0.5 * 1.0 * area * (self.speed**2)
        return linear_drag + quad_drag

    def thrust(self):
        # Very simple thrust model:
        if self.propulsion == PropulsionType.FLAGELLA:
            # Works best at low viscosity, small size
            return self.frequency * 0.0005 / (self.size**0.5)
        elif self.propulsion == PropulsionType.FIN:
            # Works best at larger size
            return self.frequency * (self.size**1.5) * 0.0000002
        return 0

    def update(self, dt):
        force = self.thrust() - self.drag_force()
        mass = self.size**3
        acc = force / mass
        self.speed += acc * dt
        self.pos.y -= self.speed * dt * 50  # move upward

    def draw(self, surface, color):
        pygame.draw.circle(surface, color, (int(self.pos.x), int(self.pos.y)), int(self.size))
        self.draw_info(surface)

    def draw_info(self, surface):
        # info box to the left of the creature
        info_lines = [
            f"{self.propulsion}",
            f"Speed: {self.speed:.4f}",
            f"Drag: {self.drag_force():.4f}",
            f"Size: {self.size:.1f}",
            f"Freq: {self.frequency:.1f}",
            f"Visc: {self.viscosity:.6f}"
        ]
        max_width = max(FONT.size(line)[0] for line in info_lines) + 8
        box_height = len(info_lines) * 18 + 6
        box_x = self.pos.x - self.size - max_width - 10
        box_y = self.pos.y - box_height//2

        pygame.draw.rect(surface, GREY, (box_x, box_y, max_width, box_height))
        y = box_y + 3
        for line in info_lines:
            text = FONT.render(line, True, WHITE)
            surface.blit(text, (box_x + 4, y))
            y += 18

def create_bacteria_race():
    top = Creature(PropulsionType.FLAGELLA, (WIDTH//2, HEIGHT//4), 5, 10, current_viscosity)
    bottom = Creature(PropulsionType.FIN, (WIDTH//2, HEIGHT*3//4), 5, 10, current_viscosity)
    return top, bottom

def create_fish_race():
    top = Creature(PropulsionType.FIN, (WIDTH//2, HEIGHT//4), 50, 2, current_viscosity)
    bottom = Creature(PropulsionType.FLAGELLA, (WIDTH//2, HEIGHT*3//4), 50, 2, current_viscosity)
    return top, bottom

# Initial settings
current_env = "Water"
current_viscosity = ENVIRONMENTS[current_env]
top_creature, bottom_creature = create_bacteria_race()

clock = pygame.time.Clock()
running = True
mode = "Bacteria"

while running:
    dt = clock.tick(60) / 1000.0

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                mode = "Bacteria"
                top_creature, bottom_creature = create_bacteria_race()
            elif event.key == pygame.K_w:
                mode = "Fish"
                top_creature, bottom_creature = create_fish_race()
            elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5):
                idx = event.key - pygame.K_1
                env_name = list(ENVIRONMENTS.keys())[idx]
                current_env = env_name
                current_viscosity = ENVIRONMENTS[env_name]
                top_creature.viscosity = current_viscosity
                bottom_creature.viscosity = current_viscosity
            elif event.key == pygame.K_UP:
                top_creature.size += 1
                bottom_creature.size += 1
            elif event.key == pygame.K_DOWN:
                top_creature.size = max(1, top_creature.size - 1)
                bottom_creature.size = max(1, bottom_creature.size - 1)
            elif event.key == pygame.K_RIGHT:
                top_creature.frequency += 0.5
                bottom_creature.frequency += 0.5
            elif event.key == pygame.K_LEFT:
                top_creature.frequency = max(0.5, top_creature.frequency - 0.5)
                bottom_creature.frequency = max(0.5, bottom_creature.frequency - 0.5)

    # Update creatures
    top_creature.update(dt)
    bottom_creature.update(dt)

    # Draw
    screen.fill(WHITE)
    pygame.draw.line(screen, BLACK, (0, HEIGHT//2), (WIDTH, HEIGHT//2), 2)

    top_creature.draw(screen, BLUE)
    bottom_creature.draw(screen, RED)

    # Environment display
    env_text = FONT.render(f"Environment: {current_env}", True, BLACK)
    screen.blit(env_text, (10, 10))

    pygame.display.flip()

pygame.quit()
sys.exit()
