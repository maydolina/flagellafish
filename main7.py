"""
Minimal Locomotion Demo – Bacteria vs Fish (Vertical Layout)

Goal: clearly demonstrate that
• at very low Reynolds number (bacteria scale) fins are ineffective, flagella work.
• at higher Reynolds number (fish scale) fins work, flagella are inefficient due to drag.

This is a **teaching-focused** toy model: the equations are simple but chosen so the
behaviour is unambiguous and robust. The UI is minimal with vertical creature layout.

Run:  pip install pygame
Keys:  1 = Bacteria demo   2 = Fish demo   F = toggle propulsion labels on bodies
       Q/A = change top creature frequency   W/S = change top creature size
       E/D = change bottom creature frequency   R/F = change bottom creature size
       Z/X/C/V/B = environment presets (Water/Honey/Air/Oil/Cytoplasm)
       H = help   ESC / close window = quit
"""

import pygame
import math
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Tuple, List, Dict


# --------------------
# Config & constants
# --------------------
@dataclass
class Config:
    WIDTH: int = 1200
    HEIGHT: int = 700
    FPS: int = 60
    FONT_SIZE: int = 18

    BG: Tuple[int, int, int] = (15, 25, 35)
    GRID: Tuple[int, int, int] = (38, 48, 60)
    TEXT: Tuple[int, int, int] = (225, 230, 235)
    HI: Tuple[int, int, int] = (255, 255, 120)

    FIN: Tuple[int, int, int] = (0, 180, 255)
    FLAG: Tuple[int, int, int] = (255, 180, 0)

    # Visual scaling so small things are still visible
    SPEED_VISUAL: float = 250.0


class Propulsion(Enum):
    FIN = "Fin"
    FLAGELLA = "Flagella"


# Environment presets
ENVIRONMENTS = {
    "Water": {"rho": 1000.0, "mu": 0.001, "color": (100, 150, 255)},
    "Honey": {"rho": 1400.0, "mu": 10.0, "color": (255, 200, 50)},
    "Air": {"rho": 1.2, "mu": 0.000018, "color": (200, 220, 255)},
    "Oil": {"rho": 900.0, "mu": 0.1, "color": (120, 80, 40)},
    "Cytoplasm": {"rho": 1100.0, "mu": 0.01, "color": (150, 255, 150)}
}


# --------------------
# Physics (simple but didactic)
# --------------------

def reynolds(rho: float, U: float, L: float, mu: float) -> float:
    if mu <= 0 or L <= 0:
        return 0.0
    return rho * U * L / mu


def cd_blend(Re: float) -> float:
    """Blend low-Re (Stokes) and quadratic drag. Returns a nominal Cd for the quadratic term.
    (We also include a Stokes linear term separately.)"""
    if Re < 1:
        return 24 / max(Re, 1e-2)
    elif Re < 1e3:
        return 24/Re + 6/(1+math.sqrt(Re)) + 0.4
    else:
        return 0.44


def fin_eff(Re: float) -> float:
    """Fins ineffective at low Re, effective at high Re."""
    # Map log10(Re) from [-3, 3] -> [~0, 1]
    x = math.log10(max(Re, 1e-6))
    # Smoothstep from ~0 near Re=1e-2 to ~1 by Re~100
    t = 1/(1 + math.exp(-(x-0.5)*2.5))
    return max(0.02, min(1.0, t))


def flag_eff(Re: float) -> float:
    """Flagella great at low Re, fade at high Re."""
    x = math.log10(max(Re, 1e-6))
    # High near Re<=0.1, drops by Re~10
    t = 1/(1 + math.exp((x-0.0)*2.5))
    return max(0.05, min(1.2, t*1.2))  # a bit >1 to emphasize advantage at low Re


# --------------------
# Creature
# --------------------
class Creature:
    def __init__(self, label: str, x: float, y: float, color: Tuple[int,int,int],
                 L: float, freq: float, propulsion: Propulsion):
        self.label = label
        self.start_x = x
        self.pos = pygame.Vector2(x, y)
        self.vel = pygame.Vector2(0.0, 0.0)
        self.color = color
        self.L = L           # characteristic length (m)
        self.freq = freq     # Hz
        self.amp = max(1e-6, 0.2 * L)  # stroke amplitude (m)
        self.pitch = max(1e-6, 0.2 * L) # flagellar pitch proxy (m/rev)
        self.propulsion = propulsion
        self.anim_t = 0.0
        self.Re = 0.0
        self.speed = 0.0
        self.eff = 0.0

    def radius_px(self) -> int:
        # Log-like mapping to keep both bacteria and fish visible
        if self.L < 1e-4:
            return 12
        if self.L < 1e-2:
            return 20
        if self.L < 1e-1:
            return 32
        if self.L < 5e-1:
            return 45
        return 58

    def update(self, dt: float, rho: float, mu: float):
        U = self.vel.x
        Re = reynolds(rho, abs(U), self.L, mu)

        # Thrust models
        if self.propulsion == Propulsion.FIN:
            eff = fin_eff(Re)
            thrust = 0.5 * rho * (self.freq * self.amp)**2 * self.L**2 * eff
        else:
            eff = flag_eff(Re)
            thrust = 6 * math.pi * mu * (self.L/2) * (self.freq * self.pitch) * eff

        # Drag: linear (Stokes) + quadratic
        r = self.L/2
        A = math.pi * r*r
        Cd = cd_blend(max(Re, 1e-6))
        drag_linear = 6 * math.pi * mu * r * abs(U)
        drag_quad = 0.5 * rho * Cd * A * U*U
        drag = drag_linear + drag_quad
        drag *= 1.0  # overall scale

        # Net
        mass = max(1e-9, rho * (4/3) * math.pi * r**3)
        # Direction: thrust always forward; drag opposes motion
        net = thrust - drag
        ax = net / mass
        # Integrate
        self.vel.x += ax * dt
        # Keep non-negative; this is a forward-swimming demo
        self.vel.x = max(0.0, min(self.vel.x, 3.0))
        self.pos.x += self.vel.x * dt * Config.SPEED_VISUAL

        self.Re = reynolds(rho, abs(self.vel.x), self.L, mu)
        self.speed = self.vel.x
        self.eff = eff

        # Wrap
        if self.pos.x > Config.WIDTH + 60:
            self.pos.x = self.start_x

        self.anim_t += dt

    def draw(self, screen: pygame.Surface, show_labels: bool = True):
        rpx = self.radius_px()
        pos = (int(self.pos.x), int(self.pos.y))
        pygame.draw.circle(screen, self.color, pos, rpx)

        # Simple propulsor hint
        if self.propulsion == Propulsion.FIN:
            # a small wavy tail triangle
            tail = [
                (pos[0]-rpx, pos[1]),
                (pos[0]-rpx-2*rpx, pos[1]-int(rpx*0.6*math.sin(self.anim_t*6))),
                (pos[0]-rpx-2*rpx, pos[1]+int(rpx*0.6*math.sin(self.anim_t*6)))
            ]
            pygame.draw.polygon(screen, self.color, tail, 2)
        else:
            # helix polyline
            pts: List[Tuple[int,int]] = []
            length = 3*rpx
            segs = 24
            for i in range(segs+1):
                p = i/segs
                x = pos[0]-rpx-int(p*length)
                y = pos[1]+int(0.35*rpx*math.sin(10*p + self.anim_t*12))
                pts.append((x,y))
            pygame.draw.lines(screen, self.color, False, pts, 2)

        if show_labels:
            lbl = f"{self.label} — {self.propulsion.value}"
            font = pygame.font.SysFont("Arial", 16, bold=True)
            surf = font.render(lbl, True, self.color)
            screen.blit(surf, (pos[0]-surf.get_width()//2, pos[1]-rpx-25))


# --------------------
# UI helpers
# --------------------
class UI:
    def __init__(self, font_main: pygame.font.Font, font_small: pygame.font.Font):
        self.font = font_main
        self.small = font_small

    def draw_grid(self, screen: pygame.Surface):
        step = 50
        for x in range(0, Config.WIDTH, step):
            pygame.draw.line(screen, Config.GRID, (x, 0), (x, Config.HEIGHT), 1)
        for y in range(0, Config.HEIGHT, step):
            pygame.draw.line(screen, Config.GRID, (0, y), (Config.WIDTH, y), 1)

    def panel(self, screen: pygame.Surface, lines: List[str], x: int, y: int, bg_color: Tuple[int,int,int] = (10, 15, 25)):
        """Auto-size panel that fits its text; no titles; no overlap."""
        pads = 12
        spacing = 3
        # Render lines first to know sizes
        line_surfs = [self.font.render(line, True, Config.TEXT) for line in lines if line.strip()]
        if not line_surfs:
            return

        w = max((s.get_width() for s in line_surfs), default=0) + pads*2
        h = sum(s.get_height() for s in line_surfs) + spacing*(len(line_surfs)-1) + pads*2

        surf = pygame.Surface((w, h), pygame.SRCALPHA)
        # Ensure bg_color is a 3-tuple, then add alpha
        if len(bg_color) >= 3:
            fill_color = (bg_color[0], bg_color[1], bg_color[2], 210)
        else:
            fill_color = (10, 15, 25, 210)
        surf.fill(fill_color)
        pygame.draw.rect(surf, (255,255,255,30), surf.get_rect(), 1)
        screen.blit(surf, (x, y))

        cy = y + pads
        for s in line_surfs:
            screen.blit(s, (x + pads, cy))
            cy += s.get_height() + spacing

    def help_overlay(self, screen: pygame.Surface):
        text = [
            "LOCOMOTION DEMO - VERTICAL LAYOUT",
            "",
            "Scenarios: 1=Bacteria  2=Fish",
            "",
            "Both Creatures Controls:",
            "Q/A: Frequency   W/S: Size",
            "",
            "Environments:",
            "Z=Water  X=Honey  C=Air  V=Oil  B=Cytoplasm",
            "",
            "Other: F=toggle labels  H=close help"
        ]
        pads = 20
        lines = [self.font.render(t, True, Config.TEXT) for t in text]
        w = max(s.get_width() for s in lines) + pads*2
        h = sum(s.get_height() for s in lines) + pads*2 + 6*(len(lines)-1)
        x = (Config.WIDTH - w)//2
        y = (Config.HEIGHT - h)//2
        box = pygame.Surface((w, h), pygame.SRCALPHA)
        box.fill((5,10,15,240))
        pygame.draw.rect(box, Config.HI, box.get_rect(), 3)
        screen.blit(box, (x,y))
        cy = y + pads
        for s in lines:
            screen.blit(s, (x+pads, cy))
            cy += s.get_height() + 6

    def draw_static_attributes(self, screen: pygame.Surface, creature: Creature, x: int, y: int):
        """Draw creature attributes at fixed position"""
        lines = [
            f"{creature.propulsion.value.upper()}",
            f"Size: {creature.L:.2e} m",
            f"Freq: {creature.freq:.1f} Hz",
            f"Re: {creature.Re:.2f}",
            f"Speed: {creature.speed:.3f} m/s",
            f"Efficiency: {creature.eff:.2f}"
        ]
        # Use creature color but ensure it's RGB format
        bg_color = creature.color[:3] if len(creature.color) >= 3 else (10, 15, 25)
        self.panel(screen, lines, x, y, bg_color)


# --------------------
# Game
# --------------------
class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((Config.WIDTH, Config.HEIGHT))
        pygame.display.set_caption("Locomotion Demo – Vertical Layout with Environment Control")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", Config.FONT_SIZE, bold=True)
        self.small = pygame.font.SysFont("Arial", 16)
        self.ui = UI(self.small, self.small)

        # Environment
        self.current_env = "Water"
        self.rho = ENVIRONMENTS[self.current_env]["rho"]
        self.mu = ENVIRONMENTS[self.current_env]["mu"]

        self.scenario = "bacteria"  # or "fish"
        self.show_labels = True
        self._build_scenario()
        self.show_help = False

    def _build_scenario(self):
        cx = Config.WIDTH * 0.5
        cy_top = Config.HEIGHT * 0.3
        cy_bottom = Config.HEIGHT * 0.7

        if self.scenario == "bacteria":
            L = 5e-6
            f = 100.0
            self.top = Creature("Bacterium", cx, cy_top, Config.FIN, L, f, Propulsion.FIN)
            self.bottom = Creature("Bacterium", cx, cy_bottom, Config.FLAG, L, f, Propulsion.FLAGELLA)
        else:
            L = 0.1
            f = 4.0
            self.top = Creature("Fish", cx, cy_top, Config.FIN, L, f, Propulsion.FIN)
            self.bottom = Creature("Fish", cx, cy_bottom, Config.FLAG, L, f, Propulsion.FLAGELLA)

    def _set_environment(self, env_name: str):
        if env_name in ENVIRONMENTS:
            self.current_env = env_name
            env = ENVIRONMENTS[env_name]
            self.rho = env["rho"]
            self.mu = env["mu"]

    def run(self):
        while True:
            dt = self.clock.tick(Config.FPS)/1000.0
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
                if e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_ESCAPE:
                        pygame.quit(); sys.exit()
                    if e.key == pygame.K_1:
                        self.scenario = "bacteria"; self._build_scenario()
                    if e.key == pygame.K_2:
                        self.scenario = "fish"; self._build_scenario()
                    if e.key == pygame.K_f:
                        self.show_labels = not self.show_labels
                    if e.key == pygame.K_h:
                        self.show_help = not self.show_help
                    # Environment keys
                    if e.key == pygame.K_z:
                        self._set_environment("Water")
                    if e.key == pygame.K_x:
                        self._set_environment("Honey")
                    if e.key == pygame.K_c:
                        self._set_environment("Air")
                    if e.key == pygame.K_v:
                        self._set_environment("Oil")
                    if e.key == pygame.K_b:
                        self._set_environment("Cytoplasm")

            keys = pygame.key.get_pressed()

            # Both creatures' frequency controls (Q/A)
            if keys[pygame.K_q]:
                self.top.freq = min(1000, self.top.freq*1.02)
                self.bottom.freq = min(1000, self.bottom.freq*1.02)
            if keys[pygame.K_a]:
                self.top.freq = max(0.1, self.top.freq*0.98)
                self.bottom.freq = max(0.1, self.bottom.freq*0.98)

            # Both creatures' size controls (W/S)
            if keys[pygame.K_w]:
                self.top.L = min(2.0, self.top.L*1.02)
                self.top.amp = max(1e-6, 0.2 * self.top.L)
                self.top.pitch = max(1e-6, 0.2 * self.top.L)
                self.bottom.L = min(2.0, self.bottom.L*1.02)
                self.bottom.amp = max(1e-6, 0.2 * self.bottom.L)
                self.bottom.pitch = max(1e-6, 0.2 * self.bottom.L)
            if keys[pygame.K_s]:
                self.top.L = max(1e-6, self.top.L*0.98)
                self.top.amp = max(1e-6, 0.2 * self.top.L)
                self.top.pitch = max(1e-6, 0.2 * self.top.L)
                self.bottom.L = max(1e-6, self.bottom.L*0.98)
                self.bottom.amp = max(1e-6, 0.2 * self.bottom.L)
                self.bottom.pitch = max(1e-6, 0.2 * self.bottom.L)

            self.top.update(dt, self.rho, self.mu)
            self.bottom.update(dt, self.rho, self.mu)

            # Draw
            env_color = ENVIRONMENTS[self.current_env]["color"]
            # Subtle environment tint
            bg_tinted = tuple(int(Config.BG[i] * 0.8 + env_color[i] * 0.2) for i in range(3))
            self.screen.fill(bg_tinted)

            self.ui.draw_grid(self.screen)
            self.top.draw(self.screen, self.show_labels)
            self.bottom.draw(self.screen, self.show_labels)

            # Static attribute displays on the left
            self.ui.draw_static_attributes(self.screen, self.top, 20, int(self.top.pos.y) - 60)
            self.ui.draw_static_attributes(self.screen, self.bottom, 20, int(self.bottom.pos.y) - 60)

            # Environment info panel (top-right)
            env_lines = [
                f"Environment: {self.current_env}",
                f"Density: {self.rho:.2g} kg/m³",
                f"Viscosity: {self.mu:.2g} Pa·s",
                "",
                f"Scenario: {'Bacteria' if self.scenario=='bacteria' else 'Fish'}",
                "",
                "Expected Winner:",
                f"• Bacteria: {'Flagella' if self.scenario=='bacteria' else 'N/A'}",
                f"• Fish: {'Fins' if self.scenario=='fish' else 'N/A'}",
            ]
            self.ui.panel(self.screen, env_lines, Config.WIDTH-320, 20)

            # Controls reminder (bottom-right)
            control_lines = [
                "Controls:",
                "Q/A=frequency W/S=size",
                "Env: Z/X/C/V/B",
                "1/2=scenario F=labels"
            ]
            self.ui.panel(self.screen, control_lines, Config.WIDTH-220, Config.HEIGHT-140)

            if self.show_help:
                self.ui.help_overlay(self.screen)

            pygame.display.flip()


if __name__ == "__main__":
    Game().run()