import pygame
import math
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Tuple, List
import analysis


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

    SPEED_VISUAL_BACTERIA: float = 50000.0
    SPEED_VISUAL_FISH: float = 250.0


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
# Physics (simple)
# --------------------
def reynolds(rho: float, U: float, L: float, mu: float) -> float:
    if mu <= 0 or L <= 0:
        return 0.0
    return (rho * U * L) / mu


def cd_blend(Re: float) -> float:
    if Re < 1:
        return 24 / max(Re, 1e-2)
    elif Re < 1e3:
        return 24 / Re + 6 / (1 + math.sqrt(Re)) + 0.4
    else:
        return 0.44


def fin_eff(Re: float) -> float:
    x = math.log10(max(Re, 1e-6))
    t = 1 / (1 + math.exp(-(x - 0.5) * 2.5))
    return max(0.02, min(1.0, t))


def flag_eff(Re: float) -> float:
    x = math.log10(max(Re, 1e-6))
    t = 1 / (1 + math.exp((x - 0.0) * 2.5))
    return max(0.05, min(1.2, t * 1.2))


# --------------------
# Creature
# --------------------
class Creature:
    def __init__(self, label: str, x: float, y: float, color: Tuple[int, int, int],
                 L: float, freq: float, propulsion: Propulsion, is_bacteria: bool = False):
        self.label = label
        self.start_x = x
        self.pos = pygame.Vector2(x, y)
        self.vel = pygame.Vector2(0.0, 0.0)
        self.color = color
        self.L = L
        self.freq = freq
        self.amp = max(1e-6, 0.2 * L)
        self.pitch = max(1e-6, 0.2 * L)
        self.propulsion = propulsion
        self.anim_t = 0.0
        self.Re = 0.0
        self.speed = 0.0
        self.eff = 0.0
        self.is_bacteria = is_bacteria

    def radius_px(self) -> int:
        if self.L < 1e-4: return 15
        if self.L < 1e-2: return 22
        if self.L < 1e-1: return 35
        if self.L < 5e-1: return 48
        return 60

    def update(self, dt: float, rho: float, mu: float):
        U = self.vel.x
        Re = reynolds(rho, abs(U), self.L, mu)

        if self.propulsion == Propulsion.FIN:
            eff = fin_eff(Re)
            thrust = 0.5 * rho * (self.freq * self.amp) ** 2 * self.L ** 2 * eff
        else:
            eff = flag_eff(Re)
            thrust = 6 * math.pi * mu * (self.L / 2) * (self.freq * self.pitch) * eff

        r = self.L / 2
        A = math.pi * r * r
        Cd = cd_blend(max(Re, 1e-6))
        drag_linear = 6 * math.pi * mu * r * abs(U)
        drag_quad = 0.5 * rho * Cd * A * U * U
        drag = drag_linear + drag_quad

        mass = max(1e-9, rho * (4 / 3) * math.pi * r ** 3)
        net = thrust - drag
        ax = net / mass
        self.vel.x += ax * dt
        self.vel.x = max(0.0, min(self.vel.x, 3.0))

        visual_scale = Config.SPEED_VISUAL_BACTERIA if self.is_bacteria else Config.SPEED_VISUAL_FISH
        self.pos.x += self.vel.x * dt * visual_scale

        self.Re = reynolds(rho, abs(self.vel.x), self.L, mu)
        self.speed = self.vel.x
        self.eff = eff

        if self.pos.x > Config.WIDTH + 60:
            self.pos.x = self.start_x

        self.anim_t += dt

    def draw(self, screen: pygame.Surface, show_labels: bool = True):
        rpx = self.radius_px()
        pos = (int(self.pos.x), int(self.pos.y))
        pygame.draw.circle(screen, self.color, pos, rpx)
        pygame.draw.circle(screen, (255, 255, 255), pos, rpx, 2)

        if self.propulsion == Propulsion.FIN:
            fin_length = 2.5 * rpx
            fin_angle = math.sin(self.anim_t * 8) * 0.4
            tail_points = [
                (pos[0] - rpx, pos[1]),
                (pos[0] - rpx - fin_length, pos[1] - int(fin_length * 0.5 * math.sin(fin_angle))),
                (pos[0] - rpx - fin_length, pos[1] + int(fin_length * 0.5 * math.sin(fin_angle)))
            ]
            pygame.draw.polygon(screen, self.color, tail_points)
            pygame.draw.polygon(screen, (255, 255, 255), tail_points, 2)
        else:
            flag_points = []
            length = 3.5 * rpx
            segments = 20
            for i in range(segments + 1):
                t = i / segments
                x = pos[0] - rpx - int(t * length)
                y = pos[1] + int(0.4 * rpx * math.sin(12 * t + self.anim_t * 15))
                flag_points.append((x, y))
            if len(flag_points) > 1:
                pygame.draw.lines(screen, self.color, False, flag_points, 3)

        if show_labels:
            lbl = f"{self.label} — {self.propulsion.value}"
            font = pygame.font.SysFont("Arial", 16, bold=True)
            surf = font.render(lbl, True, self.color)
            screen.blit(surf, (pos[0] - surf.get_width() // 2, pos[1] - rpx - 25))


# --------------------
# UI
# --------------------
class UI:
    def __init__(self, font_main, font_small, font_bold):
        self.font = font_main
        self.small = font_small
        self.bold = font_bold

    def draw_grid(self, screen):
        step = 50
        for x in range(0, Config.WIDTH, step):
            pygame.draw.line(screen, Config.GRID, (x, 0), (x, Config.HEIGHT), 1)
        for y in range(0, Config.HEIGHT, step):
            pygame.draw.line(screen, Config.GRID, (0, y), (Config.WIDTH, y), 1)

    def panel(self, screen, lines: List[str], x: int, y: int, bg_color=(10, 15, 25), font=None):
        font_to_use = font if font is not None else self.font
        pads = 12
        spacing = 3
        line_surfs = [font_to_use.render(line, True, Config.TEXT) for line in lines if line.strip()]
        if not line_surfs: return
        w = max(s.get_width() for s in line_surfs) + pads * 2
        h = sum(s.get_height() for s in line_surfs) + spacing * (len(line_surfs) - 1) + pads * 2
        surf = pygame.Surface((w, h), pygame.SRCALPHA)
        fill_color = (bg_color[0], bg_color[1], bg_color[2], 210) if len(bg_color) >= 3 else (10, 15, 25, 210)
        surf.fill(fill_color)
        pygame.draw.rect(surf, (255, 255, 255, 30), surf.get_rect(), 1)
        screen.blit(surf, (x, y))
        cy = y + pads
        for s in line_surfs:
            screen.blit(s, (x + pads, cy))
            cy += s.get_height() + spacing

    def help_overlay(self, screen):
        text = [
            "LOCOMOTION DEMO - VERTICAL LAYOUT",
            "",
            "Scenarios: 1=Bacteria  2=Fish",
            "",
            "Controls:",
            "Q/A: Frequency       W/S: Size",
            "E/D: Speed Decimals  R/T: Reynolds Decimals",
            "",
            "Environments:",
            "Z=Water  X=Honey  C=Air  V=Oil  B=Cytoplasm",
            "",
            "Other: F=toggle labels  H=close help"
        ]
        pads = 20
        lines = [self.font.render(t, True, Config.TEXT) for t in text]
        w = max(s.get_width() for s in lines) + pads * 2
        h = sum(s.get_height() for s in lines) + pads * 2 + 6 * (len(lines) - 1)
        x = (Config.WIDTH - w) // 2
        y = (Config.HEIGHT - h) // 2
        box = pygame.Surface((w, h), pygame.SRCALPHA)
        box.fill((5, 10, 15, 240))
        pygame.draw.rect(box, Config.HI, box.get_rect(), 3)
        screen.blit(box, (x, y))
        cy = y + pads
        for s in lines:
            screen.blit(s, (x + pads, cy))
            cy += s.get_height() + 6

    def draw_static_attributes(self, screen, creature: Creature, x: int, y: int, speed_prec: int, re_prec: int):
        lines = [
            f"{creature.propulsion.value.upper()}",
            f"Size: {creature.L:.2e} m",
            f"Freq: {creature.freq:.1f} Hz",
            f"Re ({re_prec}f): {creature.Re:.{re_prec}f}",
            f"Speed ({speed_prec}f): {creature.speed:.{speed_prec}f} m/s",
            f"Efficiency: {creature.eff:.2f}"
        ]
        bg_color = creature.color[:3] if len(creature.color) >= 3 else (10, 15, 25)
        self.panel(screen, lines, x, y, bg_color, font=self.bold)


# --------------------
# Game
# --------------------
class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((Config.WIDTH, Config.HEIGHT))
        pygame.display.set_caption("Locomotion Demo")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", Config.FONT_SIZE, bold=True)
        self.small = pygame.font.SysFont("Arial", 16)
        self.bold = pygame.font.SysFont("Arial", 18, bold=True)
        self.ui = UI(self.small, self.small, self.bold)

        self.current_env = "Water"
        self.rho = ENVIRONMENTS[self.current_env]["rho"]
        self.mu = ENVIRONMENTS[self.current_env]["mu"]

        self.speed_precision = 10
        self.re_precision = 10

        self.scenario = "bacteria"
        self.show_labels = True
        self.show_help = False
        self._build_scenario()

    def _build_scenario(self):
        cx = Config.WIDTH * 0.5
        cy_top = Config.HEIGHT * 0.3
        cy_bottom = Config.HEIGHT * 0.7
        if self.scenario == "bacteria":
            L = 5e-6;
            f = 100.0
            self.top = Creature("Bacterium", cx, cy_top, Config.FIN, L, f, Propulsion.FIN, True)
            self.bottom = Creature("Bacterium", cx, cy_bottom, Config.FLAG, L, f, Propulsion.FLAGELLA, True)
        else:
            L = 0.1;
            f = 4.0
            self.top = Creature("Fish", cx, cy_top, Config.FIN, L, f, Propulsion.FIN, False)
            self.bottom = Creature("Fish", cx, cy_bottom, Config.FLAG, L, f, Propulsion.FLAGELLA, False)

    def _set_environment(self, env_name: str):
        if env_name in ENVIRONMENTS:
            self.current_env = env_name
            env = ENVIRONMENTS[env_name]
            self.rho = env["rho"]
            self.mu = env["mu"]

    def run(self):
        while True:
            dt = self.clock.tick(Config.FPS) / 1000.0
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    pygame.quit();
                    sys.exit()
                if e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_ESCAPE: pygame.quit(); sys.exit()
                    if e.key == pygame.K_1: self.scenario = "bacteria"; self._build_scenario()
                    if e.key == pygame.K_2: self.scenario = "fish"; self._build_scenario()
                    if e.key == pygame.K_f: self.show_labels = not self.show_labels
                    if e.key == pygame.K_h: self.show_help = not self.show_help
                    if e.key == pygame.K_z: self._set_environment("Water")
                    if e.key == pygame.K_x: self._set_environment("Honey")
                    if e.key == pygame.K_c: self._set_environment("Air")
                    if e.key == pygame.K_v: self._set_environment("Oil")
                    if e.key == pygame.K_b: self._set_environment("Cytoplasm")
                    # -- Precision controls --
                    if e.key == pygame.K_e: self.speed_precision = min(20, self.speed_precision + 1)
                    if e.key == pygame.K_d: self.speed_precision = max(0, self.speed_precision - 1)
                    if e.key == pygame.K_r: self.re_precision = min(20, self.re_precision + 1)
                    if e.key == pygame.K_t: self.re_precision = max(0, self.re_precision - 1)
                    if e.key == pygame.K_g:
                        scen_name = self.scenario  # "bacteria" or "fish"
                        L = self.top.L  # both top/bottom share L & freq
                        f = self.top.freq
                        outdir = "./plots_export"
                        analysis.run_all_plots(outdir, scen_name, L, f, ENVIRONMENTS)
                        print(f"[Analysis] Exported plots & CSV to {outdir}")

            keys = pygame.key.get_pressed()
            if keys[pygame.K_q]: self.top.freq = self.bottom.freq = min(1000, self.top.freq * 1.02)
            if keys[pygame.K_a]: self.top.freq = self.bottom.freq = max(0.1, self.top.freq * 0.98)
            if keys[pygame.K_w]:
                for c in [self.top, self.bottom]:
                    c.L = min(2.0, c.L * 1.02)
                    c.amp = max(1e-6, 0.2 * c.L)
                    c.pitch = max(1e-6, 0.2 * c.L)
            if keys[pygame.K_s]:
                for c in [self.top, self.bottom]:
                    c.L = max(1e-6, c.L * 0.98)
                    c.amp = max(1e-6, 0.2 * c.L)
                    c.pitch = max(1e-6, 0.2 * c.L)

            self.top.update(dt, self.rho, self.mu)
            self.bottom.update(dt, self.rho, self.mu)

            env_color = ENVIRONMENTS[self.current_env]["color"]
            bg_tinted = tuple(int(Config.BG[i] * 0.8 + env_color[i] * 0.2) for i in range(3))
            self.screen.fill(bg_tinted)

            self.ui.draw_grid(self.screen)
            self.top.draw(self.screen, self.show_labels)
            self.bottom.draw(self.screen, self.show_labels)

            self.ui.draw_static_attributes(self.screen, self.top, 20, int(self.top.pos.y) - 60, self.speed_precision,
                                           self.re_precision)
            self.ui.draw_static_attributes(self.screen, self.bottom, 20, int(self.bottom.pos.y) - 60,
                                           self.speed_precision, self.re_precision)

            env_lines = [
                f"Environment: {self.current_env}",
                f"Density: {self.rho:.2g} kg/m³",
                f"Viscosity: {self.mu:.2g} Pa·s",
                "",
                f"Scenario: {'Bacteria' if self.scenario == 'bacteria' else 'Fish'}",
                "",
                "Expected Winner:",
                f"• Bacteria: {'Flagella' if self.scenario == 'bacteria' else 'N/A'}",
                f"• Fish: {'Fins' if self.scenario == 'fish' else 'N/A'}"
            ]
            self.ui.panel(self.screen, env_lines, Config.WIDTH - 320, 20)

            control_lines = [
                "Controls:",
                "Q/A=freq  W/S=size",
                "E/D=spd prec, R/T=Re prec",
                "Env: Z/X/C/V/B",
                "1/2=scenario F=labels"
            ]
            self.ui.panel(self.screen, control_lines, Config.WIDTH - 220, Config.HEIGHT - 140)

            if self.show_help:
                self.ui.help_overlay(self.screen)

            pygame.display.flip()


if __name__ == "__main__":
    Game().run()