# game.py
import pygame
import cv2
from time import time

from .player import Player
from .object import TargetObject
from .collision import MotionCollisionDetector


# -----------------------------
# Timer
# -----------------------------
class GameTimer:
    def __init__(self, duration):
        self.duration = duration
        self.start_time = time()

    def time_left(self):
        remaining = self.duration - (time() - self.start_time)
        return max(0, int(remaining))

    def is_finished(self):
        return self.time_left() <= 0

    def reset(self):
        self.start_time = time()


# -----------------------------
# Main Reaction Game Engine
# -----------------------------
class ReactionGame:
    def __init__(self, screen, duration=30, mode="single"):
        self.screen = screen
        self.width = screen.get_width()
        self.height = screen.get_height()

        self.mode = mode  # "single" or "local2"

        # Timer
        self.timer = GameTimer(duration)
        self.total_time = duration

        # Players
        if mode == "single":
            self.players = [
                Player("Player 1", color=(255, 255, 0))
            ]
        else:
            self.players = [
                Player("Player 1", color=(0, 128, 255)),
                Player("Player 2", color=(255, 0, 0))
            ]

        # Targets
        if mode == "single":
            self.targets = [
                TargetObject(self.width, self.height, size=80, image_path="assets/Duck.png")
            ]
        else:
            self.targets = [
                TargetObject(self.width, self.height, size=80, image_path="assets/Blue Duck.png"),
                TargetObject(self.width, self.height, size=80, image_path="assets/Red Duck.png")
            ]

        # One detector per target
        self.detectors = [MotionCollisionDetector() for _ in self.targets]

        # UI feedback
        self.hit_flash_time = 0

        # Store last webcam frame
        self.last_frame = None

    # -----------------------------
    # Game update logic
    # -----------------------------
    def update(self, frame):
        # Flip webcam horizontally
        frame = cv2.flip(frame, 1)

        # Resize frame to match the game window BEFORE detection
        frame = cv2.resize(frame, (self.width, self.height))

        self.last_frame = frame

        # Check each player/target pair
        for i, target in enumerate(self.targets):
            detector = self.detectors[i]
            player = self.players[i]

            if detector.hand_over_target(frame, target):
                player.add_score()
                target.respawn()
                detector.bg_region = None
                self.hit_flash_time = pygame.time.get_ticks()

        return frame

    # -----------------------------
    # HUD drawing
    # -----------------------------
    def draw_hud(self):
        font = pygame.font.SysFont(None, 50)

        # Player 1 score (left)
        p1 = self.players[0]
        text1 = font.render(f"P1: {p1.score}", True, p1.color)
        self.screen.blit(text1, (20, 20))

        # Player 2 score (right)
        if self.mode == "local2":
            p2 = self.players[1]
            text2 = font.render(f"P2: {p2.score}", True, p2.color)
            self.screen.blit(text2, (self.width - 200, 20))

        # Timer bar (center)
        left = self.timer.time_left()
        ratio = left / self.total_time

        bar_w = 300
        bar_h = 20
        x = (self.width - bar_w) // 2
        y = 20 + 60

        pygame.draw.rect(self.screen, (80, 80, 80), (x, y, bar_w, bar_h))
        pygame.draw.rect(self.screen, (0, 200, 0), (x, y, bar_w * ratio, bar_h))

    # -----------------------------
    # Rendering
    # -----------------------------
    def draw(self):
        # Draw webcam feed
        if self.last_frame is not None:
            frame_rgb = cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2RGB)
            frame_rgb = cv2.resize(frame_rgb, (self.width, self.height))
            frame_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
            self.screen.blit(frame_surface, (0, 0))
        else:
            self.screen.fill((0, 0, 0))

        # Draw all targets
        for target in self.targets:
            target.draw(self.screen)

        # Draw HUD
        self.draw_hud()

        # Hit flash overlay
        if pygame.time.get_ticks() - self.hit_flash_time < 120:
            overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            overlay.fill((255, 255, 255, 60))
            self.screen.blit(overlay, (0, 0))

    # -----------------------------
    # Game state checks
    # -----------------------------
    def is_finished(self):
        return self.timer.is_finished()