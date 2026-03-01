# object.py
import pygame
import random
import math
import os

class TargetObject:
    def __init__(self, screen_width, screen_height, size=80, image_path=None):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.size = size

        self.x = random.randint(0, screen_width - size)
        self.y = random.randint(0, screen_height - size)

        self.anim = 0

        # Load image if provided
        self.image = None
        if image_path is not None and os.path.exists(image_path):
            raw = pygame.image.load(image_path).convert_alpha()
            self.image = pygame.transform.smoothscale(raw, (size, size))

        # Fallback color if no image
        self.color = (255, 255, 0)

    def respawn(self):
        self.x = random.randint(0, self.screen_width - self.size)
        self.y = random.randint(0, self.screen_height - self.size)
        self.anim = 0

    def get_region(self):
        return self.x, self.y, self.size

    def draw(self, screen):
        self.anim += 1
        scale = 1 + 0.05 * math.sin(self.anim * 0.15)
        size = int(self.size * scale)
        offset = (self.size - size) // 2

        if self.image:
            img = pygame.transform.smoothscale(self.image, (size, size))
            screen.blit(img, (self.x + offset, self.y + offset))
        else:
            rect = pygame.Rect(self.x + offset, self.y + offset, size, size)
            pygame.draw.rect(screen, self.color, rect)