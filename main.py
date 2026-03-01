# main.py
"""Entry point for the Airbud reaction game project.

This script ties together the core engine and the YOLO-based detector.  The
main loop can optionally overlay detection results and will always show the
webcam feed as the background.  When a person is detected a yellow square is
painted on the pygame screen at the location of the first bounding box.
"""

import argparse
import pygame
import cv2

from engine.game import ReactionGame


def draw_button(screen, text, x, y, w, h, hover_color, normal_color):
    mouse = pygame.mouse.get_pos()
    click = pygame.mouse.get_pressed()[0]

    rect = pygame.Rect(x, y, w, h)

    if rect.collidepoint(mouse):
        pygame.draw.rect(screen, hover_color, rect)
        if click:
            return True
    else:
        pygame.draw.rect(screen, normal_color, rect)

    font = pygame.font.SysFont(None, 50)
    label = font.render(text, True, (255, 255, 255))
    screen.blit(label, (x + 20, y + 10))

    return False

def run_menu(screen):
    running = True
    mode = None

    while running:
        screen.fill((20, 20, 20))

        font = pygame.font.SysFont(None, 80)
        title = font.render("AIRBUD", True, (255, 255, 255))
        screen.blit(title, (screen.get_width()//2 - 200, 80))

        if draw_button(screen, "Single Player", 200, 250, 400, 80, (80,80,200), (50,50,150)):
            mode = "single"
            running = False

        if draw_button(screen, "Local 2 Player", 200, 350, 400, 80, (80,200,80), (50,150,50)):
            mode = "local2"
            running = False

        if draw_button(screen, "Online Multiplayer", 200, 450, 400, 80, (200,80,80), (150,50,50)):
            mode = "online"
            running = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        pygame.display.update()

    return mode

def run_game(mode="single", show_camera=True):
    """Start the pygame reaction game.

    The camera feed is drawn as the background; detection code has been removed
    so nothing else is overlaid.  ``show_camera`` is retained for backwards
    compatibility but is always True.
    """
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Reaction Game")
    clock = pygame.time.Clock()

    camera = cv2.VideoCapture(0)
    if camera and not camera.isOpened():
        print("Warning: camera opened but not available")

    mode = "single"
    game = ReactionGame(screen, duration=30, mode=mode)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # capture a frame
        ret, frame = camera.read()
        if not ret:
            print("Error: Could not read frame from webcam.")
            break

        game.update(frame)

        annotated = None
        boxes = []

        # show camera feed as background (always enabled)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        surf = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
        surf = pygame.transform.scale(surf, (800, 600))
        screen.blit(surf, (0, 0))

        # draw game elements on top of whatever background we already painted
        game.draw()
        pygame.display.flip()
        clock.tick(30)

        if game.is_finished():
            running = False

    if camera:
        camera.release()
    pygame.quit()


def main():
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("AIRBUD")

    while True:
        mode = run_menu(screen)
        run_game(mode)


if __name__ == "__main__":
    main()
