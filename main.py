import os
import sys
import voronoi
import textrect as tr
import pygame as pg

class Button():
    def __init__(self, img, bounds, txt_in, font, base_col, hover_col, break_style):
        self.img = img
        self.break_style = break_style
        self.x_bound, self.y_bound = bounds[0], bounds[1]
        self.txt_in = txt_in
        self.font = font
        self.base_col, self.hover_col = base_col, hover_col

        self.render_text = self.font.render(self.txt_in, True, self.base_col)
        if self.img is None:
            self.img = self.render_text
        
        self.rect = self.img.get_rect(center=(self.x_bound, self.y_bound))
        self.text_rect = self.render_text.get_rect(center=(self.x_bound, self.y_bound))

    def update(self, SCREEN):
        # blit puts an image on screen
        SCREEN.blit(self.img, self.rect)
        SCREEN.blit(self.render_text, self.text_rect)

    def checkForInput(self, pos):
        if (pos[0] in range(self.rect.left, self.rect.right)) and (pos[1] in range(self.rect.top, self.rect.bottom)):
            return True
        else:
            return False

    def changeColour(self, pos):
        # if mouse is over button (hovering), change the text colour; else default
        if (pos[0] in range(self.rect.left, self.rect.right)) and (pos[1] in range(self.rect.top, self.rect.bottom)):
            self.render_text = self.font.render(self.txt_in, True, self.hover_col)
            self.img = self.break_style
        else:
            self.render_text = self.font.render(self.txt_in, True, self.base_col)
            self.img = self.render_text

def main_menu():
    pg.display.set_caption("Glass Shattering Simulator")

    while True:
        SCREEN.blit(SCALED_BG, (0, 0))

        MENU_MOUSE_POS = pg.mouse.get_pos()
        MENU_TEXT = GAME_FONT.render("MAIN MENU", True, "antiquewhite3")
        MENU_RECT = MENU_TEXT.get_rect(center=(400, 100))

        PLAY_BUTTON = Button(None, (400, 225), "PLAY", GAME_FONT, BASE_COLOUR, HOVER_COLOR, SCALED_SEL_MENU)
        HELP_BUTTON = Button(None, (400, 300), "HELP", GAME_FONT, BASE_COLOUR, HOVER_COLOR, SCALED_SEL_MENU)
        QUIT_BUTTON = Button(None, (400, 375), "QUIT", GAME_FONT, BASE_COLOUR, HOVER_COLOR, SCALED_SEL_MENU)

        for btn in [PLAY_BUTTON, HELP_BUTTON, QUIT_BUTTON]:
            btn.changeColour(MENU_MOUSE_POS)
            btn.update(SCREEN)

        SCREEN.blit(MENU_TEXT, MENU_RECT)

        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                sys.exit()
            if event.type == pg.MOUSEBUTTONDOWN:
                if PLAY_BUTTON.checkForInput(MENU_MOUSE_POS):
                    play()
                if HELP_BUTTON.checkForInput(MENU_MOUSE_POS):
                    help()
                if QUIT_BUTTON.checkForInput(MENU_MOUSE_POS):
                    pg.quit()
                    sys.exit()

        pg.display.update()

def play():
    pg.display.set_caption("Simulator")

    while True:
        SCREEN.fill((0, 0, 0))

        voronoi.main()

        pg.display.update()

def help():
    pg.display.set_caption("Help")

    while True:
        SCREEN.fill((0, 0, 0))
        SCREEN.blit(SCALED_BG, (0, 0))

        HELP_MOUSE_POS = pg.mouse.get_pos()

        BACK_BUTTON = Button(None, (50, 475), "< BACK", HELP_FONT, BASE_COLOUR, HOVER_COLOR, SCALED_SEL_BACK)
        BACK_BUTTON.changeColour(HELP_MOUSE_POS)
        BACK_BUTTON.update(SCREEN)

        HELP_RECT = pg.Rect(100, 50, 600, 400)
        HELP_RENDER = tr.render_textrect(HELP_TEXT, HELP_FONT, HELP_RECT, BASE_COLOUR, (0, 0, 0), 1)

        if HELP_RENDER:
            SCREEN.blit(HELP_RENDER, HELP_RECT.topleft)

        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                sys.exit()
            if event.type == pg.MOUSEBUTTONDOWN:
                if BACK_BUTTON.checkForInput(HELP_MOUSE_POS):
                    main_menu()

        pg.display.update()


if __name__ == "__main__":
    pg.init()

    MAIN_DIR = os.path.dirname(__file__)

    SCREEN = pg.display.set_mode((800, 500))
    GAME_FONT = pg.font.SysFont("Cambria", 40)
    HELP_FONT = pg.font.SysFont("Cambria", 20)

    # assets
    BACKGROUND = pg.image.load(os.path.join(MAIN_DIR, "assets/background_alt4.png"))
    SCALED_BG = pg.transform.scale(BACKGROUND, (800, 500))
    SELECTED_OP = pg.image.load(os.path.join(MAIN_DIR, "assets/option_break.png"))
    SCALED_SEL_MENU = pg.transform.scale(SELECTED_OP, (100, 65))
    SCALED_SEL_BACK = pg.transform.scale(SELECTED_OP, (75, 30))

    HELP_FILE = os.path.join(MAIN_DIR, "assets/help.txt")
    with open(HELP_FILE) as f:
        lines = f.readlines()
    HELP_TEXT = "".join(str(i) for i in lines)

    BASE_COLOUR = "antiquewhite3"
    HOVER_COLOR = "lightsteelblue2"

    main_menu()