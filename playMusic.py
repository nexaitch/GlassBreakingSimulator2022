# note: timer function is not provided hence the loop is in permanent input state
# you will have to provide input 'e' in order to exit the program & finish the script

import pygame.mixer as pgmx

fname = 'Desktop\Astronaut In The Ocean.mp3'
pgmx.init()
pgmx.music.load(fname)
pgmx.music.play()

loop_coord = pgmx.music.get_busy()

while loop_coord:
    print("Press 'p' to pause, 'r' to resume, 'e' to exit")
    query = input("  ")
      
    if query == 'p':
        # pause
        pgmx.music.pause()
        print("you have paused")

    elif query == 'r':
        # resume
        pgmx.music.unpause()
        print("you have resumed")

    elif query == 'e':
        # stop
        pgmx.music.stop()
        print("you have stopped")
        break

print("music has ended")