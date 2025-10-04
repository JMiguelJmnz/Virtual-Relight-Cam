from rembg import remove

def remove_background(frame):
    # returns RGBA, we’ll just keep RGB for now
    fg = remove(frame)
    return fg[:, :, :3]
