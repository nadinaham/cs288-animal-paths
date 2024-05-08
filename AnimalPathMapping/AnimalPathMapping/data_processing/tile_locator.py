def get_image_center_pixels(identifier):
    NUM_HORIZONTAL = get_num_horizontal()
    THERMAL_STRIDE = get_thermal_stride()
    THERMAL_INTERVAL = get_thermal_interval()

    # NOTE: identifier corresopnds to identifier in tile_orthomosaic
    row = np.floor(identifier/NUM_HORIZONTAL)
    col = identifier - NUM_HORIZONTAL*np.floor(identifier/NUM_HORIZONTAL)
    # row and col are all that we need
    x_pixels = col*(THERMAL_STRIDE+THERMAL_INTERVAL/2) + THERMAL_INTERVAL/2
    y_pixels = row*(THERMAL_STRIDE+THERMAL_INTERVAL/2) + THERMAL_INTERVAL/2

    return x_pixels, y_pixels

def get_image_center_meters(x_pixels, y_pixels):
    THERMAL_LEFT = get_thermal_left()
    THERMAL_TOP = get_thermal_top()
    THERMAL_PIXEL_WIDTH = get_thermal_pixel_width()
    THERMAL_PIXEL_HEIGHT = get_thermal_pixel_height()
    
    x = THERMAL_LEFT + x_pixels*THERMAL_PIXEL_WIDTH
    y = THERMAL_TOP + y_pixels*THERMAL_PIXEL_HEIGHT

    return x, y

def tile_together():
    # TODO
    imwrite(
...     'temp.tif',
...     tiles(data, (16, 16)),
...     tile=(16, 16),
...     shape=data.shape,
...     dtype=data.dtype,
...     photometric='rgb'
... )