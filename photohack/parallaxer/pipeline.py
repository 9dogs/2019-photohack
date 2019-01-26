"""Image to GIF with parallax pipeline."""
from PIL import Image


def process_file(filename, depth_model, segmentation_model):
    """

    :param str filename: filename
    """
    image = Image.open(filename)
    seg_map, depth_map = None, None
    if segmentation_model:
        resized_image, seg_map = segmentation_model.run(image)
    if depth_model:
        resized_depth_map, depth_map = depth_model.run(image)
    return depth_map, seg_map
