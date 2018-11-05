import os

import matplotlib.animation as animation
from PIL import Image
from matplotlib import pyplot as plt, gridspec

from semsegmentation.core.segmentator import Segmentator


def standard_vis(imgs, segmentator):
    for img in imgs:
        img_res, seg_map = segmentator.run(img)
        segmentator.vis_segmentation(img_res, seg_map)


def generate_movie(imgs, segmentator):
    writer = _get_movie_writer()
    fig = plt.figure(figsize=(10, 7))
    gs = gridspec.GridSpec(2, 2, width_ratios=[6, 1])  # [[6, 1]*2]
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[:, 1])
    with writer.saving(fig, video_name + ".mp4", 500):
        for i, img in enumerate(imgs):
            img_res, seg_map = segmentator.run(img)
            segmentator.vis_movie(ax1, ax2, ax3, img_res, seg_map)
            fig.canvas.draw()
            writer.grab_frame()
            print("\r[{:d}/{:d}] frames converted...".format(i, len(imgs)), end="", flush=True)


def _get_movie_writer():
    FFMpegWriter = animation.writers['ffmpeg']
    metadata = dict(title='movie_test', artist='Matplotlib',
                    comment='Movie support!')
    return FFMpegWriter(fps=30, metadata=metadata)


def load_frames():
    imgs_path = [os.path.join(imgs_folder, x) for x in os.listdir(imgs_folder)
                 if x.endswith(".png")]
    imgs_path.sort()
    imgs_path = imgs_path[270:350]  # fixme just two for testing
    print("Loaded {:d} images".format(len(imgs_path)))
    return [Image.open(x) for x in imgs_path]


def main():
    imgs = load_frames()
    seg = Segmentator(arch=arch_id, input_size=seg_input_size)

    # standard_vis(imgs, seg)
    generate_movie(imgs, seg)


if __name__ == '__main__':
    imgs_folder = "/home/azanardi/pictures/zauron_eye/20181025_170440"
    seg_input_size = 500
    arch_id = 1
    video_name = "live_segmentation"
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    main()
