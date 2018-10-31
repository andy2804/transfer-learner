import os

from PIL import Image

from semsegmentation.core.segmentator import Segmentator


def load_frames():
    imgs_path = [os.path.join(imgs_folder, x) for x in os.listdir(imgs_folder)
                 if x.endswith(".png")][:30]  # fixme just two for testing
    print("Loaded {:d} images".format(len(imgs_path)))
    return [Image.open(x) for x in imgs_path]


def main():
    imgs = load_frames()
    seg = Segmentator(arch=arch_id)
    for img in imgs:
        img_res, seg_map = seg.run(img)
        seg.vis_segmentation(img_res, seg_map)


if __name__ == '__main__':
    imgs_folder = "/home/azanardi/pictures/zauron_eye/181026103935"
    arch_id = 1
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    main()
