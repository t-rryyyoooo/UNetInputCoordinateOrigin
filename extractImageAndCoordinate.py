import argparse
from pathlib import Path
import SimpleITK as sitk
from imageAndCoordinateExtractor import ImageAndCoordinateExtractor
from centerOfGravityCaluculater import CenterOfGravityCaluculater
from functions import getImageWithMeta, getSizeFromString

def ParseArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument("image_path", help="$HOME/Desktop/data/kits19/case_00000/imaging.nii.gz")
    parser.add_argument("label_path", help="$HOME/Desktop/data/kits19/case_00000/segmentation.nii.gz")
    parser.add_argument("save_path", help="$HOME/Desktop/data/slice/hist_0.0/case_00000", default=None)
    parser.add_argument("--mask_path", help="$HOME/Desktop/data/kits19/case_00000/label.mha")
    parser.add_argument("--image_patch_size", help="48-48-16", default="16-48-48")
    parser.add_argument("--label_patch_size", help="48-48-16", default="16-48-48")
    parser.add_argument("--nonmask", action="store_true")
    parser.add_argument("--overlap", help="1", type=int, default=1)

    args = parser.parse_args()
    return args

def main(args):
    """ Read image and label. """
    label = sitk.ReadImage(args.label_path)
    image = sitk.ReadImage(args.image_path)
    if args.mask_path is not None:
        mask = sitk.ReadImage(args.mask_path)
    else:
        mask = None

    """ Get the patch size from string."""
    image_patch_size = getSizeFromString(args.image_patch_size)
    label_patch_size = getSizeFromString(args.label_patch_size)

    center = [0., 0., 0.]
    print("Center: {}".format(center)
    iace = ImageAndCoordinateExtractor(
            image = image, 
            label = label,
            center = center,
            mask = mask,
            image_array_patch_size = image_patch_size,
            label_array_patch_size = label_patch_size,
            overlap = args.overlap,
            )

    iace.execute()
    iace.save(args.save_path, nonmask=args.nonmask)

    """
    il, ll, cl = iace.output()
    p = iace.restore(ll)
    pa = sitk.GetArrayFromImage(p)
    la = sitk.GetArrayFromImage(label)
    from functions import DICE
    dice = DICE(la, pa)
    print(dice)
    """

if __name__ == '__main__':
    args = ParseArgs()
    main(args)
