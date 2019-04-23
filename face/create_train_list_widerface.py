import os
import argparse
from collections import defaultdict


def main(args):
    content = [l.strip().split() for l in open(args.annotations_file)]

    annotations = defaultdict(list)
    for l in content:
        path = os.path.join(args.images_root, l[-1])
        box = [l[-11], l[-10], l[-9], l[-8]]
        annotations[path] += [box]

    with open(args.output_path, 'w') as f:
        for path, boxes in annotations.items():
            boxes_str = [' '.join(box) + ' 0' for box in boxes]
            f.write('{} {}\n'.format(path, ' '.join(boxes_str)))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--images_root', default=r'X:\wider-face\WFLW_images')
    p.add_argument('--annotations_file', default=r'X:\wider-face\WFLW_annotations\list_98pt_rect_attr_train_test\list_98pt_rect_attr_train.txt')
    p.add_argument('--output_path', default='train_widerface.txt')
    main(p.parse_args())
