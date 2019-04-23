import os
import argparse
from collections import defaultdict


def create_list(images_root, annotations_file, output_path):
    content = [l.strip().split() for l in open(annotations_file)]

    annotations = defaultdict(list)
    for l in content:
        path = os.path.join(images_root, l[-1])
        box = [l[-11], l[-10], l[-9], l[-8]]
        annotations[path] += [box]

    with open(output_path, 'w') as f:
        for path, boxes in annotations.items():
            boxes_str = [' '.join(box) + ' 0' for box in boxes]
            f.write('{} {}\n'.format(path, ' '.join(boxes_str)))


def main(args):
    create_list(images_root=args.images_root,
                annotations_file=args.train_annotations_file,
                output_path=args.train_output_path)
    create_list(images_root=args.images_root,
                annotations_file=args.test_annotations_file,
                output_path=args.test_output_path)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--images_root', default=r'X:\wider-face\WFLW_images')
    p.add_argument('--train_annotations_file', default=r'X:\wider-face\WFLW_annotations\list_98pt_rect_attr_train_test\list_98pt_rect_attr_train.txt')
    p.add_argument('--train_output_path', default='train_widerface.txt')
    p.add_argument('--test_annotations_file', default=r'X:\wider-face\WFLW_annotations\list_98pt_rect_attr_train_test\list_98pt_rect_attr_test.txt')
    p.add_argument('--test_output_path', default='test_widerface.txt')
    main(p.parse_args())
