import os
import argparse
from collections import defaultdict


def create_list(images_root, annotations_file, output_path):
    content = [l.strip() for l in open(annotations_file)]

    annotations = defaultdict(list)

    i = 0
    while i < len(content):
        path = os.path.join(images_root, content[i])
        i += 1
        num_boxes = int(content[i])
        i += 1
        for i in range(i, i + num_boxes):
            box = [int(a) for a in content[i].split()[0:4]]
            box = [box[0], box[1], box[0] + box[2], box[1] + box[3]]
            box = [str(a) for a in box]
            annotations[path] += [box]
        i += 1

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
    p.add_argument('--images_root', default=r'X:\wider-face\WIDER_all\images')
    p.add_argument('--train_annotations_file', default=r'X:\wider-face\wider_face_split\wider_face_train_bbx_gt.txt')
    p.add_argument('--train_output_path', default='./data/train_widerface.txt')
    p.add_argument('--test_annotations_file', default=r'X:\wider-face\wider_face_split\wider_face_val_bbx_gt.txt')
    p.add_argument('--test_output_path', default='./data/test_widerface.txt')
    main(p.parse_args())
