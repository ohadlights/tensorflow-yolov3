import os
import cv2
import time
import colorsys
import subprocess

import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw

from core import utils


def freeze_graph():
    checkpoint_file = tf.train.latest_checkpoint('./checkpoint')
    print(os.system('d:\\libraries\\python\\python\\python.exe ../convert_weight.py -cf {} -nc 1 -ap widerface_anchors.txt --freeze'.format(checkpoint_file)))
    return checkpoint_file + '.pb'


def draw_boxes(image, boxes, scores, labels, classes, detection_size, show=True):
    """
    :param boxes, shape of  [num, 4]
    :param scores, shape of [num, ]
    :param labels, shape of [num, ]
    :param image,
    :param classes, the return list from the function `read_coco_names`
    """
    if boxes is None: return image
    draw = ImageDraw.Draw(image)
    hsv_tuples = [( x / len(classes), 0.9, 1.0) for x in range(len(classes))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    for i in range(len(labels)): # for each bounding box, do:
        bbox, score, label = boxes[i], scores[i], classes[labels[i]]
        # convert_to_original_size
        detection_size, original_size = np.array(detection_size), np.array(image.size)
        ratio = original_size / detection_size
        bbox = list((bbox.reshape(2,2) * ratio).reshape(-1))
        draw.rectangle(bbox, outline=colors[labels[i]], width=3)

    image.show() if show else None
    return image


def main():

    pb_file = freeze_graph()

    IMAGE_H, IMAGE_W = 416, 416
    classes = utils.read_coco_names('face.names')
    num_classes = len(classes)

    with tf.Session() as sess:

        input_tensor, output_tensors = utils.read_pb_return_tensors(tf.get_default_graph(),
                                                                    pb_file,
                                                                    ["Placeholder:0", "concat_9:0", "mul_6:0"])

        for root, dirs, files in os.walk(r'X:\wider-face\WFLW_images'):
            for file in filter(lambda f: f.endswith('.png') or f.endswith('.jpg'), files):
                path = os.path.join(root, file)

                frame = cv2.imread(path)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame)

                img_resized = np.array(image.resize(size=(IMAGE_H, IMAGE_W)), dtype=np.float32)
                img_resized = img_resized / 255.
                prev_time = time.time()

                boxes, scores = sess.run(output_tensors, feed_dict={input_tensor: np.expand_dims(img_resized, axis=0)})
                boxes, scores, labels = utils.cpu_nms(boxes, scores, num_classes, score_thresh=0.4, iou_thresh=0.5)
                image = draw_boxes(image, boxes, scores, labels, classes, (IMAGE_H, IMAGE_W), show=False)

                curr_time = time.time()
                exec_time = curr_time - prev_time
                result = np.asarray(image)
                info = "time: %.2f ms" %(1000*exec_time)
                cv2.putText(result, text=info, org=(50, 70), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1, color=(255, 0, 0), thickness=2)
                cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
                result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
                cv2.imshow("result", result)
                cv2.waitKey(0)


if __name__ == '__main__':
    main()
