import os
import argparse

import tensorflow as tf
from core import utils, yolov3
from face.dataset import Dataset, Parser

IMAGE_H, IMAGE_W = 416, 416
SHUFFLE_SIZE = 500
CLASSES = utils.read_coco_names('./data/face.names')
NUM_CLASSES = len(CLASSES)
EVAL_INTERNAL = 100
SAVE_INTERNAL = 500


def get_variables(args):
    restore_vars = tf.contrib.framework.get_variables_to_restore()
    train_vars = tf.contrib.framework.get_variables_to_restore()
    if args.reset_head:
        restore_vars = tf.contrib.framework.get_variables_to_restore(include=["yolov3/darknet-53"])
    if args.fine_tune:
        train_vars = tf.contrib.framework.get_variables_to_restore(include=["yolov3/yolo-v3"])
    return restore_vars, train_vars


def get_restore_path(args):
    restore_path = args.restore_path
    checkpoint_path = os.path.join(restore_path, 'checkpoint')
    if os.path.exists(checkpoint_path):
        checkpoint_path = tf.train.latest_checkpoint(restore_path)
    else:
        checkpoint_path = restore_path
    print('Loading checkpoint: {}'.format(checkpoint_path))
    return checkpoint_path


def main(args):

    # configuration

    anchors = utils.get_anchors(args.anchors_path, IMAGE_H, IMAGE_W)

    # datasets

    parser = Parser(IMAGE_H, IMAGE_W, anchors, NUM_CLASSES)
    trainset = Dataset(parser, args.train_list_path, args.batch_size, shuffle=SHUFFLE_SIZE)
    testset = Dataset(parser, args.test_list_path, args.batch_size, shuffle=None)
    steps_per_epoch = int(trainset.num_samples() / args.batch_size)

    # build model

    is_training = tf.placeholder(tf.bool)
    example = tf.cond(is_training, lambda: trainset.get_next(), lambda: testset.get_next())

    images, *y_true = example
    model = yolov3.yolov3(NUM_CLASSES, anchors)

    with tf.variable_scope('yolov3'):
        pred_feature_map = model.forward(images, is_training=is_training)
        loss = model.compute_loss(pred_feature_map, y_true)
        y_pred = model.predict(pred_feature_map)

    # summary

    tf.summary.scalar("loss/coord_loss", loss[1])
    tf.summary.scalar("loss/sizes_loss", loss[2])
    tf.summary.scalar("loss/confs_loss", loss[3])
    tf.summary.scalar("loss/class_loss", loss[4])

    global_step = tf.Variable(0, trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES])
    write_op = tf.summary.merge_all()
    writer_train = tf.summary.FileWriter("./data/train")
    writer_test = tf.summary.FileWriter("./data/test")

    # optimizer

    restore_vars, update_vars = get_variables(args)
    saver_to_restore = tf.train.Saver(var_list=restore_vars)
    learning_rate = tf.train.exponential_decay(args.lr,
                                               global_step,
                                               decay_steps=steps_per_epoch * args.lr_decay_epochs,
                                               decay_rate=args.lr_decay_rate,
                                               staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate)

    # set dependencies for BN ops
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss[0], var_list=update_vars, global_step=global_step)

    # start training

    with tf.Session() as sess:

        # init

        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        saver_to_restore.restore(sess, get_restore_path(args))
        saver = tf.train.Saver(max_to_keep=2)

        # run steps

        for step in range(args.epochs * steps_per_epoch):

            run_items = sess.run([train_op, write_op, y_pred, y_true] + loss, feed_dict={is_training: True})

            if (step + 1) % EVAL_INTERNAL == 0:
                train_rec_value, train_prec_value = utils.evaluate(run_items[2], run_items[3])

            writer_train.add_summary(run_items[1], global_step=step)
            writer_train.flush()  # Flushes the event file to disk
            if (step + 1) % SAVE_INTERNAL == 0: saver.save(sess, save_path="./checkpoint/yolov3.ckpt",
                                                           global_step=step + 1)

            print("=> STEP %10d [TRAIN]:\tloss_xy:%7.4f \tloss_wh:%7.4f \tloss_conf:%7.4f \tloss_class:%7.4f"
                  % (step + 1, run_items[5], run_items[6], run_items[7], run_items[8]))

            run_items = sess.run([write_op, y_pred, y_true] + loss, feed_dict={is_training: False})
            if (step + 1) % EVAL_INTERNAL == 0:
                test_rec_value, test_prec_value = utils.evaluate(run_items[1], run_items[2])
                print("\n=======================> evaluation result <================================\n")
                print(
                    "=> STEP %10d [TRAIN]:\trecall:%7.4f \tprecision:%7.4f" % (
                    step + 1, train_rec_value, train_prec_value))
                print("=> STEP %10d [VALID]:\trecall:%7.4f \tprecision:%7.4f" % (
                step + 1, test_rec_value, test_prec_value))
                print("\n=======================> evaluation result <================================\n")

            writer_test.add_summary(run_items[0], global_step=step)
            writer_test.flush()  # Flushes the event file to disk


if __name__ == '__main__':
    p = argparse.ArgumentParser()

    p.add_argument('--anchors_path', default=r'./data/widerface_anchors.txt')
    p.add_argument('--train_list_path', default=r'./data/train_widerface_vgg.txt')
    p.add_argument('--test_list_path', default=r'./data/test_widerface.txt')

    p.add_argument('--restore_path', default='./../checkpoint/yolov3.ckpt')
    p.add_argument('--fine_tune', action='store_true')
    p.add_argument('--reset_head', action='store_true')

    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--epochs', type=int, default=50)

    p.add_argument('--lr', type=float, default=0.001, help='if Nan, set 0.0005, 0.0001')
    p.add_argument('--lr_decay_rate', type=float, default=0.1)
    p.add_argument('--lr_decay_epochs', type=int, default=10)

    main(p.parse_args())
