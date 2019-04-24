import random
import argparse

from utils.mongo.mongo_wrapper import get_collection


def main(args):
    client, collection = get_collection('face_read', 'Trio0123', 'rsg_face_train', 'original')
    records = collection.find({'dataset': 'vggface2', 'annotated_mtcnn_rect': {'$exists': True}},
                              {'image_path': 1, 'annotated_mtcnn_rect': 1})
    records = list(records)

    random.seed(0)
    random.shuffle(records)

    train_records = records[:50000]
    test_records = records[50000:55000]

    with open(r'data/train_vgg.txt', 'w') as f:
        for r in train_records:
            box = r['annotated_mtcnn_rect']
            box = [box[0], box[1], box[0] + box[2], box[1] + box[3]]
            path = r['image_path']#.replace('\\', '/').replace(r'//ger/ec/proj/ha/RSG/PersonDataCollection/Training/Recognition/DatabasesReorg/Annotated/', '')
            f.write('{} {} 0\n'.format(path, ' '.join([str(a) for a in box])))

    with open(r'data/test_vgg.txt', 'w') as f:
        for r in test_records:
            box = r['annotated_mtcnn_rect']
            box = [box[0], box[1], box[0] + box[2], box[1] + box[3]]
            path = r['image_path']#.replace('\\', '/').replace(r'//ger/ec/proj/ha/RSG/PersonDataCollection/Training/Recognition/DatabasesReorg/Annotated/', '')
            f.write('{} {} 0\n'.format(path, ' '.join([str(a) for a in box])))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    main(p.parse_args())
