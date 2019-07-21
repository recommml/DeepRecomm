import tensorflow as tf
import json
import ast
flags = tf.flags

FLAGS = flags.FLAGS


flags.DEFINE_string(
    "also_view_data", None,
    "also_view_data"
    "for the task.")

flags.DEFINE_string(
    "user_item_data", None,
    "user_item_data"
    "for the task.")


flags.DEFINE_string(
    "item_hash_map", None,
    "item_hash_map")

flags.DEFINE_string(
    "user_hash_map", None,
    "user_hash_map")

flags.DEFINE_boolean(
    "also_bought", False,
    "user_hash_map")


def user_item_data_generator():
    with open(FLAGS.user_item_data, 'r') as f:
        for each in f.readlines():
            data = json.loads(each)
            item_id = data['asin']
            user_id = data['reviewerID']
            yield user_id, item_id


def also_view_data_generator():
    with open(FLAGS.also_view_data, 'r') as f:
        for each in f.readlines():
            data = ast.literal_eval(each)
            item_id = data['asin']
            if 'related' in data:
                if 'also_viewed' in data['related']:
                    for item in data['related']['also_viewed']:
                        yield item
            if FLAGS.also_bought and 'related' in data:
                if 'also_bought' in data['related']:
                    for item in data['related']['also_bought']:
                        yield item
            yield item_id


def main(_):
    item_hash_map = dict()
    user_hash_map = dict()
    item_count = 0
    user_count = 0
    for user_id, item_id in user_item_data_generator():
        if item_id not in item_hash_map:
            item_hash_map[item_id] = str(item_count)
            item_count += 1
        if user_id not in user_hash_map:
            user_hash_map[user_id] = str(user_count)
            user_count += 1

    for item_id in also_view_data_generator():
        if item_id not in item_hash_map:
            item_hash_map[item_id] = str(item_count)
            item_count += 1

    with open(FLAGS.item_hash_map, 'w') as g:
        for key, value in item_hash_map.items():
            g.write(key)
            g.write(':')
            g.write(value)
            g.write('\n')

    with open(FLAGS.user_hash_map, 'w') as g:
        for key, value in user_hash_map.items():
            g.write(key)
            g.write(':')
            g.write(value)
            g.write('\n')


if __name__ == '__main__':
    flags.mark_flag_as_required("also_view_data")
    flags.mark_flag_as_required("user_item_data")
    flags.mark_flag_as_required("item_hash_map")
    flags.mark_flag_as_required("user_hash_map")

    tf.app.run()
