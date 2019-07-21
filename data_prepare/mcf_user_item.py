import tensorflow as tf
import json
flags = tf.flags

FLAGS = flags.FLAGS


flags.DEFINE_string(
    "input_data", None,
    "input_json_file"
    "for the task.")


flags.DEFINE_string(
    "output_data", None,
    "output_file")


flags.DEFINE_string(
    "item_hash_map", None,
    "item_hash_map")

flags.DEFINE_string(
    "user_hash_map", None,
    "user_hash_map")


def load_hash_map(path):
    hash_map = dict()
    with open(path, "r") as f:
        for each in f.readlines():
            hash_map[each.split(":")[0]] = each.split(":")[1]
    return hash_map


def user_item_data_generator(item_hash_map, user_hash_map):
    with open(FLAGS.input_data, 'r') as f:
        for each in f.readlines():
            data = json.loads(each)
            item_id = item_hash_map[data['asin']]
            rate = data['overall']
            user_id = user_hash_map[data['reviewerID']]
            yield user_id, item_id, float(rate)


def main(_):
    item_hash_map = load_hash_map(FLAGS.item_hash_map)
    user_hash_map = load_hash_map(FLAGS.user_hash_map)
    result_map = dict()
    for user_id, item_id, rate in user_item_data_generator(item_hash_map, user_hash_map):
        if user_id not in result_map:
            result_map[user_id] = []
        result_map[user_id].append(item_id + ':' + str(rate / 5))
    with open(FLAGS.output_data, 'w') as g:
        for key, value in result_map.items():
            g.write(key)
            g.write('|')
            g.write(','.join(value))
            g.write('\n')


if __name__ == '__main__':
    flags.mark_flag_as_required("input_data")
    flags.mark_flag_as_required("output_data")
    flags.mark_flag_as_required("item_hash_map")
    flags.mark_flag_as_required("user_hash_map")

    tf.app.run()
