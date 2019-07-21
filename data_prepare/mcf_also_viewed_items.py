import tensorflow as tf
import ast

flags = tf.flags

FLAGS = flags.FLAGS


flags.DEFINE_string(
    "input_data", None,
    "input_json_file"
    "for the task.")


flags.DEFINE_string(
    "output_data", None,
    "output_file")


flags.DEFINE_boolean(
    "also_bought", False,
    "user item data")

flags.DEFINE_string(
    "item_hash_map", None,
    "item_hash_map")


def load_hash_map():
    hash_map = dict()
    with open(FLAGS.item_hash_map, "r") as f:
        for each in f.readlines():
            hash_map[each.split(":")[0]] = each.split(":")[1].strip()
    return hash_map


def also_viewed_data_generator(hash_map):
    with open(FLAGS.input_data, 'r') as f:
        for each in f.readlines():
            data = ast.literal_eval(each.strip())
            raw_ret = []
            if 'related' in data:
                if 'also_viewed' in data['related']:
                    raw_ret.extend(data['related']['also_viewed'])
            if FLAGS.also_bought and 'related' in data:
                if 'also_bought' in data['related']:
                    raw_ret.extend(data['related']['also_bought'])
            ret = []
            for item in raw_ret:
                if item in hash_map:
                    ret.append(hash_map[item])
            if len(ret) != 0 and data['asin'] in hash_map:
                yield hash_map[data['asin']] + '|' + ','.join(ret).strip()


def main(_):
    hash_map = load_hash_map()
    with open(FLAGS.output_data, 'w') as g:
        for line in also_viewed_data_generator(hash_map):
            g.write(line + '\n')


if __name__ == '__main__':
    flags.mark_flag_as_required("input_data")
    flags.mark_flag_as_required("output_data")
    flags.mark_flag_as_required("item_hash_map")

    tf.app.run()
