import tensorflow as tf
import os
from tensor2tensor.data_generators import generator_utils
flags = tf.flags

FLAGS = flags.FLAGS


flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")


flags.DEFINE_string(
    "tmp_dir", None,
    "The output directory where the model checkpoints will be written.")


flags.DEFINE_string(
    "user_item", None,
    "user item data")

flags.DEFINE_string(
    "also_view", None,
    "also view data")


flags.DEFINE_integer(
    "max_seq_length", 224,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")


def user_item_example_generator(dir_name, tag='train'):
    with open(os.path.join(dir_name, "user_item_{}".format(tag)), 'r') as user_item:
        for each in user_item.readlines():
            user_id = int(each.split("|")[0])
            item_rate = each.split("|")[1]
            feature = dict()
            items = list(map(lambda x: int(x.split(":")[0]), item_rate.split(",")))[0: FLAGS.max_seq_length]
            rates = list(map(lambda x: float(x.split(":")[1]), item_rate.split(",")))[0: FLAGS.max_seq_length]
            feature['user_id'] = [user_id]
            feature['item_ids'] = items
            feature['rates'] = rates
            yield feature


def also_view_example_generator(dir_name, tag="train"):
    with open(os.path.join(dir_name, "also_view_{}".format(tag)), 'r') as user_item:
        for each in user_item.readlines():
            item_id = int(each.split("|")[0])
            also_view_items = each.split("|")[1]
            feature = dict()
            also_view_ids = list(map(lambda x: int(x), also_view_items.split(",")))[0: FLAGS.max_seq_length]
            feature['item_id'] = [item_id]
            feature['also_view_ids'] = also_view_ids
            yield feature


def main(_):
    train_shards = 100
    dev_shards = 1
    pred_shards = 1
    user_item_train_file_names = [os.path.join(FLAGS.data_dir, "{0}-train-000{1}-of-00{2}"
                                     .format(FLAGS.user_item, i, train_shards))
                        for i in range(train_shards)]

    user_item_dev_file_names = [os.path.join(FLAGS.data_dir, "{0}-dev-000{1}-of-00{2}"
                                   .format(FLAGS.user_item, i, dev_shards))
                      for i in range(dev_shards)]

    user_item_pred_file_names = [os.path.join(FLAGS.data_dir, "{0}-pred-000{1}-of-00{2}"
                                    .format(FLAGS.user_item,i, pred_shards))
                       for i in range(pred_shards)]

    user_item_train_generator = user_item_example_generator(FLAGS.tmp_dir, "train")

    user_item_dev_generator = user_item_example_generator(FLAGS.tmp_dir, "dev")

    user_item_pred_generator = user_item_example_generator(FLAGS.tmp_dir, "pred")

    generator_utils.generate_files(user_item_train_generator, user_item_train_file_names, cycle_every_n=10)

    generator_utils.generate_files(user_item_dev_generator, user_item_dev_file_names, cycle_every_n=10)

    generator_utils.generate_files(user_item_pred_generator, user_item_pred_file_names, cycle_every_n=10)

    generator_utils.shuffle_dataset(user_item_train_file_names)

    generator_utils.shuffle_dataset(user_item_dev_file_names)

    generator_utils.shuffle_dataset(user_item_pred_file_names)

    also_view_train_file_names = [os.path.join(FLAGS.data_dir, "{0}-train-000{1}-of-00{2}"
                                     .format(FLAGS.also_view, i, train_shards))
                        for i in range(train_shards)]

    also_view_dev_file_names = [os.path.join(FLAGS.data_dir, "{0}-dev-000{1}-of-00{2}"
                                   .format(FLAGS.also_view, i, dev_shards))
                      for i in range(dev_shards)]

    also_view_pred_file_names = [os.path.join(FLAGS.data_dir, "{0}-pred-000{1}-of-00{2}"
                                    .format(FLAGS.also_view, i, pred_shards))
                       for i in range(pred_shards)]

    also_view_train_generator = also_view_example_generator(FLAGS.tmp_dir, "train")

    also_view_dev_generator = also_view_example_generator(FLAGS.tmp_dir, "dev")

    also_view_pred_generator = also_view_example_generator(FLAGS.tmp_dir, "pred")

    generator_utils.generate_files(also_view_train_generator, also_view_train_file_names, cycle_every_n=10)

    generator_utils.generate_files(also_view_dev_generator, also_view_dev_file_names, cycle_every_n=10)

    generator_utils.generate_files(also_view_pred_generator, also_view_pred_file_names, cycle_every_n=10)

    generator_utils.shuffle_dataset(also_view_train_file_names)

    generator_utils.shuffle_dataset(also_view_dev_file_names)

    generator_utils.shuffle_dataset(also_view_pred_file_names)


if __name__ == '__main__':
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("user_item")
    flags.mark_flag_as_required("also_view")
    flags.mark_flag_as_required("vocab")
    flags.mark_flag_as_required("tmp_dir")
    tf.app.run()
