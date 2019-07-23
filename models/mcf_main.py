# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""
import os
import mcf_modeling
import optimization
import tensorflow as tf
from tensorflow.python.client import device_lib

PAD_ID = 0

flags = tf.flags

FLAGS = flags.FLAGS

# Required parameters

flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "mcf_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")


flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

# Other parameters

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")


flags.DEFINE_bool("do_train", True, "Whether to run training.")

flags.DEFINE_bool("do_eval", True, "Whether to run training.")

flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_steps", 100000,
                   "Total number of training epochs to perform.")

flags.DEFINE_integer(
    "num_warmup_steps", 1500,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("max_also_view_length", 128,
                     "max_also_view_length")

flags.DEFINE_integer("max_items_length", 128,
                     "max_items_length")

flags.DEFINE_float("also_view_ratio", 0.1,
                   "also_view_ratio")

flags.DEFINE_float("l2_penalty", 0.1, "l2_penalty")


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def file_based_input_fn_builder(user_item_files, also_view_files, is_training, batch_size):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    user_items_name_to_features = {
        "user_id": tf.VarLenFeature(tf.int64),
        "item_ids": tf.VarLenFeature(tf.int64),
        "rates": tf.VarLenFeature(tf.float32)
    }

    also_view_name_to_features = {
        "item_id": tf.VarLenFeature(tf.int64),
        "also_view_ids": tf.VarLenFeature(tf.int64)
    }

    def _user_items_decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

        return example["user_id"].values, example["item_ids"].values, example["rates"].values

    def _also_view_decode_record(record, name_to_features):
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

        return example["item_id"].values, example["also_view_ids"].values

    def input_fn():
        """The actual input function."""

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        user_item_dataset = tf.data.TFRecordDataset(user_item_files)
        also_view_dataset = tf.data.TFRecordDataset(also_view_files)

        if is_training:
            user_item_dataset = user_item_dataset.repeat()
            user_item_dataset = user_item_dataset.shuffle(buffer_size=100)
            also_view_dataset = also_view_dataset.repeat()
            also_view_dataset = also_view_dataset.shuffle(buffer_size=100)

        d = tf.data.Dataset.zip((user_item_dataset, also_view_dataset))

        d = d.map(lambda user_items, also_view: (_user_items_decode_record(user_items, user_items_name_to_features),
                                                 _also_view_decode_record(also_view, also_view_name_to_features)))

        d = d.map(lambda user_items_feature, also_view_feature: (
            user_items_feature[1][:FLAGS.max_items_length],
            user_items_feature[2][:FLAGS.max_items_length],
            also_view_feature[1][:FLAGS.max_also_view_length],
            user_items_feature[0],
            also_view_feature[0]
        ))

        d = d.map(lambda item_ids, rates, also_view_ids, user_id, item_id: (
            item_ids,
            rates,
            also_view_ids,
            tf.ones_like(item_ids, dtype=tf.float32),
            tf.ones_like(also_view_ids,  dtype=tf.float32),
            user_id,
            item_id
        ))

        def batching_func(x):
            return x.padded_batch(
                batch_size,
                # The entry is the source line rows;
                # this has unknown-length vectors.  The last entry is
                # the source row size; this is a scalar.
                padded_shapes=(
                    tf.TensorShape([None]),  # item_ids
                    tf.TensorShape([None]),  # rates
                    tf.TensorShape([None]),  # also_view_ids
                    tf.TensorShape([None]),  # items_ids mask
                    tf.TensorShape([None]),  # also_view_mask
                    tf.TensorShape([None]),  # user_id
                    tf.TensorShape([None])),  # item_id
                # Pad the source sequences with eos tokens.
                # (Though notice we don't generally need to do this since
                # later on we will be masking out calculations past the true sequence.
                padding_values=(
                    PAD_ID,  # src
                    float(PAD_ID),
                    PAD_ID,
                    float(PAD_ID),
                    float(PAD_ID),
                    PAD_ID,
                    PAD_ID))  # src_len -- unused

        batched_dataset = batching_func(d)
        features = \
            batched_dataset.map(lambda item_ids, rates, also_view_ids, items_mask, also_view_mask, user_id, item_id:
                                {
                                    "item_ids": item_ids,
                                    "rates": rates,
                                    "also_view_ids": also_view_ids,
                                    "items_mask": items_mask,
                                    "also_view_mask": also_view_mask,
                                    "user_id": user_id,
                                    "item_id": item_id
                                })

        return features

    return input_fn


def create_model(mcf_config, user_id, item_id, item_ids, also_view_ids, rates, items_mask, also_view_mask,
                 use_one_hot_embeddings):
    """Creates a classification model."""

    model = mcf_modeling.MCFModel(
        config=mcf_config,
        user_id=user_id,
        item_id=item_id,
        item_ids=item_ids,
        also_view_ids=also_view_ids,
        use_one_hot_embeddings=use_one_hot_embeddings,
        scope=None)

    # In the demo, we are doing a simple classification task on the entire
    # segment.
    #
    # If you want to use the token-level output, use model.get_sequence_output()
    # instead.
    user_item_prod, also_view_prod = model.get_user_item_products()
    with tf.variable_scope("scaled_prod"):
        scaled_user_item_prod = tf.sigmoid(user_item_prod, name='scaled_user_item_prod')
        scaled_also_view_prod = tf.sigmoid(also_view_prod, name='scaled_also_view_prod')

    with tf.variable_scope("loss"):
        user_item_prod_raw_loss = scaled_user_item_prod - rates
        also_view_prod_raw_loss = scaled_also_view_prod - 1

        mask_user_item_loss = user_item_prod_raw_loss * items_mask
        mask_also_view_loss = also_view_prod_raw_loss * also_view_mask

        user_item_loss = tf.reduce_sum(mask_user_item_loss ** 2, axis=1, name='user_item_loss')
        also_view_loss = tf.reduce_sum(mask_also_view_loss ** 2, axis=1, name='also_view_loss')

        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

        loss = tf.reduce_mean((user_item_loss + FLAGS.also_view_ratio * also_view_loss)) + \
            FLAGS.l2_penalty * tf.reduce_mean(reg_losses)

        return model, loss, user_item_loss, also_view_loss


def model_fn_builder(mcf_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        item_ids = features["item_ids"]
        rates = features["rates"]
        also_view_ids = features["also_view_ids"]
        items_mask = features["items_mask"]
        also_view_mask = features["also_view_mask"]
        user_id = features["user_id"]
        item_id = features["item_id"]

        (model, loss, user_item_loss, also_view_loss) = create_model(mcf_config, user_id, item_id, item_ids,
                                                                     also_view_ids, rates, items_mask,
                                                                     also_view_mask, use_one_hot_embeddings)
        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = mcf_modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        if mode == tf.estimator.ModeKeys.TRAIN:

            train_op = optimization.create_adam_optimizer(
                loss, learning_rate, num_train_steps, num_warmup_steps, False)

            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=train_op,
            )
        elif mode == tf.estimator.ModeKeys.EVAL:

            eval_metrics = {
                    "user_item_loss": tf.metrics.mean(user_item_loss),
                    "also_view_loss": tf.metrics.mean(also_view_loss)
                }

            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                eval_metric_ops=eval_metrics
            )
        else:
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                predictions={"item_embedding": model.item_embedding_output,
                             "user_embedding": model.user_embedding_output}
            )
        return output_spec

    return model_fn


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    if not FLAGS.do_train and not FLAGS.do_predict and not FLAGS.do_eval:
        raise ValueError(
            "At least one of `do_train`, or `do_predict' must be True.")

    mcf_config = mcf_modeling.MCFConfig.from_json_file(FLAGS.mcf_config_file)

    tf.gfile.MakeDirs(FLAGS.output_dir)

    mirrored_strategy = tf.contrib.distribute.MirroredStrategy(
        devices=get_available_gpus(),
    )

    run_config = tf.estimator.RunConfig(
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        keep_checkpoint_max=20,
        train_distribute=mirrored_strategy,
        eval_distribute=mirrored_strategy
    )

    model_fn = model_fn_builder(
        mcf_config=mcf_config,
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=FLAGS.num_train_steps,
        num_warmup_steps=FLAGS.num_warmup_steps,
        use_one_hot_embeddings=False)

    # or GPU.
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config
    )

    if FLAGS.do_train:
        train_files = os.listdir(FLAGS.data_dir)
        user_item_train_files = [os.path.join(FLAGS.data_dir, path) for path in train_files if "user_item-train"in path]
        also_view_train_files = [os.path.join(FLAGS.data_dir, path) for path in train_files if "also_view-train"in path]

        tf.logging.info("***** Running training *****")
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", FLAGS.num_train_steps)

        train_input_fn = file_based_input_fn_builder(
            user_item_files=user_item_train_files,
            also_view_files=also_view_train_files,
            is_training=True,
            batch_size=FLAGS.train_batch_size)

        train_spec = tf.estimator.TrainSpec(
            input_fn=train_input_fn,
            max_steps=FLAGS.num_train_steps
        )

        eval_files = os.listdir(FLAGS.data_dir)
        user_item_eval_files = [os.path.join(FLAGS.data_dir, path) for path in eval_files if "user_item-dev" in path]
        also_view_eval_files = [os.path.join(FLAGS.data_dir, path) for path in eval_files if "also_view-dev" in path]
        # This tells the estimator to run through the entire set.

        # However, if running eval on the TPU, you will need to specify the
        # number of steps.

        eval_input_fn = file_based_input_fn_builder(
            user_item_files=user_item_eval_files,
            also_view_files=also_view_eval_files,
            is_training=False,
            batch_size=FLAGS.eval_batch_size)

        eval_spec = tf.estimator.EvalSpec(
            input_fn=eval_input_fn,
            steps=None,
            throttle_secs=30
        )

        tf.estimator.train_and_evaluate(
            estimator,
            train_spec,
            eval_spec
        )

    if FLAGS.do_eval:
        eval_files = os.listdir(FLAGS.data_dir)
        user_item_eval_files = [os.path.join(FLAGS.data_dir, path) for path in eval_files if "user_item-dev" in path]
        also_view_eval_files = [os.path.join(FLAGS.data_dir, path) for path in eval_files if "also_view-dev" in path]

        # This tells the estimator to run through the entire set.
        eval_steps = None
        # However, if running eval on the TPU, you will need to specify the
        # number of steps.

        eval_input_fn = file_based_input_fn_builder(
            user_item_files=user_item_eval_files,
            also_view_files=also_view_eval_files,
            is_training=False,
            batch_size=FLAGS.eval_batch_size)

        result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)

        output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")

        with tf.gfile.GFile(output_eval_file, "w") as writer:
            tf.logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                tf.logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    if FLAGS.do_predict:

        predict_files = os.listdir(FLAGS.data_dir)
        user_item_eval_files = \
            [os.path.join(FLAGS.data_dir, path) for path in predict_files if "user_item-pred" in path]
        also_view_eval_files = \
            [os.path.join(FLAGS.data_dir, path) for path in predict_files if "also_view-pred" in path]

        tf.logging.info("***** Running prediction*****")

        tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

        predict_input_fn = file_based_input_fn_builder(
            user_item_files=user_item_eval_files,
            also_view_files=also_view_eval_files,
            is_training=False,
            batch_size=FLAGS.predict_batch_size)

        result = estimator.predict(input_fn=predict_input_fn)

        user_embedding_file = os.path.join(FLAGS.output_dir, "user_embedding.tsv")
        item_embedding_file = os.path.join(FLAGS.output_dir, "item_embedding.tsv")

        with tf.gfile.GFile(user_embedding_file, "w") as writer_1, tf.gfile.GFile(item_embedding_file, 'w') as writer_2:
            num_written_lines = 0
            tf.logging.info("***** Predict results *****")
            for (i, prediction) in enumerate(result):
                user_embedding = prediction["user_embedding"]
                item_embedding = prediction["item_embedding"]
                output_line_1 = ",".join(
                    str(index)
                    for index in user_embedding) + "\n"
                writer_1.write(output_line_1)

                output_line_2 = ",".join(
                    str(index)
                    for index in item_embedding) + "\n"

                writer_2.write(output_line_2)

                num_written_lines += 1


if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("mcf_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.app.run()
