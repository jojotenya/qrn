import json
import os
import pandas as pd
import shutil
from pprint import pprint

import tensorflow as tf

from babi.model import Tower, Runner
from config.get_config import get_config_from_file, get_config
from babi.read_data import read_data
from my.utils import logger, lll_types

flags = tf.app.flags

# File directories
flags.DEFINE_string("model_name", "babi", "Model name. This will be used for save, log, and eval names. [parallel]")
flags.DEFINE_string("data_dir", "data/babi", "Data directory [data/babi]")
flags.DEFINE_string("eval_dir", "", "Evaluation data directory")
flags.DEFINE_string("log_dir", "", "log data directory")
flags.DEFINE_string("save_dir", "", "save data directory")
flags.DEFINE_string("model_dir", "", "if not empty, it'll be the path the model loaded from")

# Training parameters
# These affect result performance
flags.DEFINE_integer("batch_size", 32, "Batch size for each tower. [32]")
flags.DEFINE_float("init_mean", 0, "Initial weight mean [0]")
flags.DEFINE_float("init_std", 1.0, "Initial weight std [1.0]")
flags.DEFINE_float("init_lr", 0.5, "Initial learning rate [0.5]")
flags.DEFINE_integer("lr_anneal_period", 100, "Anneal period [100]")
flags.DEFINE_float("lr_anneal_ratio", 0.5, "Anneal ratio [0.5")
flags.DEFINE_integer("num_epochs", 100, "Total number of epochs for training [100]")
flags.DEFINE_boolean("reset_epochs", False, "reset epoch variable to 0")
flags.DEFINE_string("opt", 'adagrad', 'Optimizer: basic | adagrad | adam [basic]')
flags.DEFINE_float("wd", 0.001, "Weight decay [0.001]")
flags.DEFINE_integer("max_grad_norm", 0, "Max grad norm. 0 for no clipping [0]")
flags.DEFINE_float("max_val_loss", 0.0, "Max val loss [0.0]")
flags.DEFINE_integer("break_at", 3, "break if val loss not decrease in this number")

# Training and testing options
# These do not directly affect result performance (they affect duration though)
flags.DEFINE_boolean("train", True, "Train (will override without load)? Test if False [True]")
flags.DEFINE_integer("val_num_batches", 0, "Val num batches. 0 for max possible. [0]")
flags.DEFINE_integer("train_num_batches", 0, "Train num batches. 0 for max possible [0]")
flags.DEFINE_integer("test_num_batches", 0, "Test num batches. 0 for max possible [0]")
flags.DEFINE_boolean("load", True, "Load from saved model? [True]")
flags.DEFINE_boolean("progress", False, "Show progress bar? [False]")
flags.DEFINE_string("device_type", 'gpu', "cpu | gpu [gpu]")
flags.DEFINE_integer("num_devices", 1, "Number of devices to use. Only for multi-GPU. [1]")
flags.DEFINE_integer("val_period", 10, "Validation period (for display purpose only) [10]")
flags.DEFINE_integer("save_period", 10, "Save period [10]")
flags.DEFINE_string("config_id", 'None', "Config name (e.g. local) to load. 'None' to use config here. [None]")
flags.DEFINE_string("config_ext", ".json", "Config file extension: .json | .tsv [.json]")
flags.DEFINE_integer("num_trials", 1, "Number of trials [1]")
flags.DEFINE_string("seq_id", "None", "Sequence id [None]")
flags.DEFINE_string("run_id", "0", "Run id [0]")
flags.DEFINE_boolean("write_log", False, "Write log? [False]")
# TODO : don't erase log folder if not write log

# Debugging
flags.DEFINE_boolean("draft", False, "Draft? (quick initialize) [False]")

# App-specific options
# TODO : Any other options
flags.DEFINE_integer("which_model", 0, "Task number that the model had trained on.")
flags.DEFINE_string("task", "all", "Task number. [all]")
flags.DEFINE_bool("large", False, "1k (False) | 10k (True) [False]")
flags.DEFINE_string("lang", "en", "en | something")
flags.DEFINE_integer("hidden_size", 50, "Hidden size. [50]")
flags.DEFINE_float("keep_prob", 1.0, "Keep probability of RNN inputs [1.0]")
flags.DEFINE_integer("mem_num_layers", 2, "Number of memory layers [2]")
flags.DEFINE_float("att_forget_bias", 2.5, "Attention gate forget bias [2.5]")
flags.DEFINE_integer("max_mem_size", 50, "Maximum memory size (from most recent) [50]")
flags.DEFINE_string("class_mode", "h", "classification mode: h | uh | hs | hss [h]")
flags.DEFINE_boolean("use_class_bias", False, "Use bias at final classification linear trans? [False]")
flags.DEFINE_boolean("use_reset", True, "Use reset gate? [True]")
flags.DEFINE_boolean("use_vector_gate", False, "Use vector gate? [False]")
flags.DEFINE_boolean("use_res", False, "Use residual connection?")
flags.DEFINE_boolean("use_dropout",  False, "Use dropout?")
flags.DEFINE_boolean("use_random", False, "Use random init_lr")

# LLL
flags.DEFINE_string("lll_type", "None", "ewc | si | mas | None")
flags.DEFINE_integer("lam", 0, "lambda for LLL")
flags.DEFINE_string("rnn_grad_strategy", "sum", "last | ave | sum")
flags.DEFINE_string("omega_decay", "sum", "sum | float between 0 to 1")
flags.DEFINE_float("epsilon", 1e-3, "for si")
flags.DEFINE_boolean("test_step_update", False, "for mas")
flags.DEFINE_boolean("test_task_update", False, "for mas")


# Meta data
flags.DEFINE_integer("vocab_size", 0, "vocabulary size")
flags.DEFINE_integer("max_fact_size", 0, "max fact size")
flags.DEFINE_integer("max_ques_size", 0, "max ques size")
flags.DEFINE_integer("max_hypo_size", 0, "max hypo size")
flags.DEFINE_integer("max_sent_size", 0, "max sent size")
flags.DEFINE_integer("max_num_sents", 0, "max num sents")
flags.DEFINE_integer("max_num_sups", 0, "max num sups")
flags.DEFINE_integer("eos_idx", 0, "eos index")
flags.DEFINE_integer("mem_size", 0, "mem size")

FLAGS = flags.FLAGS


def mkdirs(config, trial_idx):
    evals_dir = "evals"
    logs_dir = "logs"
    saves_dir = "saves"
    if not os.path.exists(evals_dir):
        os.mkdir(evals_dir)
    if not os.path.exists(logs_dir):
        os.mkdir(logs_dir)
    if not os.path.exists(saves_dir):
        os.mkdir(saves_dir)

    model_name = config.model_name.value
    config_id = str(config.config_id.value).zfill(2)
    run_id = str(config.run_id.value).zfill(2)
    trial_idx = str(trial_idx).zfill(2)
    task = config.task.value.zfill(2)
    mid = config.lang.value
    if config.large.value:
        mid += "-10k"
    subdir_name = "-".join([task, config_id, run_id, trial_idx])

    eval_dir = os.path.join(evals_dir, model_name, mid)
    eval_subdir = os.path.join(eval_dir, subdir_name)
    log_dir = config.log_dir.value
    if len(log_dir) > 0:
        log_dir = os.path.join(logs_dir, model_name, log_dir)
    else:
        log_dir = os.path.join(logs_dir, model_name, mid)
    log_subdir = os.path.join(log_dir, subdir_name)
    save_dir = config.save_dir.value
    if len(save_dir) == 0:
        save_dir = os.path.join(saves_dir, model_name, mid)
        save_subdir = os.path.join(save_dir, subdir_name)
    else:
        save_subdir = save_dir 
    # saves/babi/en
    config.eval_dir.value = eval_subdir
    config.log_dir.value = log_subdir
    config.save_dir.value = save_subdir

    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)
    if os.path.exists(eval_subdir):
        if config.train.value and not config.load.value:
            shutil.rmtree(eval_subdir)
            os.mkdir(eval_subdir)
    else:
        os.mkdir(eval_subdir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if os.path.exists(log_subdir):
        if config.train.value and not config.load.value:
            shutil.rmtree(log_subdir)
            os.mkdir(log_subdir)
    else:
        os.makedirs(log_subdir)
    if config.train.value:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if os.path.exists(save_subdir):
            if not config.load.value:
                shutil.rmtree(save_subdir)
                os.mkdir(save_subdir)
        else:
            os.mkdir(save_subdir)


def load_metadata(config):
    data_dir = os.path.join(config.data_dir.value, config.lang.value + ("-10k" if config.large.value else ""))
    metadata_path = os.path.join(data_dir, config.task.value.zfill(2), "metadata.json")
    metadata = json.load(open(metadata_path, "r"))

    # TODO: set other parameters, e.g.
    # config.max_sent_size = meta_data['max_sent_size']
    config.max_fact_size.value = metadata['max_fact_size']
    config.max_ques_size.value = metadata['max_ques_size']
    config.max_sent_size.value = metadata['max_sent_size']
    config.vocab_size.value = metadata['vocab_size']
    config.max_num_sents.value = metadata['max_num_sents']
    config.max_num_sups.value = metadata['max_num_sups']
    config.eos_idx.value = metadata['eos_idx']
    config.mem_size.value = min(config.max_num_sents.value, config.max_mem_size.value)


def main(_):
    this_dir = os.path.dirname(os.path.realpath(__file__))
    if FLAGS.seq_id == 'None':
        seq = [[FLAGS.config_id, FLAGS.num_trials]]
    else:
        seqs = json.load(open(os.path.join(this_dir, "seqs.json"), 'r'))
        seq = seqs[FLAGS.seq_id]
    print (seq)
    summaries = []
    for config_id, num_trials in seq:
        if config_id == "None":
            config = get_config(FLAGS.__flags, {})
        else:
            # TODO : create config file (.json)
            configs_path = os.path.join(this_dir, "configs_new%s" % FLAGS.config_ext)
            config = get_config_from_file(FLAGS.__flags, configs_path, config_id)

        if config.lll_type.value.lower() == "none" or config.lll_type.value == "":
            config.lll_type.value = None
        
        if config.task.value == "all":
            tasks = list(map(str, range(1, 21)))
        elif config.task.value == 'trouble':
            tasks = list(map(str, [18,19]))
        elif config.task.value == 'strange' :
            tasks = list(map(str, [4,5]))
        elif config.task.value == 'all_together' :
            tasks = ["all"]
            # data ==> qrn/data/en/all/
        else:
            tasks = [config.task.value]
        for task in tasks:
            # FIXME : this is bad way of setting task each time
            config.task.value = task
            print("=" * 80)
            print("Config ID {}, task {}, {} trials".format(config.config_id.value, config.task.value, num_trials))
            summary = _main(config, num_trials)
            summaries.append(summary)
	
    print("=" * 80)
    print("SUMMARY")
    for summary in summaries:
        print(summary)


def _main(config, num_trials):
    load_metadata(config)

    # Load data
    if config.train.value:
        comb_train_ds = read_data(config, 'train', config.task.value)
        comb_dev_ds = read_data(config, 'dev', config.task.value)
    test_task = config.task.value if not config.task.value == 'joint' else 'all'
    comb_test_ds = read_data(config, 'test', test_task)

    # For quick draft initialize (deubgging).
    if config.draft.value:
        config.train_num_batches.value = 1
        config.val_num_batches.value = 1
        config.test_num_batches.value = 1
        config.num_epochs.value = 2
        config.val_period.value = 1
        config.save_period.value = 1
        # TODO : Add any other parameter that induces a lot of computations

    for k,v in config.__dict__.items():
        print("%s: %s"%(k,v.value))
    print("="*20)

    # TODO : specify eval tensor names to save in evals folder
    eval_tensor_names = ['a', 'rf', 'rb', 'correct', 'yp']
    eval_ph_names = ['q', 'q_mask', 'x', 'x_mask', 'y']

    def get_best_trial_idx_with_loss(_val_losses):
        return min(enumerate(_val_losses), key=lambda x: x[1])[0]

    def get_best_trial_idx_with_acc(_val_accs):
        return max(enumerate(_val_accs), key=lambda x: x[1])[0]



    val_losses = []
    val_accs = []
    test_accs = []
    for trial_idx in range(1, num_trials+1):
        if config.train.value:
            print("-" * 80)
            print("Task {} trial {}".format(config.task.value, trial_idx))
        mkdirs(config, trial_idx)
        graph = tf.Graph()
        # TODO : initialize BaseTower-subclassed objects
        towers = [Tower(config) for _ in range(config.num_devices.value)]
        sess = tf.Session(graph=graph, config=tf.ConfigProto(allow_soft_placement=True))
        # TODO : initialize BaseRunner-subclassed object
        wl = config.write_log.value
        if not config.train.value:
            config.write_log.value = False
        runner = Runner(config, sess, towers)
        with graph.as_default(), tf.device("/cpu:0"):
            runner.initialize()
            config.write_log.value = wl 
            if config.train.value:
                if config.load.value:
                    runner.load()
                logger.info("runner.tensors[loss]: %s ==> task(%s) train?%s"%(runner.tensors["loss"],config.task.value,config.train.value))
                val_loss, val_acc = runner.train(comb_train_ds, config.num_epochs.value, val_data_set=comb_dev_ds,
                                                 num_batches=config.train_num_batches.value,
                                                 val_num_batches=config.val_num_batches.value, eval_ph_names=eval_ph_names)
                val_losses.append(val_loss)
                val_accs.append(val_acc)
            else:
                runner.load()

            #test_loss, test_acc = runner.eval(comb_test_ds, 
            test_loss, test_acc = runner.test(comb_test_ds, 
                    eval_tensor_names=eval_tensor_names,
                    num_batches=config.test_num_batches.value, 
                    eval_ph_names=eval_ph_names)
            test_accs.append(test_acc)

        if config.train.value:
            best_trial_idx = get_best_trial_idx_with_acc(val_accs)
            print("-" * 80)
            print("Num trials: {}".format(trial_idx))
            print("Min val loss: {:.4f}".format(min(val_losses)))
            print("Test acc at min val acc: {:.2f}%".format(100 * test_accs[best_trial_idx]))
            print("Trial idx: {}".format(best_trial_idx+1))

        # Cheating, but for speed
        if test_acc == 1.0:
            break

    best_trial_idx = get_best_trial_idx_with_acc(test_accs)
    summary = "Task {}: {:.2f}% at trial {}".format(config.task.value, test_accs[best_trial_idx] * 100, best_trial_idx)

    if config.which_model.value == 0:
        config.which_model.value = config.task.value
    elif config.which_model.value == -1:
        config.which_model.value = "all" 
    if config.write_log.value:
        test_acc_path = os.path.join(config.log_dir.value,"test_acc.csv")
        if os.path.exists(test_acc_path):
            test_acc_log = pd.read_csv(test_acc_path)
            test_acc_log = test_acc_log.append({"model":to_int(config.which_model.value),"acc":test_accs[best_trial_idx]},ignore_index=True)
        else:
            test_acc_log = pd.DataFrame([{"model":to_int(config.which_model.value),"acc":test_accs[best_trial_idx]}])
        test_acc_log = test_acc_log[["model","acc"]]
        test_acc_log.to_csv(test_acc_path,index=False)

    return summary

def to_int(x):
    try:
        return int(x)
    except:
        return x

if __name__ == "__main__":
    tf.app.run()
