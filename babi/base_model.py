import itertools
import pandas as pd
import json
import os
import glob
import gc
from copy import deepcopy
import time
from collections import defaultdict,OrderedDict
import logging, math, random
#from pympler import muppy, summary as sumy
from multiprocessing import Process

import numpy as np
import tensorflow as tf

from my.tensorflow import average_gradients
from my.utils import get_pbar, collect_indices, get_indices, logger, lll_types
from babi.read_data import DataSet
from protocols import PROTOCOLS, get_protocol_tensors 
from regularizers import quadratic_regularizer


class BaseRunner(object):
    def __init__(self, params, sess, towers):
        assert isinstance(sess, tf.Session)
        self.sess = sess
        self.params = params
        self.towers = towers
        self.ref_tower = towers[0]
        self.num_towers = len(towers)
        self.placeholders = {}
        self.tensors = {}
        self.saver = None
        self.writer = None
        self.initialized = False
        self.train_ops = {}
        self.write_log = params.write_log.value
        self.init_lr = random.uniform(0.06, 0.6) if params.use_random.value else params.init_lr.value
        print("init_lr : %.3f" % self.init_lr)
        self.Fishers = []
        #self.Fishers_prev = []
        self.lam = params.lam.value 
        self.lll_type = params.lll_type.value
        self.omega_decay = params.omega_decay.value
        try:
            self.omega_decay = float(self.omega_decay)
        except ValueError:
            pass
        self.epsilon = params.epsilon.value 
        self.regularizer_fn = quadratic_regularizer if self.lll_type in lll_types else None
        self.task_ops = [] 
        self.step_ops = []
        if self.lll_type in lll_types:
            self.protocol_tensors = get_protocol_tensors(PROTOCOLS[self.lll_type]) 
        self.protocol_name, self.protocol = PROTOCOLS[self.lll_type](self.omega_decay,self.epsilon) if self.lll_type else (None,None)
        self.test_step_ops = [] 
        self.test_task_ops = [] 
        self.test_step_update = params.test_step_update.value 
        self.test_task_update = params.test_task_update.value 

    def initialize(self):
        params = self.params
        sess = self.sess
        device_type = params.device_type.value
        summaries = []

        global_step = tf.get_variable('global_step', shape=[], dtype='int32',
                                      initializer=tf.constant_initializer(0), trainable=False)
        self.tensors['global_step'] = global_step

        epoch = tf.get_variable('epoch', shape=[], dtype='int32',
                                initializer=tf.constant_initializer(0), trainable=False)
        self.tensors['epoch'] = epoch
        with tf.name_scope("accuracy"):
            var_train_acc = tf.get_variable('train_acc', shape=[], dtype='float32',initializer=tf.constant_initializer(0), trainable=False)
            var_val_acc = tf.get_variable('val_acc', shape=[], dtype='float32',initializer=tf.constant_initializer(0), trainable=False)
            self.tensors['train_acc'] = var_train_acc 
            self.tensors['val_acc'] = var_val_acc 
        summaries.append(tf.summary.scalar('train_acc', var_train_acc))
        summaries.append(tf.summary.scalar('val_acc', var_val_acc))

        learning_rate = tf.placeholder('float32', name='learning_rate')
        summaries.append(tf.summary.scalar("learning_rate", learning_rate))
        self.placeholders['learning_rate'] = learning_rate

        if params.opt.value == 'basic':
            opt = tf.train.GradientDescentOptimizer(learning_rate)
        elif params.opt.value == 'adagrad':
            opt = tf.train.AdagradOptimizer(learning_rate)
        elif params.opt.value == 'adam':
            opt = tf.train.AdamOptimizer()
        elif params.opt.value == 'adadelta':
            opt = tf.train.AdadeltaOptimizer(learning_rate)
        else:
            raise Exception()

        grads_pairs_dict = defaultdict(list)
        grads_pairs_dict = defaultdict(list)
        correct_tensors = []
        loss_tensors = []
        with tf.variable_scope("towers"):
            for device_id, tower in enumerate(self.towers):
                with tf.device("/%s:%d" % (device_type, device_id)), tf.name_scope("%s_%d" % (device_type, device_id)):
                    tower.initialize()
                    self.probs = tower.tensors["probs"]
                    tf.get_variable_scope().reuse_variables()
                    loss_tensor = tower.get_loss_tensor()
                    loss_tensors.append(loss_tensor)
                    correct_tensor = tower.get_correct_tensor()
                    correct_tensors.append(correct_tensor)

                    # key = "all" ==> see babi/model.py#252
                    for key, variables in tower.variables_dict.items():
                        grads_pair = opt.compute_gradients(loss_tensor, var_list=variables)
                        grads_pairs_dict[key].append(grads_pair)

                    self.tensors["variables"] = tower.variables_dict["all"]
                    self.variables_len = len(self.tensors["variables"])

        # init_steps 
        if self.lll_type == "ewc":
            self.Fishers = [np.zeros(v.get_shape().as_list()) for v in tower.variables_dict["all"]]
            self.tensors["lll_counts"] = tf.get_variable('lll_counts', shape=[], dtype='float32',initializer=tf.constant_initializer(0), trainable=False)
        else:
            if self.lll_type in lll_types:
                for t in self.protocol_tensors:
                    #self.tensors[t] = [np.zeros(v.get_shape().as_list()) for v in tower.variables_dict["all"]]
                    self.tensors[t] = self.create_zeros_vars(t)

                if self.lll_type == "si":
                    #self.tensors["delta"] = [np.zeros(v.get_shape().as_list()) for v in tower.variables_dict["all"]]
                    self.tensors["previous_vars"] = self.clone_vars("previous_vars")
                elif self.lll_type == "mas":
                    l2_grads = tf.gradients(tf.norm(tower.tensors["logits"]), self.tensors["variables"])
                    self.tensors["l2_grads"] = l2_grads
                    self.tensors["lll_counts"] = tf.get_variable('lll_counts', shape=[], dtype='float32',initializer=tf.constant_initializer(0), trainable=False)


        with tf.name_scope("gpu_sync"):
            loss_tensor = tf.reduce_mean(tf.stack(loss_tensors), 0, name='loss')
            correct_tensor = tf.concat(correct_tensors, 0, name="correct")
            # Average over the 'tower' dimension.
            with tf.name_scope("average_gradients"):
                grads_pair_dict = {key: average_gradients(grads_pairs)
                                   for key, grads_pairs in grads_pairs_dict.items()}
                if params.max_grad_norm.value:
                    grads_pair_dict = {key: [(tf.clip_by_norm(grad, params.max_grad_norm.value), var)
                                             for grad, var in grads_pair]
                                       for key, grads_pair in grads_pair_dict.items()}

        self.tensors['unreg_grads'] =[t[0] for t in  grads_pair_dict["all"]]
        self.tensors['loss'] = loss_tensor
        self.tensors['correct'] = correct_tensor
        summaries.append(tf.summary.scalar(loss_tensor.op.name, loss_tensor))

        for key, grads_pair in grads_pair_dict.items():
            for grad, var in grads_pair:
                if grad is not None:
                    summaries.append(tf.summary.histogram(var.op.name+'/gradients/'+key, grad))

        for var in tf.trainable_variables():
            summaries.append(tf.summary.histogram(var.op.name, var))

        apply_grads_op_dict = {key: opt.apply_gradients(grads_pair, global_step=global_step)
                               for key, grads_pair in grads_pair_dict.items()}

        self.train_ops = {key: tf.group(apply_grads_op)
                          for key, apply_grads_op in apply_grads_op_dict.items()}

        #saver = tf.train.Saver(tf.all_variables(), max_to_keep=2)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=2)
        self.saver = saver

        summary_op = tf.summary.merge(summaries)
        self.tensors['summary'] = summary_op

        #init_op = tf.initialize_all_variables()
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        if self.write_log:
            self.writer = tf.summary.FileWriter(params.log_dir.value, sess.graph)
        self.initialized = True

        if self.lll_type == 'si':
            self.star()

    def _get_feed_dict(self, batches, mode, **kwargs):
        placeholders = self.placeholders
        learning_rate_ph = placeholders['learning_rate']
        learning_rate = kwargs['learning_rate'] if mode == 'train' else 0.0
        feed_dict = {learning_rate_ph: learning_rate}
        for tower_idx, tower in enumerate(self.towers):
            batch = batches[tower_idx] if tower_idx < len(batches) else None
            cur_feed_dict = tower.get_feed_dict(batch, mode, **kwargs)
            feed_dict.update(cur_feed_dict)
        return feed_dict

    def _train_batches(self, batches, **kwargs):
        sess = self.sess
        tensors = self.tensors
        # get inputs data
        feed_dict = self._get_feed_dict(batches, 'train', **kwargs)
        train_op = self._get_train_op(**kwargs)
        ops = [train_op, tensors['summary'], tensors['global_step']]
        train, summary, global_step = sess.run(ops, feed_dict=feed_dict)
        sess.run(self.step_ops, feed_dict=feed_dict)
        #print(self.tensors["previous_vars"][0].eval(sess))
        #print("-"*10)
        #print(self.tensors["variables"][0].eval(sess))
        #print("-"*10)
        #print(self.tensors["delta"][0].eval(sess))
        #print("="*10)
        if self.lll_type == 'si':
            update_previous_vars_ops = self.update_previous_vars()
            #print(self.tensors["previous_vars"][0].eval(sess))
            #print("-"*10)
            #print(self.tensors["variables"][0].eval(sess))
            sess.run(update_previous_vars_ops,feed_dict=feed_dict)
            #print("-"*10)
            #print(self.tensors["previous_vars"][0].eval(sess))
            #print("-"*10)
            #print(self.tensors["delta"][0].eval(sess))
            #print("="*10)
        return train, summary, global_step

    def _eval_batches(self, batches, eval_tensor_names=(), **eval_kwargs):
        sess = self.sess
        tensors = self.tensors
        num_examples = sum(len(batch[0]) for batch in batches)
        feed_dict = self._get_feed_dict(batches, 'eval', **eval_kwargs)
        ops = [tensors[name] for name in ['correct', 'loss', 'summary', 'global_step']]
        correct, loss, summary, global_step = sess.run(ops, feed_dict=feed_dict)
        num_corrects = np.sum(correct[:num_examples])
        if len(eval_tensor_names) > 0:
            valuess = [sess.run([tower.tensors[name] for name in eval_tensor_names], feed_dict=feed_dict)
                       for tower in self.towers]
        else:
            valuess = [[]]

        return (num_corrects, loss, summary, global_step), valuess

    def compute_Fishers(self,batches,train_args): 
        probs = self.probs
        sess = self.sess
        params = self.params
        #class_inds_ = tf.to_int32(tf.multinomial(tf.log(probs), 1))
        # class_inds_:  Tensor("Cast:0", shape=(32, 1), dtype=int32, device=/device:CPU:0) 
        class_inds_ = tf.cast(tf.random.categorical(tf.log(probs), 1),tf.int32)
        class_inds = []
        for i in range(class_inds_.shape[0]): 
            class_inds.append([i,class_inds_[i][0]])
        try:
            feed_dict = self._get_feed_dict(batches, 'train', **train_args)
        except KeyError:
            feed_dict = self._get_feed_dict(batches, 'eval', **train_args)
        # gn:  Tensor("GatherNd:0", shape=(32,), dtype=float32, device=/device:CPU:0)
        gn = tf.gather_nd(probs,class_inds)
        der = tf.gradients(gn, self.tensors["variables"])
        ders = sess.run(der, feed_dict=feed_dict)
        # der0: dense_shape=array([211,  50], dtype=int32)
        der0 = ders.pop(0)
        indices_dict = collect_indices(der0,params.vocab_size.value,params.hidden_size.value)
        ind_ave = get_indices(indices_dict,params.rnn_grad_strategy.value)
        ders.insert(0,ind_ave)
        Fishers = [] 
        for d in ders:
            #Fishers[v] += np.square(ders[v])
            Fishers.append(d)
        np.save(os.path.join("tmp","%s.npy"%(str(int(time.time())))),Fishers)
        sess.run(tf.assign_add(self.tensors["lll_counts"],1))

        #self.tensors["lll_counts"] += 1*params.batch_size.value
        #del class_inds 
        #del class_inds_ 
        #del Fishers
        #del indices_dict
        #del ind_ave 
        #del der
        #del der0 
        #del ders 
        #del feed_dict 
        #del gn 
        #collected = gc.collect()
        #print ("Garbage collector: collected %d objects." % (collected))

    def train(self, train_data_set, num_epochs, \
              val_data_set=None, eval_ph_names=(),\
              eval_tensor_names=(), num_batches=None,\
              val_num_batches=None):
        for f in glob.glob("tmp/*"):
            os.remove(f)
        assert isinstance(train_data_set, DataSet)
        assert self.initialized, "Initialize tower before training."

        sess = self.sess
        writer = self.writer
        params = self.params
        progress = params.progress.value
        protocol = self.protocol
        # step updates
        if protocol and "step_updates" in protocol:
            # make sure doing lll
            assert self.lll_type in lll_types 
            step_updates = OrderedDict(protocol["step_updates"])
            for name, func in step_updates.items():
                #step_op = list(map(lambda v,pv: func(self.tensors,v,pv),range(self.variables_len),self.tensors[name]))
                for i in range(self.variables_len):  
                    step_op = func(self.tensors,i,self.tensors[name][i])
                    step_op = tf.assign(self.tensors[name][i],step_op) 
                    self.step_ops.append(step_op)

        # task updates
        if protocol and "task_updates" in protocol:
            # make sure doing lll
            assert self.lll_type in lll_types 
            task_updates = OrderedDict(protocol["task_updates"])
            for name, func in task_updates.items():
                #task_op = list(map(lambda v,pv: func(self.tensors,v,pv),range(self.variables_len),self.tensors[name]))
                for i in range(self.variables_len):  
                    task_op = func(self.tensors,i,self.tensors[name][i])
                    task_op = tf.assign(self.tensors[name][i],task_op) 
                    self.task_ops.append(task_op)

        val_acc = None
        # if num batches is specified, then train only that many
        num_batches = num_batches or train_data_set.get_num_batches(partial=False)
        num_iters_per_epoch = int(num_batches / self.num_towers)
        num_digits = int(np.log10(num_batches))

        epoch_op = self.tensors['epoch']
        epoch = sess.run(epoch_op)
        logging.info("num iters per epoch: %d" % num_iters_per_epoch)
        logging.info("starting from epoch %d." % (epoch+1))
        best_global_step = self.tensors['global_step']
        best_val_acc = 0.0
        best_val_loss = 99999

        train_accs, val_accs = [],[]
        while epoch < num_epochs:
            train_args = self._get_train_args(epoch)
            if progress:
                pbar = get_pbar(num_iters_per_epoch, "epoch %s|" % str(epoch+1).zfill(num_digits)).start()
            # num_iters_per_epoch = int(num_batches / self.num_towers)
            print("epoch: ",epoch,'  ', num_epochs)
            for iter_idx in range(num_iters_per_epoch):
                #print("iter_idx: ",iter_idx+1)
                batches = [train_data_set.get_next_labeled_batch() for _ in range(self.num_towers)]
                _, summary, global_step = self._train_batches(batches, **train_args)
                if self.write_log:
                    writer.add_summary(summary, global_step)
                if progress:
                    pbar.update(iter_idx)

                # compute Fisher
                #if iter_idx % 10 == 0 and iter_idx !=0:
                #    self.compute_Fishers(batches,train_args)
                #    #p = Process(target=self.compute_Fishers, args = (batches,train_args))
                #    #p.start()
                #    #p.join()
                #    self.tensors["lll_counts"] += 1 

            #collected = gc.collect()
            #print ("Garbage collector: collected %d objects." % (collected))
            #print("%"*30)
            #all_objects = muppy.get_objects()
            #lists = [ao for ao in all_objects if isinstance(ao, list)]
            #tuples = [ao for ao in all_objects if isinstance(ao, tuple)]
            #sum1 = sumy.summarize(all_objects)
            #sumy.print_(sum1)
            #print("len of list: ",len(lists))
            #print("len of tuple: ",len(tuples))
            #del all_objects
            #del lists
            #del sum1
            #print("%"*30)
            #print ("gc.garbage: " % (gc.garbage))


            if progress:
                pbar.finish()
            train_data_set.complete_epoch()

            assign_op = epoch_op.assign_add(1)
            _, epoch = sess.run([assign_op, epoch_op])

            global_step = sess.run(self.tensors['global_step'])
            if val_data_set and epoch % params.val_period.value == 0:
                _, train_acc = self.eval(train_data_set, eval_tensor_names=eval_tensor_names, num_batches=val_num_batches)
                val_loss, val_acc = self.eval(val_data_set, eval_tensor_names=eval_tensor_names, num_batches=val_num_batches)
                sess.run(self.tensors['train_acc'].assign(train_acc))
                sess.run(self.tensors['val_acc'].assign(val_acc))
                train_accs.append({'epoch':epoch,'global_step':global_step,'acc':train_acc})
                val_accs.append({'epoch':epoch,'global_step':global_step,'acc':val_acc})
	        
                if val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss):
                    count = 0
                    best_val_acc = val_acc
                    best_val_loss = val_loss
                    best_global_step = global_step
                    self.save()

                elif val_loss < best_val_loss:
                    count = 0
                else:
                    count += 1
                    if count >= int(params.break_at.value):
                        break

        # update Fishers information
        #for v in range(len(self.Fishers)):
        #    #self.Fishers[v] /= (num_iters_per_epoch*num_epochs)
        #    #self.Fishers[v] /= num_epochs
        #    self.Fishers[v] /= ewc_counts 
        #    if len(self.Fishers_prev) > 0:
        #        self.Fishers[v] += self.Fishers_prev[v] 

        if self.lll_type == "ewc":
            #self.update_fishers_at_the_end(val_data_set, num_batches=val_num_batches)
            self.update_fishers_at_the_end(train_data_set, num_batches=10)
            # update Fishers information
            for f in glob.glob("tmp/*"):
                fisher = list(np.load(f))
                for v in range(len(self.Fishers)):
                    self.Fishers[v] += np.square(fisher[v])
                    #self.Fishers[v] += fisher[v]
            for v in range(len(self.Fishers)):
                self.Fishers[v] /= self.tensors["lll_counts"] 
            # plus Fishers_prev
            #if len(self.Fishers_prev) > 0:
            #    logger.info("len(self.Fishers_prev) > 0 ==> task(%s) train?%s"%(self.params.task.value,self.params.train.value))
            #    for v in range(len(self.Fishers)):
            #        self.Fishers[v] += self.Fishers_prev[v] 
            np.save(os.path.join("previous_tensors","omega.npy"),self.Fishers)
            #self.tensors["lll_counts"] = 0

        elif self.lll_type in lll_types:
            sess.run(self.task_ops)
            omega_vals = [t.eval(sess) for t in self.tensors["omega"]]
            np.save(os.path.join("previous_tensors","omega.npy"),omega_vals)


        if not best_global_step == global_step:
            save_dir = self.params.save_dir.value
            name = self.params.model_name.value
            save_path = os.path.join(save_dir, name)
            print("save_path: ",save_path)
            print("best_global_step: ",best_global_step)
            self.saver.restore(sess, '%s-%s'%(save_path,best_global_step))
        if self.write_log:
            df = pd.DataFrame(train_accs)
            df = df[["epoch","global_step","acc"]]
            df.to_csv(os.path.join(self.params.log_dir.value,"train_acc.csv"),index=False)
            df = pd.DataFrame(val_accs)
            df = df[["epoch","global_step","acc"]]
            df.to_csv(os.path.join(self.params.log_dir.value,"val_acc.csv"),index=False)

        return best_val_loss, best_val_acc

    def test(self, data_set, eval_tensor_names=(), eval_ph_names=(), num_batches=None):
        # test step updates
        if self.protocol and "test_step_updates" in self.protocol:
            # make sure doing lll
            assert self.lll_type in lll_types 
            #self.test_step_update = True
            self.test_step_ops = []
            test_step_updates = OrderedDict(self.protocol["test_step_updates"])
            for name, func in test_step_updates.items():
                for i in range(self.variables_len):  
                    step_op = func(self.tensors,i,self.tensors[name][i])
                    step_op = tf.assign(self.tensors[name][i],step_op) 
                    self.test_step_ops.append(step_op)

        # test task updates
        if self.protocol and "test_task_updates" in self.protocol:
            # make sure doing lll
            assert self.lll_type in lll_types 
            #self.test_task_update = True
            self.test_task_ops = []
            test_task_updates = OrderedDict(self.protocol["test_task_updates"])
            for name, func in test_task_updates.items():
                #task_op = list(map(lambda v,pv: func(self.tensors,v,pv),range(self.variables_len),self.tensors[name]))
                for i in range(self.variables_len):  
                    task_op = func(self.tensors,i,self.tensors[name][i])
                    task_op = tf.assign(self.tensors[name][i],task_op) 
                    self.test_task_ops.append(task_op)
        return self.eval(data_set, eval_tensor_names, eval_ph_names, num_batches)
        

    def eval(self, data_set, eval_tensor_names=(), eval_ph_names=(), num_batches=None):
        # TODO : eval_ph_names
        assert isinstance(data_set, DataSet)
        assert self.initialized, "Initialize tower before training."
#
        params = self.params
        sess = self.sess
        epoch_op = self.tensors['epoch']
        epoch = sess.run(epoch_op)
        progress = params.progress.value
        num_batches = num_batches or data_set.get_num_batches(partial=True)
        num_iters = int(np.ceil(num_batches / self.num_towers))
        num_corrects, total, total_loss = 0, 0, 0.0
        eval_values = []
        idxs = []
        N = data_set.batch_size * num_batches
        if N > data_set.num_examples:
            N = data_set.num_examples
        eval_args = self._get_eval_args(epoch)
        string = "eval on %s, N=%d|" % (data_set.name, N)
        if progress:
            pbar = get_pbar(num_iters, prefix=string).start()
        for iter_idx in range(num_iters):
            batches = []
            for _ in range(self.num_towers):
                if data_set.has_next_batch(partial=True):
                    idxs.extend(data_set.get_batch_idxs(partial=True))
                    batches.append(data_set.get_next_labeled_batch(partial=True))
            (cur_num_corrects, cur_avg_loss, _, global_step), eval_value_batches = \
                self._eval_batches(batches, eval_tensor_names=eval_tensor_names, **eval_args)
            num_corrects += cur_num_corrects
            cur_num = sum(len(batch[0]) for batch in batches)
            total += cur_num
            for eval_value_batch in eval_value_batches:
                eval_values.append([x.tolist() for x in eval_value_batch])  # numpy.array.toList
            total_loss += cur_avg_loss * cur_num
            if progress:
                pbar.update(iter_idx)

            if self.test_step_update and len(self.test_step_ops) > 0:
                feed_dict = self._get_feed_dict(batches, 'eval', **eval_args)
                sess.run(self.test_step_ops,feed_dict=feed_dict)
                sess.run(tf.assign_add(self.tensors["lll_counts"],1))
            #self.compute_Fishers(batches,eval_args)

        if progress:
            pbar.finish()
        loss = float(total_loss) / total
        data_set.reset()

        acc = float(num_corrects) / total
        print("%s at epoch %d: acc = %.2f%% = %d / %d, loss = %.4f" %
              (data_set.name, epoch, 100 * acc, num_corrects, total, loss))

        # For outputting eval json files
        if len(eval_tensor_names) > 0:
            ids = [data_set.idx2id[idx] for idx in idxs]
            zipped_eval_values = [list(itertools.chain(*each)) for each in zip(*eval_values)]
            values = {name: values for name, values in zip(eval_tensor_names, zipped_eval_values)}
            out = {'ids': ids, 'values': values}
            eval_path = os.path.join(params.eval_dir.value, "%s_%s.json" % (data_set.name, str(epoch).zfill(4)))
            json.dump(out, open(eval_path, 'w'))

        if self.test_task_update and len(self.test_task_ops) > 0:
            sess.run(self.test_task_ops)
            omega_vals = [t.eval(sess) for t in self.tensors["omega"]]
            np.save(os.path.join("previous_tensors","omega.npy"),omega_vals)

        return loss, acc

    def update_fishers_at_the_end(self, data_set, num_batches=None):
        # TODO : eval_ph_names
        assert isinstance(data_set, DataSet)
#
        params = self.params
        sess = self.sess
        num_batches = num_batches or data_set.get_num_batches(partial=True)
        num_iters = int(np.ceil(num_batches / self.num_towers))
        N = data_set.batch_size * num_batches
        if N > data_set.num_examples:
            N = data_set.num_examples
        eval_args = self._get_eval_args(0)
        for iter_idx in range(num_iters):
            batches = []
            for _ in range(self.num_towers):
                if data_set.has_next_batch(partial=True):
                    batches.append(data_set.get_next_labeled_batch(partial=True))

            self.compute_Fishers(batches,eval_args)

    def _get_train_op(self, **kwargs):
        return self.train_ops['all']

    def _get_train_args(self, epoch_idx):
        params = self.params
        learning_rate = self.init_lr

        anneal_period = params.lr_anneal_period.value
        anneal_ratio = params.lr_anneal_ratio.value
        num_periods = int(epoch_idx / anneal_period)
        factor = anneal_ratio ** num_periods
        learning_rate *= factor

        train_args = self._get_common_args(epoch_idx)
        train_args['learning_rate'] = learning_rate
        return train_args

    def _get_eval_args(self, epoch_idx):
        return self._get_common_args(epoch_idx)

    def _get_common_args(self, epoch_idx):
        return {}

    def star(self):
        self.tensors["star_vars"] = self.eval_current_vars()
        if self.lll_type == "si":
            self.update_previous_vars()

    def eval_current_vars(self):
        current_vars = []
        for v in range(self.variables_len):
            current_vars.append(self.tensors["variables"][v].eval(session=self.sess))
        return current_vars

    def create_zeros_vars(self,name):
        zeros = []
        for i,v in enumerate(self.tensors["variables"]):
            with tf.variable_scope(name) as scope:
                var = tf.get_variable(name="%s"%(i),initializer=np.zeros(v.get_shape().as_list(),dtype=np.float32))
            zeros.append(var)
        return zeros 

    def clone_vars(self,name):
        clones = []
        for i,v in enumerate(self.tensors["variables"]):
            with tf.variable_scope(name) as scope:
                var = tf.get_variable(name="%s"%(i),initializer=v.initialized_value())
            clones.append(var)
        return clones 

    def update_previous_vars(self):
        ops = []
        for i in range(self.variables_len):
            ops.append(tf.assign(self.tensors["previous_vars"][i],self.tensors["variables"][i]))
        return ops

    def update_loss(self, lam):
        # put log ==> lam
        logger.info("lam: %s ==> task(%s) train?%s"%(lam,self.params.task.value,self.params.train.value))
        regularizer = 0.0 if self.regularizer_fn is None else self.regularizer_fn(self.tensors["variables"], self.tensors, norm=2)
        if lam > 0:
            self.tensors["loss"] += tf.cast((lam/2) * regularizer, tf.float32)
            #for v in range(self.variables_len):
            #    self.tensors['loss'] += (lam/2) * tf.reduce_sum(tf.multiply(self.Fishers_prev[v].astype(np.float32),tf.square(self.tensors["variables"][v] - self.tensors["star_vars"][v])))

    def save(self):
        assert self.initialized, "Initialize tower before saving."

        sess = self.sess
        params = self.params
        save_dir = params.save_dir.value
        #name ==> save_dir/${name}-*.index, ... 
        name = params.model_name.value
        global_step = self.tensors['global_step']
        logging.info("saving model ...")
        save_path = os.path.join(save_dir, name)
        self.saver.save(sess, save_path, global_step)
        logging.info("saving done.")

    def load(self):
        assert self.initialized, "Initialize tower before loading."

        sess = self.sess
        params = self.params
        save_dir = params.save_dir.value
        logging.info("loading model ...")
        checkpoint = tf.train.get_checkpoint_state(save_dir)
        print("loading model ... ==>",checkpoint)
        assert checkpoint is not None, "Cannot load checkpoint at %s" % save_dir
        print("llltype: ",self.lll_type)
        self.saver.restore(sess, checkpoint.model_checkpoint_path)
        if checkpoint and self.lll_type in lll_types:
            self.star()
            omegas = list(np.load(os.path.join("previous_tensors","omega.npy")))
            if self.lll_type == "ewc":
                self.tensors["omega"] = omegas 
            else:
                omega_ops = []
                for i in range(self.variables_len):

                    op = tf.assign(self.tensors["omega"][i],omegas[i])
                    omega_ops.append(op)
                sess.run(omega_ops)

            print("===== self.update_loss =====")
            print("lam: ",self.lam)
            self.tensors['init_loss'] = self.tensors['loss']  
            self.update_loss(self.lam)
        if params.reset_epochs.value:
            epoch_op = self.tensors['epoch']
            global_step_op = self.tensors['global_step']
            sess.run([epoch_op.assign(0),global_step_op.assign(0)])
        logging.info("loading done.")


class BaseTower(object):
    def __init__(self, params):
        self.params = params
        self.placeholders = {}
        self.tensors = {}
        self.variables_dict = {}
        # this initializer is used for weight init that shouldn't be dependent on input size.
        # for MLP weights, the tensorflow default initializer should be used,
        # i.e. uniform unit scaling initializer
        self.initializer = tf.truncated_normal_initializer(params.init_mean.value, params.init_std.value)

    def initialize(self):
        # Actual building
        # Separated so that GPU assignment can be done here.
        raise Exception("Implement this!")

    def get_correct_tensor(self):
        return self.tensors['correct']

    def get_loss_tensor(self):
        return self.tensors['loss']

    def get_init_loss_tensor(self):
        return self.tensors['init_loss']

    def get_variables_dict(self):
        return self.variables_dict

    def get_feed_dict(self, batch, mode, **kwargs):
        # TODO : MUST handle batch = None
        raise Exception("Implment this!")
