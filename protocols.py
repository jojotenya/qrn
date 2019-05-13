import tensorflow as tf
from my.utils import ema
from regularizers import quadratic_regularizer

# var = weight
SI_PROTOCOL = lambda omega_decay,epsilon: (
    "si[omega_decay=%s,epsilon=%s]"%(omega_decay,epsilon),
    {
        "init_updates": [
            ("star_vars", lambda tensors, var, prev_val:\
                    var.value()
            ),
        ],
        "step_updates": [
            ("delta", lambda tensors, var, prev_val:\
                tensors["variables"][var] - tensors["previous_vars"][var]
            ),
            ("loss_accum", lambda tensors, var, prev_val:\
                prev_val - tensors["unreg_grads"][var]*tensors["delta"][var]
            ),
            #("previous_vars", lambda tensors, var, prev_val:\
            #    tensors["variables"][var]
            #),
        ],
        "task_updates": [
            ("omega", lambda tensors, var, prev_val:\
                tf.nn.relu(\
                  ema(omega_decay,\
                      prev_val,\
                      tensors["loss_accum"][var]/((tensors["star_vars"][var]-tensors["variables"][var])**2+epsilon)
                  ) 
                ) 
            ),
            #("loss_accum", lambda tensors, var, prev_val:\
            #    prev_val*0.0
            #),
        ],
        "regularizer_fn": quadratic_regularizer,
    }
)

EWC_PROTOCOL = lambda omega_decay,epsilon: (
    "si[omega_decay=%s,epsilon=%s]"%(omega_decay,epsilon),
    {
        "init_updates": [
            ("star_vars", lambda tensors, var, prev_val:\
                    var.value()
            ),
        ],
        #"task_updates": [
        #    ("omega", lambda tensors, var, prev_val:\
        #        tf.nn.relu(\
        #          ema(omega_decay,\
        #              prev_val,\
        #              tensors["loss_accum"][var]/((tensors["star_vars"][var]-var.value())**2+epsilon)
        #          ) 
        #        ) 
        #    ),
        #],
        "regularizer_fn": quadratic_regularizer,
    }
)

MAS_PROTOCOL = lambda omega_decay,epsilon: (
    "si[omega_decay=%s,epsilon=%s]"%(omega_decay,epsilon),
    {
        "init_updates": [
            ("star_vars", lambda tensors, var, prev_val:\
                    var.value()
            ),
        ],
        "test_step_updates": [
            ("l2_grads_accum", lambda tensors, var, prev_val:\
                tf.abs(tensors["l2_grads"][var]) + prev_val
            ),
        ],
        "test_task_updates": [
            ("omega", lambda tensors, var, prev_val:\
                #tf.nn.relu(\
                  ema(omega_decay,\
                      prev_val,\
                      tensors["l2_grads_accum"][var]/tensors["lll_counts"]+epsilon
                  ) 
                #) 
            ),
            #("loss_accum", lambda tensors, var, prev_val:\
            #    prev_val*0.0
            #),
        ],
        "regularizer_fn": quadratic_regularizer,
    }
)


PROTOCOLS = {
    "si" : SI_PROTOCOL,
    "ewc" : EWC_PROTOCOL,
    "mas" : MAS_PROTOCOL,
    #"mas" : MAS_PROTOCOL,
}

def get_protocol_tensors(protocol):
    tensors = []
    steps = protocol(None,None)[1]
    updates = []
    for u in ["init_updates","step_updates","task_updates","test_step_updates","test_task_updates"]:
        if u in steps:
            updates.append(u)
    for u in updates:
        for keys in steps[u]:
            name = keys[0]
            if name not in tensors and name != "star_vars":
                tensors.append(name)
    return tensors 

#print(get_protocol_tensors(PROTOCOLS["si"]))
