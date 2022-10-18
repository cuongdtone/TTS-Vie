# -*- coding: utf-8 -*-
# @Organization  : TMT
# @Author        : Cuong Tran
# @Time          : 10/18/2022


import jax
import jax.numpy as jnp
import haiku as hk
from tts.hifi_model import *
from tts.nat_model import *
import pickle

@hk.transform_with_state
def f(tokens, durations, n_frames):
    net = AcousticModel(is_training=False)
    return net.inference(tokens, durations, n_frames)


ckpt_fn = "src/acoustic_latest_ckpt.pickle"
with open(ckpt_fn, "rb") as file:
    dic = pickle.load(file)
    last_step, params, aux, rng, optim_state = (
        dic["step"],
        dic["params"],
        dic["aux"],
        dic["rng"],
        dic["optim_state"],
    )

forward = jax.jit(f.apply, static_argnums=[5])

x = jnp.ones([23], dtype='int').tolist()
print(x)

out_jax = forward(aux, rng, x)


import tensorflow as tf
from jax.experimental import jax2tf

forward_tf = tf.function(jax2tf.convert(forward, enable_xla=False))

