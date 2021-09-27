import os
# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['JAX_PLATFORM_NAME'] = 'gpu'
# os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

import jax
import jax.lax
import numpy as np
import jax.numpy as jnp

from tqdm import trange, tqdm

import haiku as hk
import optax
import jmp

from typing import Any, Iterable, Mapping, NamedTuple, Tuple
import atexit
import tree
import functools
import einops
import tensorflow_datasets as tfds
import tensorflow as tf
import dataset


class FLAGS(NamedTuple):
    DATA_ROOT = '/ramdisk/'
    LOG_ROOT = '/workspace/runs/imagenet_ResNet18_lrelu/'
    KEY = jax.random.PRNGKey(1)
    BATCH_SIZE = 128
    MAX_STEP = 10000
    INIT_LR = 1e-1
    WEIGHT_DECAY = 1e-5
    SAVE = False


MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def tprint(obj):
    tqdm.write(obj.__str__())


class TrainState(NamedTuple):
    params: hk.Params
    state: hk.State
    opt_state: optax.OptState


def softmax_cross_entropy(logits, labels):
    logp = jax.nn.log_softmax(logits)
    loss = -jnp.take_along_axis(logp, labels[:, None], axis=-1)
    return loss


def correct_topk(logits, labels, k):
    logits = jnp.asarray(logits)
    labels = jnp.asarray(labels)[..., None]
    preds = jnp.argsort(logits, axis=-1)[..., -k:]
    return jnp.any(preds == labels, axis=-1)


def weight_penalty(params):
    l2_params = [p for (mod_name, _), p in tree.flatten_with_path(
        params) if 'batchnorm' not in mod_name]
    return 0.5 * sum(jnp.sum(jnp.square(p)) for p in l2_params)


def forward(images, is_training: bool):
    net = hk.nets.ResNet18(num_classes=1000)
    return net(images, is_training=is_training)


def main():
    ## DATASET ##
    train_dataset = dataset.load(True, FLAGS.BATCH_SIZE, jnp.float32)

    ## MODEL TRANSFORM ##
    model = hk.without_apply_rng(hk.transform_with_state(forward))

    ## OPTIMIZER ##
    learning_rate_fn = optax.cosine_onecycle_schedule(
        transition_steps=FLAGS.MAX_STEP,
        peak_value=FLAGS.INIT_LR,
        pct_start=0.1,
        div_factor=25.0,
        final_div_factor=1e4
    )
    optimizer = optax.sgd(learning_rate_fn, momentum=0.9, nesterov=False)


    ## TRAIN STATE ##
    def get_init_trainstate(rng_key):
        params, state = model.init(rng_key, jnp.empty((1, 224, 224, 3), dtype=jnp.float32), is_training=True)
        opt_state = optimizer.init(params)
        train_state = TrainState(params, state, opt_state)
        return train_state

    train_state = get_init_trainstate(FLAGS.KEY)
    train_state = jax.device_put_replicated(train_state, jax.local_devices())
    n_devices = len(jax.local_devices())


    @functools.partial(jax.pmap, axis_name='device', donate_argnums=(0,))
    def train_step(train_state: TrainState, batch: dict):
        params, state, opt_state = train_state
        input, target = batch['images'], batch['labels']
        def loss_fn(p):
            logits, state_new = model.apply(p, state, input, is_training=True)
            loss = softmax_cross_entropy(logits, target).mean() + weight_penalty(p)*FLAGS.WEIGHT_DECAY
            aux = {
                'state': state_new,
                'loss': loss,
                'logits': logits,
            }
            return loss, aux
        grads, aux = jax.grad(loss_fn, has_aux=True)(params)
        grads = jax.lax.pmean(grads, axis_name='device')
        delta, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, delta)
        summary = {
            'loss': aux['loss'],
            # 'correct': correct_topk(aux['logits'], target, k=1),
        }
        return TrainState(params, aux['state'], opt_state), summary


    @functools.partial(jax.pmap, axis_name='j', donate_argnums=(0,))
    def eval_step(train_state: TrainState, batch: dict):
        params, state, _ = train_state
        input, target = batch['images'], batch['labels']
        logits, _ = model.apply(params, state, input, is_training=False)
        summary = {
            "loss": softmax_cross_entropy(logits, target),
            "correct_top1": correct_topk(logits, target, k=1),
            "correct_top5": correct_topk(logits, target, k=5)
        }
        return summary


    def evaluate(train_state, data_loader):
        corrects_t1 = []
        corrects_t5 = []
        losses = []
        for batch in data_loader:
            batch = {
                'images': jnp.asarray(einops.rearrange(batch['images'], '(n b) h w c -> n b h w c', n=n_devices)),
                'labels': jnp.asarray(einops.rearrange(batch['labels'], '(n b) -> n b', n=n_devices)),
            }
            summary = eval_step(train_state, batch)
            corrects_t1.append(summary['correct_top1'].flatten())
            corrects_t5.append(summary['correct_top5'].flatten())
            losses.append(summary['loss'].flatten())
        summary = {
            "loss": np.mean(losses).item(),
            "acc_t1": np.mean(corrects_t1).item()*100,
            "acc_t5": np.mean(corrects_t5).item()*100,
        }
        return summary


    def save_pickle(filename='model.pickle'):
        state_dict = jax.tree_map(lambda x: x[0], train_state)._asdict()
        pickle_path = os.path.join(FLAGS.LOG_ROOT, filename)
        np.save(pickle_path, state_dict)
        print(f'[SAVE] {pickle_path}')


    if FLAGS.SAVE:
        atexit.register(save_pickle)


    train_loader = iter(train_dataset)
    losses = []
    corrects = []

    for iter_idx in (pbar := trange(FLAGS.MAX_STEP)):
        batch = next(train_loader)
        batch = {
            'images': jnp.asarray(einops.rearrange(batch['images'], '(n b) h w c -> n b h w c', n=n_devices)),
            'labels': jnp.asarray(einops.rearrange(batch['labels'], '(n b) -> n b', n=n_devices)),
        }
        train_state, aux = train_step(train_state, batch)
        # losses.append(aux['loss'].flatten())
        # corrects.append(aux['correct'].flatten())

        if iter_idx > 0:
            pbar.set_description(f"{FLAGS.BATCH_SIZE*pbar.format_dict['rate']:.1f} samples/sec")

        if iter_idx % 100 == 0:
            # loss = np.mean(losses).item()
            # acc = np.mean(corrects).item()*100
            losses.clear()
            corrects.clear()
            last_lr = learning_rate_fn(train_state.opt_state[-1].count[0]).item()
            tprint(f'[{iter_idx}/{len(pbar)}] LR: {last_lr:.3f}')
            # tprint(f'[{iter_idx}/{len(pbar)}] LR: {last_lr:.3f} | [Train] Loss {loss:.3f} AccT1 {acc:.2f}')


if __name__ == '__main__':
    main()
