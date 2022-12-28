import time
import numpy as np
import tensorflow as tf
from resnet18 import resnet18
from resdualnetv1 import resdualnet_v1
from resdualnetv2 import resdualnet_v2

tf.config.threading.set_inter_op_parallelism_threads(
    1
)

iters = 10
time_resnet = 0.0
time_rdv1 = 0.0
time_rdv2 = 0.0

x = np.random.randn(1, 32, 32, 3)
dummy_input = tf.convert_to_tensor(x, dtype=np.float64)

for iter in range(iters):
    start = time.time()
    dummy_output_1 = resnet18(dummy_input, num_classes=10)
    time_resnet += (time.time() - start)

    start = time.time()
    dummy_output_2 = resdualnet_v1(dummy_input, num_classes=10)
    time_rdv1 += (time.time() - start)

    start = time.time()
    dummy_output_3 = resdualnet_v2(dummy_input, num_classes=10)
    time_rdv2 += (time.time() - start)

print(f"resnet18: [{time_resnet/iters}] resdualnetv1: [{time_rdv1/iters}] resdualnetv2: [{time_rdv2/iters}]")
