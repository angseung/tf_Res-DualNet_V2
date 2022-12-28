import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.activations import selu as SeLU

kaiming_normal = keras.initializers.VarianceScaling(
    scale=2.0, mode="fan_out", distribution="untruncated_normal"
)


def conv3x3(x, out_planes, stride=1, name=None):
    x = layers.ZeroPadding2D(padding=1, name=f"{name}_pad")(x)
    return layers.Conv2D(
        filters=out_planes,
        kernel_size=3,
        strides=stride,
        use_bias=False,
        kernel_initializer=kaiming_normal,
        name=name,
    )(x)


def pconv(x, out_planes, stride=1, name=None):
    # x = layers.ZeroPadding2D(padding=1, name=f"{name}_pad")(x)
    return layers.Conv2D(
        filters=out_planes,
        kernel_size=1,
        strides=stride,
        use_bias=False,
        kernel_initializer=kaiming_normal,
        name=name,
    )(x)


def dconv3x3(x, stride=1, name=None):
    return layers.DepthwiseConv2D(
        kernel_size=3,
        strides=stride,
        use_bias=False,
        padding="same",
        name=name,
    )(x)


def dwht_block(x, planes, stride=1, downsample=None, name=None):
    identity = x

    out_1 = dconv3x3(x, stride=stride, name=f"{name}.dconv1_1")
    out_2 = dconv3x3(x, stride=stride, name=f"{name}.dconv1_2")
    out = SeLU(0.5 * (out_1 + out_2))
    out = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f"{name}.bn1")(out)

    if x.shape[3] != planes:
        pad_tensor = tf.zeros(shape=out.shape)
        out = layers.Concatenate(axis=3)([out, pad_tensor])
        out = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f"{name}.bn2")(out)


    out_1 = dconv3x3(out, stride=1, name=f"{name}.dconv2_1")
    out_2 = dconv3x3(out, stride=1, name=f"{name}.dconv2_2")
    out = SeLU(0.5 * (out_1 + out_2))
    out = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f"{name}.bn3")(out)

    if downsample is not None:
        for layer in downsample:
            identity = layer(identity)

    out = layers.Add(name=f"{name}.add")([identity, out])
    out = SeLU(out)

    return out


def make_layer(x, planes, blocks, stride=1, name=None):
    downsample = None
    inplanes = x.shape[3]
    if stride != 1 or inplanes != planes:
        downsample = [
            layers.Conv2D(
                filters=planes,
                kernel_size=1,
                strides=stride,
                use_bias=False,
                kernel_initializer=kaiming_normal,
                name=f"{name}.0.downsample.0",
            ),
            layers.BatchNormalization(
                momentum=0.9, epsilon=1e-5, name=f"{name}.0.downsample.1"
            ),
        ]

    x = dwht_block(x, planes, stride, downsample, name=f"{name}.0")
    for i in range(1, blocks):
        x = dwht_block(x, planes, name=f"{name}.{i}")

    return x


def resdualnetv2(x, blocks_per_layer, num_classes=10):
    # x = layers.ZeroPadding2D(padding=3, name="conv1_pad")(x)
    x = layers.Conv2D(
        filters=64,
        kernel_size=3,
        strides=1,
        padding="same",
        use_bias=False,
        kernel_initializer=kaiming_normal,
        name="conv1",
    )(x)
    x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="bn1")(x)
    x = layers.ReLU(name="relu1")(x)
    # x = layers.ZeroPadding2D(padding=1, name="maxpool_pad")(x)
    # x = layers.MaxPool2D(pool_size=3, strides=2, name="maxpool")(x)

    x = make_layer(x, 64, blocks_per_layer[0], name="layer1")
    x = make_layer(x, 128, blocks_per_layer[1], stride=2, name="layer2")
    x = make_layer(x, 256, blocks_per_layer[2], stride=2, name="layer3")
    x = make_layer(x, 512, blocks_per_layer[3], stride=2, name="layer4")

    x = layers.GlobalAveragePooling2D(name="avgpool")(x)
    # initializer = keras.initializers.RandomUniform(
    #     -1.0 / math.sqrt(512), 1.0 / math.sqrt(512)
    # )
    x = layers.Dense(
        units=num_classes,
        # kernel_initializer=initializer,
        # bias_initializer=initializer,
        name="fc",
    )(x)

    return x


def resdualnet_v2(x, **kwargs):
    return resdualnetv2(x, [2, 2, 2, 2], **kwargs)


if __name__ == "__main__":
    x = np.random.randn(1, 32, 32, 3)
    x = tf.convert_to_tensor(x, dtype=np.float64)
    model = resdualnet_v2(x, num_classes=10)
