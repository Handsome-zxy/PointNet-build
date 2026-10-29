"""Binary PointNet segmentation model (XYZ + RGB input)."""

from __future__ import annotations

import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

from . import tf_util
from .transform_nets import feature_transform_net, input_transform_net


def placeholder_inputs(batch_size: int, num_point: int):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 6))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point))
    return pointclouds_pl, labels_pl


def get_model(
    point_cloud,
    is_training,
    bn_decay=None,
    use_input_tnet: bool = True,
    use_feature_tnet: bool = True,
    dropout: float = 0.5,
    num_classes: int = 2,
):
    """Return per-point logits [B, N, num_classes] and end_points.

    The 3x3 input T-Net, when enabled, acts on XYZ only.  RGB is concatenated
    after the transform.  The 64x64 feature T-Net is shared by both arms.
    """
    batch_size = point_cloud.get_shape()[0]
    num_point = point_cloud.get_shape()[1]
    end_points = {}

    xyz = point_cloud[:, :, 0:3]
    rgb = point_cloud[:, :, 3:6]

    if use_input_tnet:
        with tf.variable_scope("transform_net1"):
            transform = input_transform_net(xyz, is_training, bn_decay, K=3)
        end_points["input_transform"] = transform
        xyz = tf.matmul(xyz, transform)

    features = tf.concat([xyz, rgb], axis=-1)
    input_image = tf.expand_dims(features, -1)  # [B,N,6,1]

    # conv1 spans the 6 input channels, like the original [1,3] conv over XYZ.
    net = tf_util.conv2d(
        input_image, 64, [1, 6],
        padding="VALID", stride=[1, 1],
        bn=True, is_training=is_training, scope="conv1", bn_decay=bn_decay,
    )
    net = tf_util.conv2d(
        net, 64, [1, 1],
        padding="VALID", stride=[1, 1],
        bn=True, is_training=is_training, scope="conv2", bn_decay=bn_decay,
    )

    if use_feature_tnet:
        with tf.variable_scope("transform_net2"):
            transform = feature_transform_net(net, is_training, bn_decay, K=64)
        end_points["transform"] = transform
        net_transformed = tf.matmul(tf.squeeze(net, axis=[2]), transform)
        point_feat = tf.expand_dims(net_transformed, [2])
    else:
        point_feat = net

    net = tf_util.conv2d(
        point_feat, 64, [1, 1],
        padding="VALID", stride=[1, 1],
        bn=True, is_training=is_training, scope="conv3", bn_decay=bn_decay,
    )
    net = tf_util.conv2d(
        net, 128, [1, 1],
        padding="VALID", stride=[1, 1],
        bn=True, is_training=is_training, scope="conv4", bn_decay=bn_decay,
    )
    net = tf_util.conv2d(
        net, 1024, [1, 1],
        padding="VALID", stride=[1, 1],
        bn=True, is_training=is_training, scope="conv5", bn_decay=bn_decay,
    )

    global_feat = tf_util.max_pool2d(
        net, [num_point, 1], padding="VALID", scope="maxpool"
    )
    global_feat_expand = tf.tile(global_feat, [1, num_point, 1, 1])
    concat_feat = tf.concat([point_feat, global_feat_expand], axis=3)

    net = tf_util.conv2d(
        concat_feat, 512, [1, 1],
        padding="VALID", stride=[1, 1],
        bn=True, is_training=is_training, scope="conv6", bn_decay=bn_decay,
    )
    net = tf_util.dropout(net, keep_prob=dropout, is_training=is_training, scope="dp1")
    net = tf_util.conv2d(
        net, 256, [1, 1],
        padding="VALID", stride=[1, 1],
        bn=True, is_training=is_training, scope="conv7", bn_decay=bn_decay,
    )
    net = tf_util.dropout(net, keep_prob=dropout, is_training=is_training, scope="dp2")
    net = tf_util.conv2d(
        net, 128, [1, 1],
        padding="VALID", stride=[1, 1],
        bn=True, is_training=is_training, scope="conv8", bn_decay=bn_decay,
    )
    net = tf_util.conv2d(
        net, num_classes, [1, 1],
        padding="VALID", stride=[1, 1],
        activation_fn=None, scope="conv9",
    )
    net = tf.squeeze(net, [2])  # [B,N,C]

    return net, end_points


def get_loss(
    logits,
    labels,
    end_points,
    class_weights,
    dice_weight: float = 1.0,
    reg_weight: float = 0.001,
):
    """Weighted CE + Dice + feature-transform orthogonality."""
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    weights = tf.gather(class_weights, labels)
    ce_loss = tf.reduce_mean(ce * weights)

    probs = tf.nn.softmax(logits, axis=-1)[..., 1]
    labels_f = tf.cast(labels, tf.float32)
    intersection = tf.reduce_sum(probs * labels_f)
    denom = tf.reduce_sum(probs) + tf.reduce_sum(labels_f)
    dice_loss = 1.0 - (2.0 * intersection + 1.0) / (denom + 1.0)

    reg_loss = tf.constant(0.0, dtype=tf.float32)
    if "transform" in end_points:
        transform = end_points["transform"]
        k = transform.get_shape()[1]
        mat_diff = tf.matmul(transform, tf.transpose(transform, perm=[0, 2, 1]))
        mat_diff -= tf.constant(np.eye(k), dtype=tf.float32)
        reg_loss = tf.nn.l2_loss(mat_diff)

    total = ce_loss + dice_weight * dice_loss + reg_weight * reg_loss
    return total, ce_loss, dice_loss, reg_loss


if __name__ == "__main__":
    with tf.Graph().as_default():
        inputs = tf.zeros((2, 4096, 6))
        logits, eps = get_model(inputs, tf.constant(True), use_input_tnet=True)
        print(logits, eps.keys())
