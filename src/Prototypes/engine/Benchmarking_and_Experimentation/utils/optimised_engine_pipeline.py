# utils/optimised_engine_pipeline.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
from typing import Tuple

import tensorflow as tf
import tensorflow_hub as hub

# 项目内配置与数据流水线
from config.system_config import SC                 # 全局系统配置
from config.model_configs import MODELS            # 模型列表与超参
from utils.data_pipeline import create_datasets    # 数据集拆分/读取
try:
    # 如果你的 data_pipeline 里有 build_datasets，我们优先使用（带 batch/cache 等优化）
    from utils.data_pipeline import build_datasets  # type: ignore
    _HAS_BUILD_PIPELINES = True
except Exception:
    _HAS_BUILD_PIPELINES = False


# ---------------------------------------------------------------------
# 构建分类头前的主干网络（统一只用 tf.keras）
# ---------------------------------------------------------------------
def build_model(
    model_name: str,
    num_classes: int,
    l2_regularization: bool = False,
    l2_coefficient: float = 1e-4,
) -> tf.keras.Model:
    """
    用 tf.keras.applications 的 MobileNetV3Small 做主干（不走 TF-Hub）。
    """
    h = int(SC.get("MODEL_INPUT_IMAGE_HEIGHT", 224))
    w = int(SC.get("MODEL_INPUT_IMAGE_WIDTH", 224))
    c = int(SC.get("MODEL_INPUT_IMAGE_CHANNELS", 3))

    inputs = tf.keras.Input(shape=(h, w, c), dtype=tf.float32, name="input")

    # 官方预处理：把 RGB 输入转换到 MobileNetV3 需要的范围
    from tensorflow.keras.applications import mobilenet_v3
    x = mobilenet_v3.preprocess_input(inputs)

    # 主干：imagenet 预训练，去掉顶层，做全局平均池化
    backbone = tf.keras.applications.MobileNetV3Small(
        include_top=False,
        weights="imagenet",
        input_shape=(h, w, c),
        pooling="avg"  # 直接得到 (batch, features)
    )
    backbone.trainable = True  # 如需冻结可设 False
    y = backbone(x, training=False)

    # 分类头
    reg = tf.keras.regularizers.l2(l2_coefficient) if l2_regularization else None
    outputs = tf.keras.layers.Dense(
        num_classes, activation="softmax", kernel_regularizer=reg, name="classifier"
    )(y)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name=f"{model_name}_clf")



# ---------------------------------------------------------------------
# 训练入口（Notebook 会调用这个函数）
# ---------------------------------------------------------------------
def train_model(
    model_name: str,
    epochs: int = 2,
    batch_size: int = 16,
    l2_regularization: bool = False,
    l2_coefficient: float = 1e-4,
):
    """
    训练指定模型；返回 (model, history)
    该函数依赖：
      - SC['AUDIO_DATA_DIRECTORY'] 指向包含子类文件夹的音频目录
      - utils.data_pipeline.create_datasets / build_datasets
    """
    # 1) 读取数据路径并校验
    audio_dir = SC.get("AUDIO_DATA_DIRECTORY")
    if not audio_dir or not os.path.isdir(audio_dir):
        raise ValueError(f"Directory does not exist: {audio_dir}")

    # 可选：把 batch_size 写回全局配置（兼容旧代码）
    SC["CLASSIFIER_BATCH_SIZE"] = batch_size

    # 2) 拆分数据（返回初级 dataset 及类别名）
    # create_datasets 的具体签名以你现有实现为准；多数实现返回：
    # train_ds_init, val_ds_init, test_ds_init, class_names
    ds_tuple = create_datasets(audio_dir)
    if len(ds_tuple) == 4:
        train_ds_init, val_ds_init, test_ds_init, class_names = ds_tuple
    else:
        # 兜底：有些实现可能不返回 test
        train_ds_init, val_ds_init, class_names = ds_tuple
        test_ds_init = None

    num_classes = len(class_names)

    # 3) 构建模型（只用 tf.keras）
    model = build_model(
        model_name=model_name,
        num_classes=num_classes,
        l2_regularization=l2_regularization,
        l2_coefficient=l2_coefficient,
    )

    # 4) 组装训练/验证数据流水线
    if _HAS_BUILD_PIPELINES:
        # 推荐：你的 build_datasets 一般会把 batch/缓存/并行等都配置好
        train_ds, val_ds, test_ds = build_datasets(
            train_ds_init, val_ds_init, test_ds_init, class_names, model_name=model_name
        )
    else:
        # 简易兜底：仅做 batch / prefetch
        AUTOTUNE = tf.data.AUTOTUNE
        train_ds = train_ds_init.batch(batch_size).prefetch(AUTOTUNE)
        val_ds = val_ds_init.batch(batch_size).prefetch(AUTOTUNE)
        test_ds = None if test_ds_init is None else test_ds_init.batch(batch_size).prefetch(AUTOTUNE)

    # 5) 编译（只用 tf.keras 组件）
    lr = float(MODELS.get(model_name, {}).get("learning_rate", 1e-4))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",   # ← 这里从 sparse_... 改成 categorical_...
        metrics=["accuracy"],
    ) 


    # 6) 训练
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=int(epochs),
        verbose=1,
    )

    # 如果需要评估测试集，可在外部调用 model.evaluate(test_ds)
    return model, history
