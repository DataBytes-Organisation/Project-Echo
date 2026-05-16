import os
import numpy as np
import librosa
import tensorflow as tf


AUDIO_PATH = r"\src\Prototypes\data\database\pig.wav"
CNN14_TFLITE_PATH = r"\src\Prototypes\engine\torch_impl\model\cnn14\cnn14.tflite"
EFF_TFLITE_PATH = r"\src\Prototypes\engine\torch_impl\model\efficientnetv2_rw_s\efficientnetv2_rw_s.tflite"


def load_audio_fixed(path, target_sr=32000, duration_s=5.0):
    audio, _ = librosa.load(path, sr=target_sr, mono=True)
    target_len = int(target_sr * duration_s)

    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)), mode="constant")
    else:
        audio = audio[:target_len]

    return audio.astype(np.float32)


def make_melspec(
    audio,
    sr=32000,
    n_mels=128,
    hop_length=512,
    n_fft=1024,
    fmin=20,
    fmax=14000,
    top_db=80.0,
):
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max, top_db=top_db).astype(np.float32)

    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-6)
    return mel_db.astype(np.float32)


def pad_or_crop_time(spec, target_time_bins):
    current = spec.shape[1]
    if current < target_time_bins:
        pad_width = target_time_bins - current
        spec = np.pad(spec, ((0, 0), (0, pad_width)), mode="constant")
    else:
        spec = spec[:, :target_time_bins]
    return spec.astype(np.float32)


def preprocess_for_cnn14(audio_path):
    audio = load_audio_fixed(audio_path, target_sr=32000, duration_s=5.0)
    spec = make_melspec(
        audio,
        sr=32000,
        n_mels=64,
        hop_length=320,
        n_fft=1024,
        fmin=50,
        fmax=14000,
        top_db=80.0,
    )
    spec = pad_or_crop_time(spec, 500)

    x = np.expand_dims(spec, axis=0)   
    x = np.expand_dims(x, axis=0)      
    return x.astype(np.float32)


def preprocess_for_eff(audio_path):
    audio = load_audio_fixed(audio_path, target_sr=32000, duration_s=5.0)
    spec = make_melspec(
        audio,
        sr=32000,
        n_mels=128,
        hop_length=512,
        n_fft=1024,
        fmin=20,
        fmax=14000,
        top_db=80.0,
    )
    spec = pad_or_crop_time(spec, 313)

    # NCHW
    x = np.expand_dims(spec, axis=0)  
    x = np.expand_dims(x, axis=0)     
    return x.astype(np.float32)


def adapt_to_interpreter_input(x_nchw, input_shape):
    """
    TFLite model may expect either NCHW or NHWC.
    We start from NCHW and adapt if needed.
    """
    x = x_nchw.astype(np.float32)

    if len(input_shape) != 4:
        raise ValueError(f"Expected 4D input shape, got {input_shape}")

    if tuple(input_shape) == tuple(x.shape):
        return x
  
    x_nhwc = np.transpose(x, (0, 2, 3, 1))
    if tuple(input_shape) == tuple(x_nhwc.shape):
        return x_nhwc

    raise ValueError(
        f"Could not adapt input. Model expects {input_shape}, but prepared NCHW={x.shape} and NHWC={x_nhwc.shape}"
    )


def run_tflite(model_path, prepared_input_nchw, model_name):
    print("\n" + "=" * 70)
    print(f"Testing {model_name}")
    print("=" * 70)

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_index = input_details[0]["index"]
    input_shape = tuple(input_details[0]["shape"])
    input_dtype = input_details[0]["dtype"]

    print("Model path:", model_path)
    print("Interpreter input shape:", input_shape)
    print("Interpreter input dtype:", input_dtype)
    print("Prepared NCHW shape:", prepared_input_nchw.shape)

    x = adapt_to_interpreter_input(prepared_input_nchw, input_shape).astype(input_dtype)

    interpreter.set_tensor(input_index, x)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]["index"])

    print("Actual fed input shape:", x.shape)
    print("Output shape:", output.shape)
    print("Output dtype:", output.dtype)

    flat = output.flatten()
    preview = flat[:10] if flat.size >= 10 else flat
    print("Output preview:", preview)

    return output


def main():
    if not os.path.exists(AUDIO_PATH):
        raise FileNotFoundError(f"Audio file not found: {AUDIO_PATH}")
    if not os.path.exists(CNN14_TFLITE_PATH):
        raise FileNotFoundError(f"CNN14 TFLite not found: {CNN14_TFLITE_PATH}")
    if not os.path.exists(EFF_TFLITE_PATH):
        raise FileNotFoundError(f"EfficientNetV2 TFLite not found: {EFF_TFLITE_PATH}")

    cnn14_input = preprocess_for_cnn14(AUDIO_PATH)
    eff_input = preprocess_for_eff(AUDIO_PATH)

    run_tflite(CNN14_TFLITE_PATH, cnn14_input, "CNN14 TFLite")
    run_tflite(EFF_TFLITE_PATH, eff_input, "EfficientNetV2 TFLite")

    print("\nDone. If both models printed output shapes without crashing, your TFLite validation basically worked.")


if __name__ == "__main__":
    main()