import sys, os, traceback

try:
    sys.stdout.reconfigure(encoding='utf-8')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

    print("=== 1. MEMUAT INVENTORY BERSIH ===", flush=True)
    import numpy as np
    import pandas as pd
    from PIL import Image

    df = pd.read_csv(r"model_coins\splits\dataset_detailed_inventory.csv")
    df_unique = df[~df['is_duplicate']].reset_index(drop=True)
    print(f"Total citra unik: {len(df_unique)}", flush=True)
    print(df_unique['target_class_5way'].value_counts().to_string(), flush=True)

    print("\n=== 2. MEMUAT TFLITE MODEL UNTUK EXTRACT PROBABILITIES ===", flush=True)
    import tensorflow as tf

    tflite_path = r"model_coins\models\best.tflite"
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']
    target_dtype = input_details[0]['dtype']
    print(f"TFLite Input shape: {input_shape}, dtype: {target_dtype}", flush=True)

    probs_list = []
    for p in df_unique['path']:
        im = Image.open(p).convert('RGB').resize((int(input_shape[2]), int(input_shape[1])), Image.Resampling.BILINEAR)
        arr = np.expand_dims(np.array(im, dtype=target_dtype), axis=0)
        interpreter.set_tensor(input_details[0]['index'], arr)
        interpreter.invoke()
        pr = interpreter.get_tensor(output_details[0]['index'])[0]
        probs_list.append(pr)
    
    probs_all = np.array(probs_list)
    print(f"Probabilitas berhasil dihitung untuk {len(probs_all)} citra: shape {probs_all.shape}", flush=True)

    # Simpan hasil sementara
    np.save(r"model_coins\results\probs_89_tflite.npy", probs_all)
    print("Selesai tahap ekstraksi probabilitas TFLite!", flush=True)

except Exception as e:
    with open(r"model_coins\eval_err.txt", 'w', encoding='utf-8') as ef:
        traceback.print_exc(file=ef)
    print(f"CRASH: {e}", flush=True)
    sys.exit(1)

