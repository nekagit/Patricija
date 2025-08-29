from __future__ import annotations
import threading
import queue
import traceback
from pathlib import Path

import pandas as pd

from utils.training import TrainConfig, train_and_select, save_artifacts
from utils.dataset_compatibility import clean_column_names


def _worker_train_and_save(
    df: pd.DataFrame,
    config: TrainConfig,
    out_dir_str: str,
    q: queue.Queue
):
    try:
        def progress_callback(progress: float, message: str):
            q.put({"type": "progress", "progress": progress, "message": message})

        # Clean column names to handle extra spaces
        df_cleaned = clean_column_names(df.copy())
        
        # Update config target to use cleaned column name
        if config.target not in df_cleaned.columns:
            # Try to find the target column with different variations
            target_variations = [config.target, config.target.strip(), config.target.lower(), config.target.upper()]
            for variation in target_variations:
                if variation in df_cleaned.columns:
                    config.target = variation
                    break
            else:
                # If still not found, show available columns and raise error
                available_cols = list(df_cleaned.columns)
                raise ValueError(f"Target column '{config.target}' not found in DataFrame. Available columns: {available_cols}")

        model, model_name, features, scaler, encoders, _, _, metrics, lime_bg = train_and_select(
            df=df_cleaned,
            config=config,
            progress_cb=progress_callback
        )

        progress_callback(0.9, f"Saving '{model_name}' artifacts...")

        out_dir = Path(out_dir_str)
        save_artifacts(
            out_dir=str(out_dir),
            model_name=model_name,
            model=model,
            scaler=scaler,
            encoders=encoders,
            features=list(features),
            lime_background=lime_bg
        )

        final_result = {
            "ok": True,
            "model_name": model_name,
            "metrics": metrics,
            "features": list(features),
            "model": model,
            "out_path": out_dir,
        }
        q.put({"type": "result", "result": final_result})

    except Exception as e:
        tb_str = traceback.format_exc()
        error_result = { "ok": False, "err": str(e), "trace": tb_str }
        q.put({"type": "result", "result": error_result})


def launch_training_and_saving(df: pd.DataFrame, config: TrainConfig, out_dir: str):
    q = queue.Queue()
    result_container = {} 
    out_path = Path(out_dir)

    thread = threading.Thread(
        target=_worker_train_and_save,
        args=(df, config, out_dir, q)
    )
    thread.start()

    return thread, result_container, out_path, q