import json, pickle, argparse
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, Bidirectional, LSTM
from tensorflow.keras.callbacks import EarlyStopping

COLS6 = ["gravity", "ph", "osmo", "cond", "urea", "calc"]

def make_features(df_raw: pd.DataFrame):
    df1 = df_raw.copy()
    df1["osmo_cond_ratio"] = df1["osmo"] / df1["cond"]
    df1["urea_calc_diff"] = df1["urea"] - df1["calc"]
    df1["osmo_urea_interaction"] = df1["osmo"] * df1["urea"]

    mu = df1[COLS6].mean()
    sigma = df1[COLS6].std().replace(0, 1.0)
    df1.loc[:, COLS6] = (df1[COLS6] - mu) / sigma

    q_edges = {}
    for c in COLS6:
        bins, edges = pd.qcut(df_raw[c], 5, retbins=True, duplicates="drop")
        q_edges[c] = edges.tolist()
        df1[f"{c}_bin"] = pd.cut(df_raw[c], bins=edges, labels=False, include_lowest=True)
    return df1, mu.to_dict(), sigma.to_dict(), q_edges

def build_model(input_len: int):
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=3, activation="relu", input_shape=(input_len, 1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Bidirectional(LSTM(32, return_sequences=False)))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model

def main(args):
    df = pd.read_csv(args.csv)
    df_feat, mu, sigma, q_edges = make_features(df)

    feature_cols = ["gravity","ph","osmo","cond","urea","calc",
                    "osmo_cond_ratio","urea_calc_diff","osmo_urea_interaction",
                    "gravity_bin","ph_bin","osmo_bin","cond_bin","urea_bin","calc_bin"]

    X = df_feat[feature_cols].to_numpy().astype("float32").reshape((-1, len(feature_cols), 1))
    y = df_feat["target"].astype("float32").to_numpy()

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model = build_model(X.shape[1])

    es = EarlyStopping(monitor="val_accuracy", mode="max", patience=10, restore_best_weights=True)
    hist = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                     epochs=args.epochs, batch_size=args.batch_size, callbacks=[es], verbose=1)

    model.save(args.model_out)
    with open(args.preproc_out, "wb") as f:
        pickle.dump({"mu": mu, "sigma": sigma, "q_edges": q_edges, "feature_cols": feature_cols}, f)

    best_val_acc = max(hist.history["val_accuracy"])
    print(f"Best validation accuracy: {best_val_acc:.4f}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="kidney_stone_data (1).csv")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--model_out", default="my_model.h5")
    p.add_argument("--preproc_out", default="preprocessing.pkl")
    args = p.parse_args()
    main(args)
