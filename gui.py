import streamlit as st
import pandas as pd
import os

from figure_functions import plot_policy_heatmaps


BASE_PATH = "data"
st.title("Influencer Island Decision Helper")

col1, col2 = st.columns(2)

with col1:
    color_mode = st.selectbox(
        "Color Mode",
        ["default", "colorblind"]
    )

with col2:
    theme = st.get_option("theme.base")  # "light" or "dark"


dataset_options = [
    d for d in os.listdir(BASE_PATH)
    if os.path.isdir(os.path.join(BASE_PATH, d))
]

dataset = st.selectbox("Dataset", dataset_options)
dataset_path = os.path.join(BASE_PATH, dataset)

optimal_Q = pd.read_csv(
    os.path.join(dataset_path, "tables", "optimal_Q_table.csv")
)

full_Q = pd.read_csv(
    os.path.join(dataset_path, "tables", "full_Q_table.csv")
)


col1, col2, col3, col4 = st.columns(4)

stages = sorted(optimal_Q["round"].unique())

with col1:
    stage = st.selectbox("Stage", stages)

rounds = sorted(
    optimal_Q[optimal_Q["round"] == stage]["trial"].unique()
)

with col2:
    round_num = st.selectbox("Round", rounds)

vs_vals = sorted(optimal_Q["vs_left"].unique())

with col3:
    vs_left = st.selectbox("Chocolate Left", vs_vals)

with col4:
    score = int(st.number_input("Score", step=1))


mask = (
    (optimal_Q["round"] == stage) &
    (optimal_Q["trial"] == round_num) &
    (optimal_Q["score"] == score) &
    (optimal_Q["vs_left"] == vs_left)
)

if not mask.any():
    st.error("Invalid state")
    st.stop()


action_label_map = {
    "very_small": "chocolate [2-9]",
    "small": "small [1-20]",
    "large": "large [20-45]"
}

opt_row = optimal_Q[mask].iloc[0]

df = full_Q[
    (full_Q["round"] == stage) &
    (full_Q["trial"] == round_num) &
    (full_Q["score"] == score) &
    (full_Q["vs_left"] == vs_left)
].copy()

df = df.sort_values("win_probability", ascending=False)

best_action = opt_row["action"]
best_prob = opt_row["win_probability"]

df["action_display"] = df["action"].map(action_label_map)


# gui interface

st.subheader("Recommendation")
st.success(f"Best action: {action_label_map.get(best_action, best_action)}")
st.write(f"Win probability: {best_prob:.4f}")


st.subheader("Decision Map")

figs = plot_policy_heatmaps(
    optimal_Q,
    lower_only=True,
    highlight_state=(stage, round_num, score, vs_left),
    figsize_scale=1.4,
    font_scale=1.4,
    color_mode=color_mode,
    theme=theme
)

st.pyplot(figs[stage])