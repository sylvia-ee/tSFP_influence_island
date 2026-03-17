from functools import lru_cache
import pandas as pd

from preprocess_functions import load_game_config

def solve_round(round_rules, actions, action_probs, p_convince):

    """ 
    description: solves one round (score carries over between trials) of game 

    inputs: 
    - round_rules (list of dicts): bounds of "convince" and "win" states for each trial in ORDER of rounds
        e.g. {"conv": None, "win": (50, 100), 
               "conv": (60, 70), "win": (70, 100)} 
    - actions (dict): increments for each possible action 
        e.g. {"small": [1, 2, 3...20], "large": [20, 21..., 44, 45]} 
    - action_probs (dict): probability for each possible increment for each action 
        e.g. {"small": [1/20,...,1/20], "large": [1/25,...,1/25]} 
    - p_convince (float): probability of success if in convince range and "convince" action chosen 
        e.g. 0.5 # 50% chance that convince succeeds

    outputs: 
    - V (function): value function caching max. win probability for a given state (trial, score, vs_left) assuming optimal action
    - Q (function): action-value function returning best possible outcome for a given state over all possible actions
    """

    n_trials = len(round_rules) 

    @lru_cache(None)
    def V(t, score, vs_left):

        """
        description: caches the maximum win probability for a given state (trial, score, vs_left) assuming optimal action
        inputs:
        - t (int): current trial
        - score (int): current score
        - vs_left (int): "very small" actions left
        outputs:
        - max_prob (float): maximum win probability for the given state
    
        """
        if t == n_trials:
            return 1.0

        best = 0
        for action in actions:
            if action == "very_small" and vs_left == 0:
                continue
            best = max(best, Q(t, score, vs_left, action))
        
        max_prob = min(1.0, max(0.0, best))

        return max_prob


    def Q(t, score, vs_left, action):

        """
        description: returns the expected win probability for a given state (trial, score, vs_left) and action
        inputs:
        - t (int): current trial
        - score (int): current score
        - vs_left (int): "very small" actions left
        - action (str): action to evaluate
        outputs:
        - expected (float): expected win probability for the given state and action
        """

        rule = round_rules[t]
        win_low, win_high = rule["win"]
        conv_range = rule["conv"]

        if action == "convince":
            if conv_range and conv_range[0] <= score <= conv_range[1]:
                return p_convince
            return 0

        increments = actions[action]
        probs = action_probs[action]

        vs_next = vs_left - 1 if action == "very_small" else vs_left

        expected = 0

        for inc, p in zip(increments, probs):

            new_score = score + inc

            if new_score > 101:
                val = 0

            elif win_low <= new_score <= win_high:
                val = V(t + 1, new_score, vs_next)

            else:
                val = V(t, new_score, vs_next)

            expected += p * val

        return expected

    return V, Q

def compute_policy(round_rules, actions, action_probs, p_convince):
    """
    description: computes optimal user decision for a trial 
    inputs: see solve_round function
    outputs: df of every state and every action with win probabilities 
    """

    V, Q = solve_round(round_rules, actions, action_probs, p_convince)
    n_trials = len(round_rules)

    rows = []

    for t in range(n_trials):
        for score in range(0, 101):
            for vs_left in range(4):

                for action in actions:

                    if action == "very_small" and vs_left == 0:
                        continue

                    val = Q(t, score, vs_left, action)

                    rows.append({
                        "trial": t + 1,
                        "score": score,
                        "vs_left": vs_left,
                        "action": action,
                        "win_probability": val
                    })
    
    full_Q_table = pd.DataFrame(rows)

    optimal_Q_table = full_Q_table.groupby(["trial", "score", "vs_left"]).apply(lambda x: x.loc[x["win_probability"].idxmax()])

    return full_Q_table, optimal_Q_table


def build_decision_tbl(folder_path):

    actions, action_probs, round_map, vs_limit, p_convince = load_game_config(folder_path)

    all_full = []
    all_optimal = []

    for r, rules in round_map.items():

        full_Q, optimal_Q = compute_policy(
            rules,
            actions,
            action_probs,
            p_convince
        )

        full_Q["round"] = r
        optimal_Q["round"] = r

        for t, rule in enumerate(rules, start=1):

            win_low, win_high = rule["win"]

            full_Q.loc[full_Q["trial"] == t, "win_low"] = win_low
            full_Q.loc[full_Q["trial"] == t, "win_high"] = win_high

            optimal_Q.loc[optimal_Q["trial"] == t, "win_low"] = win_low
            optimal_Q.loc[optimal_Q["trial"] == t, "win_high"] = win_high

            if rule["conv"] is not None:
                conv_low, conv_high = rule["conv"]
            else:
                conv_low, conv_high = None, None

            full_Q.loc[full_Q["trial"] == t, "conv_low"] = conv_low
            full_Q.loc[full_Q["trial"] == t, "conv_high"] = conv_high

            optimal_Q.loc[optimal_Q["trial"] == t, "conv_low"] = conv_low
            optimal_Q.loc[optimal_Q["trial"] == t, "conv_high"] = conv_high

        all_full.append(full_Q)
        all_optimal.append(optimal_Q)

    full_Q_all = pd.concat(all_full, ignore_index=True)
    optimal_Q_all = pd.concat(all_optimal, ignore_index=True)

    return full_Q_all, optimal_Q_all

