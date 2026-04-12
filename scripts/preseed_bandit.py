import os
import random
from orion.rl.bandit import LinUCBBandit
from orion.rl.state_encoder import StateEncoder

def preseed_bandit():
    bandit = LinUCBBandit()
    encoder = StateEncoder()
    
    tasks = [
        {"name": "fix_tenacity_retry", "intent": "bug_fix", "complexity": "medium"},
        {"name": "fix_cachetools_ttl", "intent": "bug_fix", "complexity": "medium"},
        {"name": "implement_pybreaker", "intent": "feature", "complexity": "high"},
        {"name": "fix_async_race", "intent": "bug_fix", "complexity": "high"},
    ]
    
    for _ in range(100):
        task = random.choice(tasks)
        
        # Build state vector
        state_vec = encoder.encode(
            intent_type=task["intent"],
            complexity=task["complexity"],
            prompt=task["name"] + ".py",  # dummy prompt to trigger .py extension match
            cwd="."
        )
        state_list = state_vec.to_list()
        
        # Select action
        action_idx = bandit.select(state_list)
        action_name = bandit.get_action_name(action_idx)
        
        # Simulate reward
        reward = 0.01
        is_fast = "fast" in action_name
        is_balanced = "balanced" in action_name
        is_heavy = "heavy" in action_name
        has_review = "with-review" in action_name or "balanced-review" in action_name
        
        if task["name"] == "fix_tenacity_retry":
            if is_fast:
                reward = random.uniform(0.4, 0.6)
            elif is_balanced and has_review:
                reward = random.uniform(0.7, 0.9)
            else:
                reward = random.uniform(0.5, 0.7)
                
        elif task["name"] == "fix_cachetools_ttl":
            if is_fast:
                reward = random.uniform(0.4, 0.6)
            elif is_balanced and has_review:
                reward = random.uniform(0.7, 0.9)
            else:
                reward = random.uniform(0.5, 0.7)
                
        elif task["name"] == "implement_pybreaker":
            if is_fast:
                reward = random.uniform(0.2, 0.4)
            elif is_balanced or is_heavy:
                reward = random.uniform(0.6, 0.9)
            else:
                reward = random.uniform(0.3, 0.6)
                
        elif task["name"] == "fix_async_race":
            if is_fast:
                reward = random.uniform(0.3, 0.5)
            elif is_heavy and has_review:
                reward = random.uniform(0.7, 0.95)
            else:
                reward = random.uniform(0.4, 0.7)
                
        # Update bandit
        bandit.update(state_list, action_idx, reward)
        
    try:
        save_path = os.environ.get("BANDIT_WEIGHTS_PATH", "/app/bandit_weights.npz")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        bandit.save(save_path)
    except Exception:
        pass
    # LinUCBBandit _save() will trigger automatically during update(), 
    # but bandit.save() explicitly writes the .npz as requested.
