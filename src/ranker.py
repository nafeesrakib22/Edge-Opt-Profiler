import pandas as pd
import numpy as np

class ModelRanker:
    def rank_models(self, df, alpha, beta, gamma):
        """
        Applies the Weighted Min-Max Normalization formula.
        """
        # Create a copy so we don't mess up the original table
        df = df.copy()

        # 1. Get Min and Max for each metric
        s_min, s_max = df['size_mb'].min(), df['size_mb'].max()
        t_min, t_max = df['latency_ms'].min(), df['latency_ms'].max()
        a_min, a_max = df['accuracy'].min(), df['accuracy'].max()

        # Helper to avoid division by zero (if max == min)
        def normalize(val, min_val, max_val):
            if max_val - min_val == 0:
                return 0.0 # All models are equal in this metric
            return (val - min_val) / (max_val - min_val)

        # 2. Calculate Scores
        scores = []
        for index, row in df.iterrows():
            # Normalize Size (Lower is better)
            norm_size = normalize(row['size_mb'], s_min, s_max)
            
            # Normalize Latency (Lower is better)
            norm_time = normalize(row['latency_ms'], t_min, t_max)
            
            # Normalize Accuracy (Higher is better, so we do 1 - Norm)
            norm_acc = normalize(row['accuracy'], a_min, a_max)

            # --- YOUR EQUATION ---
            # Cost = α(Size) + β(1 - Accuracy) + γ(Time)
            score = (alpha * norm_size) + \
                    (beta * (1 - norm_acc)) + \
                    (gamma * norm_time)
            
            scores.append(score)

        df['final_score'] = scores
        
        # 3. Sort by Lowest Score (Best)
        return df.sort_values(by='final_score', ascending=True)