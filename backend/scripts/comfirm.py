import numpy as np
from backend.ml.recommender_hybrid import HybridRecommender

hyb = HybridRecommender()
print("ALS users:", len(hyb.cf_model.uid_map))
print("Max user id:", max(hyb.cf_model.uid_map.keys()))
