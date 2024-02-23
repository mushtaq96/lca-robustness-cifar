# -*- coding: utf-8 -*-
"""
Created on Sat May  6 22:37:41 2023

@author: jakob
"""

import csv

# Class mapping via dictionary
sign_map = {
                0:  "Speed Limit 20 km/h",
                1:  "Speed Limit 30 km/h",
                2:  "Speed Limit 50 km/h",
                3:  "Speed Limit 60 km/h",
                4:  "Speed Limit 70 km/h",
                5:  "Speed Limit 80 km/h",
                6:  "End of speed limit",
                7:  "Speed Limit 100 km/h",
                8:  "Speed Limit 120 km/h",
                9:  "No passing",
                10: "No passing for vehicle",
                11: "Right of way the next intersection",
                12: "Priority road",
                13: "Yield",
                14: "Stop",
                15: "No vehicles",
                16: "Vehicles over 3.5t not allowed",
                17: "No entry",
                18: "General caution",
                19: "Dangerous curve to the left",
                20: "Dangerous curve to the right",
                21: "Double curve",
                22: "Bumpy road",
                23: "Slippery road",
                24: "Road narrows",
                25: "Road work",
                26: "Traffic lights",
                27: "Pedestrians",
                28: "Children crossing",
                29: "Bicycles crossing",
                30: "Beware of ice/snow",
                31: "Wild animals crossing",
                32: "End of all speed limits",
                33: "Turn right ahead",
                34: "Turn left ahead",
                35: "Ahead only",
                36: "Go straight or right",
                37: "Go straight or left",
                38: "Keep right",
                39: "Keep left",
                40: "Rounabout mandatory",
                41: "End of no passing",
                42: "End of no passing by vehicles",
                    }

stem_dir = "/home/bokhars/thesis/robustness/Code"
# "C:\\Weiterbildung\\Master_Autonomes_Fahren\\Semester_4_Masterarbeit\\Code\\"

patch_dict = {
        "rotation_max": 0.0,
        "scale_min": 0.1,
        "scale_max": 0.5,
        "learning_rate": 1.0,
        "max_iter": 500,
        "patch_shape": (3, 5, 5),
        "patch_location": (1,1),
        "patch_type": "circle",
        "optimizer": "Adam",
        "targeted": "False",
        }

patch_shapes = [(3, 0, 0), (3, 3, 3), (3, 6, 6), (3, 9, 9), (3, 12, 12)]


