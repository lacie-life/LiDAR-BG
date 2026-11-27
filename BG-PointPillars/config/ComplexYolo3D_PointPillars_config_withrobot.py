import numpy as np

boundary = {
    "minX": -60,
    "maxX": 0,
    "minY": -30,
    "maxY": 30,
    "minZ": -10,
    "maxZ": 1
}

BEV_WIDTH = 608  # across y axis 0m ~ 100m
BEV_HEIGHT = 608  # across x axis -50m ~ 50m

DISCRETIZATION = (boundary["maxX"] - boundary["minX"]) / BEV_HEIGHT