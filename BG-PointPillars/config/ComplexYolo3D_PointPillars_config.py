import numpy as np

boundary = {
    "minX": 0,
    "maxX": 100,
    "minY": -50,
    "maxY": 50,
    "minZ": -10,
    "maxZ": 0
}

BEV_WIDTH = 608  # across y axis 0m ~ 100m
BEV_HEIGHT = 608  # across x axis -50m ~ 50m

DISCRETIZATION = (boundary["maxX"] - boundary["minX"]) / BEV_HEIGHT