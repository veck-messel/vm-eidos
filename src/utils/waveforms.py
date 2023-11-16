from math import sin, cos

def hanning(f, t, n):
    return (1 / 2) * (1 - cos(f * t / n)) * (sin(f * t))
