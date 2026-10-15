import re
import json
import math


def clamp(value, low, high):
    return max(low, min(value, high))


def is_even(n):
    return n % 2 == 0


def is_missing(x):
    return x == None


def format_percent(value):
    ratio = value / 100
    return f"{value:.1f}%"


def to_json(data):
    return json.dumps(data, ensure_ascii=False)


def round_up(x):
    return math.ceil(x)
