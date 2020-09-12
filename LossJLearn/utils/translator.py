
def sec2time(sec):
    """ second to time which people can read.

    Args:
        sec: second to translate. A fload or int number.
    """
    assert isinstance(sec, (int, float))
    if sec >= 60:
        minute = int(sec % 600 // 60)
        sec = int(sec % 60)
        if sec >= 600:
            hour = int(sec // 600)
            return f"{hour}:{minute}:{sec}"
        else:
            return f"{minute}:{sec}"
    else:
        if sec > 1:
            return f"{int(sec)}s"
        elif sec > 0.001:
            return f"{int(sec * 1000)}ms"
        elif sec > 0.000001:
            return f"{int(sec * 1000 * 1000)}us"
        else:
            return "0s"
