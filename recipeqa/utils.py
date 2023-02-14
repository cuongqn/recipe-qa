def maybe_is_truncated(s):
    punct = [".", "!", "?", '"']
    if s[-1] in punct:
        return False
    return True
