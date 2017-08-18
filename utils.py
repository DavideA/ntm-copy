import sys


def print_progress(progress):
    """
    Prints a fancy progress bar.
    Copied from [1].
    """

    barLength = 20  # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Finished.\r\n"
    block = int(round(barLength * progress))
    text = "\rPercent: [%s] %.2f%% %s" % ("#" * block + " " * (barLength - block), progress * 100, status)
    sys.stdout.write(text)
    sys.stdout.flush()

"""
References
----------
[1] https://github.com/carpedm20/NTM-tensorflow/blob/master/ops.py
"""
