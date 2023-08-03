from scipy.io import savemat
from imageio.v3 import imread
import numpy as np
import glob
import os

# example code
#
# npzFiles = glob.glob("*.npz")
# for f in npzFiles:
#     fm = os.path.splitext(f)[0] + '.mat'
#     d = np.load(f)
#     savemat(fm, d)
#     print('generated ', fm, 'from', f)

workdir = os.path.expanduser("../rawdata")
dataset = "livingroom_full"
ts_mut = 1e6


def process_events(events_list):
    events = np.concatenate(event_list)
    events = events[events[:, 2].argsort()]

    x = events[:, 0].astype(np.uint16)
    x = np.expand_dims(x, axis=-1)

    y = events[:, 1].astype(np.uint16)
    y = np.expand_dims(y, axis=-1)

    timeStamp = (events[:, 2] * ts_mut).astype(np.uint32)
    timeStamp = np.expand_dims(timeStamp, axis=-1)

    polarity = np.where(events[:, 3] > 0, True, False)
    polarity = np.expand_dims(polarity, axis=-1)

    numEvents = events.shape[0]

    events = {'timeStamp': timeStamp, 'y': y, 'x': x, 'polarity': polarity, 'numEvents': numEvents}

    return events


def process_frames(frames_list, ts):
    """
        The frame list ranges from 0 to N, but it should from 1 to N + 3 in .mat data.
        The output from EDI from 2 to N + 2 represents images from 0 to N.
    """
    frames = np.stack(frameslist)
    N, H, W = frames.shape[:3]

    frames = np.empty((len(frameslist) + 2,), dtype=object)
    for i in range(len(frameslist) + 2):
        if i == 0:
            frames[i] = frameslist[i].astype(np.double)
            continue
        elif i == len(frameslist) + 1:
            frames[i] = frameslist[i - 2].astype(np.double)
            continue
        frames[i] = frameslist[i - 1].astype(np.double)
    frames = np.expand_dims(frames, axis=-1)

    xLength = np.repeat(W, N + 2).astype(np.uint16)
    xLength = np.expand_dims(xLength, axis=-1)

    yLength = np.repeat(H, N + 2).astype(np.uint16)
    yLength = np.expand_dims(yLength, axis=-1)

    xPosition = np.repeat(0, N + 2).astype(np.uint16)
    xPosition = np.expand_dims(xPosition, axis=-1)

    yPosition = np.repeat(0, N + 2).astype(np.uint16)
    yPosition = np.expand_dims(yPosition, axis=-1)

    timeStampStart = (np.insert(ts, 0, ts[0]) * ts_mut).astype(np.uint32)
    timeStampStart = np.expand_dims(timeStampStart, axis=-1)

    timeStampEnd = (np.insert(ts, -1, ts[-1]) * ts_mut).astype(np.uint32)
    timeStampEnd = np.expand_dims(timeStampEnd, axis=-1)

    return {"timeStampStart": timeStampStart,
            "timeStampEnd": timeStampEnd,
            "samples": frames,
            "xLength": xLength,
            "yLength": yLength,
            "xPosition": xPosition,
            "yPosition": yPosition,
            "numEvents": N + 2}


if __name__ == '__main__':
    datadir = os.path.join(workdir, dataset)
    imgdir = os.path.join(datadir, "images")
    eventdir = os.path.join(datadir, "events")
    outputdir = os.path.join(datadir, "output")
    ts = os.path.join(datadir, "poses_ts.txt")

    # process events
    eventfiles = [os.path.join(eventdir, f) for f in sorted(os.listdir(eventdir)) if
                  f.endswith('npy') and f.startswith(('0', '1', '2', '3', '4', '5', '6', '7', '8', '9'))]
    event_list = [np.load(e) for e in eventfiles]

    # frames
    framefiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if
                  f.endswith(('png', 'jpg', 'jpeg'))]
    frameslist = [imread(f) for f in framefiles]

    data = {"polarity": process_events(event_list),
            "frame": process_frames(frameslist, np.loadtxt(ts))}

    fm = os.path.join(outputdir, "data.mat")

    # save matlabdata
    matlabdata = {"importParams": "path",
                  "info": "info",
                  "data": data}
    savemat(fm, {"matlabdata": matlabdata})
