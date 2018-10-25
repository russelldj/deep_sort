import glob
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

DATA_FOLDER="/home/drussel1/data/readonly/groundtruth/ADL/P_{:02d}.csv"

def interpolate(slice_, style='linear'):
    is_first = True

    # weird half-scale error 
    xmin = 2 * (slice_['xmin'])
    ymin = 2 * (slice_['ymin'])
    xmax = 2 * (np.array(slice_['width']) + np.array(slice_['xmin']))
    ymax = 2 * (np.array(slice_['height']) + np.array(slice_['ymin']))
    frame = np.array(slice_['frame'])
    
    x = np.arange(frame[0], frame[-1] + 1)
    IDs = np.full(x.shape, slice_['ID'].index[0]) 
    
    # linear
    if style == 'linear':
        interp_xmin  = np.interp(x, frame, xmin)
        interp_ymin  = np.interp(x, frame, ymin) 
        interp_xmax  = np.interp(x, frame, xmax) 
        interp_ymax  = np.interp(x, frame, ymax) 
        interp_frame = np.interp(x, frame, frame) 
    elif style == 'no_interp':
        interp_xmin = xmin
        interp_ymin = ymin
        interp_xmax = xmax
        interp_ymax = ymax
        interp_frame = frame
    elif style == 'cubic': 
        if frame.shape[0] == 1:
            # you can't do cubic interpolation on one point
            interp_xmin = xmin
            interp_ymin = ymin
            interp_xmax = xmax
            interp_ymax = ymax
        else:
            f_xmin = interp1d(frame, xmin)
            f_ymin = interp1d(frame, ymin)
            f_xmax = interp1d(frame, xmax)
            f_ymax = interp1d(frame, ymax)

            interp_xmin  = f_xmin(x) 
            interp_ymin  = f_ymin(x) 
            interp_xmax  = f_xmax(x) 
            interp_ymax  = f_ymax(x) 
            
        interp_frame = np.interp(x, frame, frame) 
    else:
        raise NotImplementedError('not cubic or linear')

    width = interp_xmax - interp_xmin
    height= interp_ymax - interp_ymin

    conf = np.ones(interp_xmin.shape)
    XYZ = np.full((interp_xmin.shape[0], 3), -1)

    out = np.concatenate((np.expand_dims(interp_frame, 1), np.expand_dims(IDs, 1), np.expand_dims(interp_xmin, 1), np.expand_dims(interp_ymin, 1), np.expand_dims(width, 1), np.expand_dims(height, 1), np.expand_dims(conf, 1), XYZ), axis=1)
    #print('expanded',np.expand_dims(IDs, 1).shape)
    #print('interp', np.expand_dims(interp_xmin, 1).shape)
    #out = np.concatenate((np.expand_dims(interp_frame, 1), np.expand_dims(interp_xmin, 1), np.expand_dims(interp_ymin, 1), np.expand_dims(width, 1), np.expand_dims(height, 1), np.expand_dims(conf, 1), XYZ), axis=1)

    print(out.shape)
    return out

def break_tracks(slice_, max_seperation=30):
    """this function takes a slice of a dataframe where all of the annotaitons have
    one ID and return as set/list/something of them split such that each chunk is temporally 
    consistent"""
    # I think it's easiest to just return a list of views
    frames = slice_['frame'].values
    diffs  = np.diff(frames)
    breaks = diffs > max_seperation 
    where = list(np.where(breaks)[0])
    first = [0] + where[:-1]

    output_views = []

    for i, ind in enumerate(where):
        output_views.append(slice_[first[i]: ind])

    if len(where) > 0:
        output_views.append(slice_.ix[where[-1]: len(frames)])
     
    print(output_views)
    return output_views


STYLE = 'cubic'
OUTPUT_DIR = "/home/drussel1/data/readonly/groundtruth/ADL_linear_interpolation"

for i in range(1,21):
    padded_index = '{:02d}'.format(i)
    df = pd.read_csv(DATA_FOLDER.format(i), sep=' ', header=None, names=['frame', 'ID', 'xmin', 'ymin', 'width', 'height', 'conf', 'X', 'Y', 'Z'])
    df.sort_values(by=['ID', 'frame'], inplace=True)
    
    IDs = set(df['ID'].tolist())
    chunk_list = []
    output_list = []
    for id_ in IDs: 
        chunk_list += break_tracks(df[df['ID'] == id_])

    chunk_list = [x for x in chunk_list if len(x) > 0]
    for chunk in chunk_list: 
        output_list.append(interpolate(chunk, style=STYLE))

    full_output = np.concatenate((output_list), axis=0)
    np.savetxt('/home/drussel1/data/readonly/groundtruth/ADL_{}_interpolation/P_{}.txt'.format(STYLE, padded_index), full_output, fmt='%i')
