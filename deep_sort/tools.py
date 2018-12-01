import numpy as np
def safe_crop_ltbr(image, x1, y1, x2, y2):
    """
    Returns a crop of an image based on the image and an [left, top, right, bottom] bounding box
    >>> safe_crop_ltbr(np.zeros((720, 1080, 3)), 0, 10, 100, 200).shape
    (190, 100, 3)
    >>> safe_crop_ltbr(np.zeros((720, 1080, 3)), 100, 100, 20, 200)
    Traceback (most recent call last):
    ValueError: {x,y}1 should be less than {x,y}2
    """
    if image is None or len(image.shape) != 3:
        raise ValueError("Image should be a 3d array")
    if x1 >= x2 or y1 >= y2:
        raise ValueError("{x,y}1 should be less than {x,y}2")
    x1 = int(np.clip(x1, 0, image.shape[1]))
    x2 = int(np.clip(x2, 0, image.shape[1]))
    y1 = int(np.clip(y1, 0, image.shape[0]))
    y2 = int(np.clip(y2, 0, image.shape[0]))
    return image[y1:y2, x1:x2].copy()

def ltwh_to_tlbr(bbox): # these boxes are really ltwh
    left, top = bbox[:2]
    bottom = top + bbox[3]
    right  = left + bbox[2]
    return np.array([top, left, bottom, right])

def tlbr_to_ltrb(bbox):
    """
    >>> ltbr = np.asarray([10, 20, 101, 122])
    ... assert np.array_equal(tlbr_to_ltrb(tlbr_to_ltrb(ltbr)), ltbr)
    """
    return ltrb_to_tlbr(bbox) # it's the same as the other way



def ltrb_to_tlbr(bbox):
    #this composed with itself is an identity mapping
    return np.asarray([bbox[1], bbox[0], bbox[3], bbox[2]]) 

def tlbr_to_ltwh(bbox):
    """
    >>> ltwh = tlbr_to_ltwh([10, 20, 101, 122])
    ... print(ltwh_to_tlbr(ltwh))
    """
    top, left = bbox[:2]
    width     = bbox[3] - left
    height    = bbox[2] - top
    return np.asarray([left, top, width, height])

def ltwh_to_xyah(ltwh_bbox):
    """
    >>> ltwh_bbox = [123, 142, 45, 34]
    ... xyah_bbox = ltwh_to_xyah( ltwh_bbox )
    ... assert xyah_to_ltwh(xyah_bbox) == ltwh_bbox
    """
    #"""convert bounding box to format `(center x, center y, aspect ratio,
    #height)`, where the aspect ratio is `width / height`.
    #"""
    #ret = self.tlwh.copy()
    #ret[:2] += ret[2:] / 2
    #ret[2] /= ret[3]
    #return ret
    #note the a is aspect ratio, not the area
    bbox = ltwh_bbox.copy() 
    bbox[0] += ltwh_bbox[2] / 2
    bbox[1] += ltwh_bbox[3] / 2
    bbox[2] /= ltwh_bbox[3]
    return bbox

def xyah_to_ltwh(xyah_bbox):
    #"""convert bounding box from format `(center x, center y, aspect ratio,
    #height)`, where the aspect ratio is `width / height` to left, top, width, height
    #"""
    #ret = self.tlwh.copy()
    #ret[:2] += ret[2:] / 2
    #ret[2] /= ret[3]
    #return ret
    #note the a is aspect ratio, not the area
    bbox = xyah_bbox.copy() 
    bbox[2] = xyah_bbox[2] * xyah_bbox[3] # height * width / height
    bbox[0] -= bbox[2] / 2
    bbox[1] -= bbox[3] / 2
    return bbox

def debug_signal_handler(signal, frame):
    import pdb
    pdb.set_trace()

def pdb_on_ctrl_c():
    import signal
    signal.signal(signal.SIGINT, debug_signal_handler)
