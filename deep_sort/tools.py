import numpy as np
def safe_crop_ltbr(image, x1, y1, x2, y2):
    assert image is not None
    assert x1 <= x2 and y1 <= y2
    return image[int(y1):int(y2), int(x1):int(x2)].copy()

def ltwh_to_tlbr(bbox): # these boxes are really ltwh
    left, top = bbox[:2]
    bottom = top + bbox[3]
    right  = left + bbox[2]
    return np.array([top, left, bottom, right])

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
    print("tlwh: {}, xyah: {}".format(ltwh_bbox, bbox))
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
    print("xyah: {}, tlwh: {}".format(xyah_bbox, bbox))
    return bbox

def debug_signal_handler(signal, frame):
    import pdb
    pdb.set_trace()

def pdb_on_ctrl_c():
    import signal
    signal.signal(signal.SIGINT, debug_signal_handler)
