"""
author: az
"""
from objdetection.rgb2events.nets import ssd_net

networks_obj = {  # 'ssd_davisSAE_master': ssd_davisSAE_master_deprecated.SSDNet,
    # 'ssd_davisSAE_devel1': ssd_davisSAE_devel1_deprecated.SSDNet,
    # 'ssd_davisSAE_devel2': ssd_davisSAE_devel2_deprecated.SSDNet,
    'ssd_net': ssd_net.SSDnet
}


def get_network(name):
    """ Get a network object from a name.
    """
    if name not in networks_obj.keys():
        raise ValueError("The requested network model is unavailable!")
    return networks_obj[name]

# todo maybe add a super class of SSD with all common methods and subclasses 
# with specific-only methods
