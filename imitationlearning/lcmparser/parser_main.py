import lcm

from imitationlearning.lcmparser.gokart_status_event import GokartStatusEvent
from imitationlearning.lcmparser.seye_aedvs_event import SeyeAeApsEvent

if __name__ == '__main__':
    log_path = "/media/ale/dubi_usb1/20181005T135151_1cb189b4.lcm.00"
    log = lcm.EventLog(log_path, 'r')
    for event in log:

        if "gokart.status.get" in event.channel and False:
            # msg = BinaryBlob.decode(event.data)
            gokart_status_event = GokartStatusEvent(event.data)

        if "seye.overview.aeaps" in event.channel:
            print("reading events")
            events_event = SeyeAeApsEvent(event.data)
