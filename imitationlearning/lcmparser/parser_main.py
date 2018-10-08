import lcm

from imitationlearning.lcmparser.gokart_status_event import GokartStatusEvent

if __name__ == '__main__':
    log_path = "/media/ale/dubi_usb1/20181005T135151_1cb189b4.lcm.00"
    log = lcm.EventLog(log_path, 'r')
    for event in log:

        if "gokart.status.get" in event.channel:
            # msg = BinaryBlob.decode(event.data)
            gokart_status_event = GokartStatusEvent(event.data)

            # gokart_status_event = GokartStatusEvent(msg)
            print("steering:", gokart_status_event.value)
