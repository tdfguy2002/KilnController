import threading,logging,json,time,datetime,csv,os
from oven import Oven
log = logging.getLogger(__name__)

RUNS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'runs')

CSV_HEADER = ['timestamp', 'elapsed_s', 'state', 'temperature', 'target',
              'heat', 'heat_rate', 'cost', 'tc_error_pct',
              'pid_p', 'pid_i', 'pid_d', 'pid_err']

class OvenWatcher(threading.Thread):
    def __init__(self,oven):
        self.last_profile = None
        self.last_log = []
        self.started = None
        self.recording = False
        self.observers = []
        self._csv_file = None
        self._csv_writer = None
        threading.Thread.__init__(self)
        self.daemon = True
        self.oven = oven
        self.start()

# FIXME - need to save runs of schedules in near-real-time
# FIXME - this will enable re-start in case of power outage
# FIXME - re-start also requires safety start (pausing at the beginning
# until a temp is reached)
# FIXME - re-start requires a time setting in minutes.  if power has been
# out more than N minutes, don't restart
# FIXME - this should not be done in the Watcher, but in the Oven class

    def run(self):
        while True:
            oven_state = self.oven.get_state()

            # record state for any new clients that join
            if oven_state.get("state") in ("RUNNING", "PAUSED"):
                self.last_log.append(oven_state)
                self._write_csv_row(oven_state)
            else:
                if self.recording:
                    self.recording = False
                    self._close_csv()
            self.notify_all(oven_state)
            time.sleep(self.oven.time_step)

    def lastlog_subset(self,maxpts=50):
        '''send about maxpts from lastlog by skipping unwanted data'''
        totalpts = len(self.last_log)
        if (totalpts <= maxpts):
            return self.last_log
        every_nth = int(totalpts / (maxpts - 1))
        return self.last_log[::every_nth]

    def record(self, profile):
        self.last_profile = profile
        self.last_log = []
        self.started = datetime.datetime.now()
        self.recording = True
        #we just turned on, add first state for nice graph
        self.last_log.append(self.oven.get_state())
        self._open_csv(profile)

    def _open_csv(self, profile):
        try:
            os.makedirs(RUNS_DIR, exist_ok=True)
            safe_name = profile.name.replace(' ', '_').replace('/', '-')
            filename = self.started.strftime('%Y-%m-%d_%H-%M-%S') + '_' + safe_name + '.csv'
            path = os.path.join(RUNS_DIR, filename)
            self._csv_file = open(path, 'w', newline='')
            self._csv_writer = csv.writer(self._csv_file)
            self._csv_writer.writerow(CSV_HEADER)
            self._csv_file.flush()
            log.info("run log started: %s" % path)
        except Exception as e:
            log.error("could not open run log CSV: %s" % e)
            self._csv_file = None
            self._csv_writer = None

    def _write_csv_row(self, state):
        if not self._csv_writer:
            return
        try:
            pidstats = state.get('pidstats') or {}
            self._csv_writer.writerow([
                datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
                round(state.get('runtime', 0), 1),
                state.get('state', ''),
                round(state.get('temperature', 0), 2),
                round(state.get('target', 0), 2),
                state.get('heat', 0),
                round(state.get('heat_rate', 0), 2),
                round(state.get('cost', 0), 4),
                round(state.get('tc_error_pct', 0), 2),
                round(pidstats.get('p', 0), 4),
                round(pidstats.get('i', 0), 4),
                round(pidstats.get('d', 0), 4),
                round(pidstats.get('err', 0), 2),
            ])
            self._csv_file.flush()
        except Exception as e:
            log.error("could not write run log row: %s" % e)

    def _close_csv(self):
        if self._csv_file:
            try:
                self._csv_file.close()
                log.info("run log closed")
            except Exception as e:
                log.error("could not close run log: %s" % e)
            self._csv_file = None
            self._csv_writer = None

    def add_observer(self,observer):
        if self.last_profile:
            p = {
                "name": self.last_profile.name,
                "data": self.last_profile.data,
                "type" : "profile"
            }
        else:
            p = None

        backlog = {
            'type': "backlog",
            'profile': p,
            'log': self.lastlog_subset(),
            #'started': self.started
        }
        print(backlog)
        backlog_json = json.dumps(backlog)
        try:
            print(backlog_json)
            observer.send(backlog_json)
        except:
            log.error("Could not send backlog to new observer")

        self.observers.append(observer)

    def notify_all(self,message):
        message_json = json.dumps(message)
        log.debug("sending to %d clients: %s"%(len(self.observers),message_json))

        dead = []
        for wsock in self.observers:
            if wsock:
                try:
                    wsock.send(message_json)
                except:
                    log.error("could not write to socket %s"%wsock)
                    dead.append(wsock)
            else:
                dead.append(wsock)
        for wsock in dead:
            self.observers.remove(wsock)
