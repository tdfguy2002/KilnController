#!/usr/bin/env python
import requests
import json
import time
import datetime
import logging

# this monitors your kiln stats every N seconds
# if X checks fail, an alert is sent via ntfy.sh
# install the ntfy app on your phone and subscribe to your topic

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

class Watcher(object):

    def __init__(self,kiln_url,ntfy_topic,bad_check_limit=6,temp_error_limit=10,sleepfor=10):
        self.kiln_url = kiln_url
        self.ntfy_url = "https://ntfy.sh/" + ntfy_topic
        self.bad_check_limit = bad_check_limit
        self.temp_error_limit = temp_error_limit
        self.sleepfor = sleepfor
        self.bad_checks = 0
        self.stats = {}
        self.alarm_sent = False

    def get_stats(self):
        try:
            r = requests.get(self.kiln_url,timeout=1)
            return r.json()
        except requests.exceptions.Timeout:
            log.error("network timeout. check kiln_url and port.")
            return {}
        except requests.exceptions.ConnectionError:
            log.error("network connection error. check kiln_url and port.")
            return {}
        except:
            return {}

    def send_alert(self,msg):
        log.error("sending alert: %s" % msg)
        try:
            requests.post(self.ntfy_url, data=msg, headers={"Title": "Kiln Alert"})
        except:
            pass

    def get_settings(self):
        try:
            settings_url = self.kiln_url.replace('/api/stats', '/api/settings')
            r = requests.get(settings_url, timeout=1)
            return r.json()
        except Exception:
            return {}

    def has_errors(self, tc_error_alerts=True):
        if not self.stats:
            log.error("no data from kiln")
            return True
        if self.stats.get('state') not in ('RUNNING', 'PAUSED'):
            return False
        if 'err' in self.stats:
            if abs(self.stats['err']) > self.temp_error_limit:
                log.error("temp out of whack %0.2f" % self.stats['err'])
                return True
        if tc_error_alerts and self.stats.get('tc_error_pct', 0) > 0:
            log.error("thermocouple read errors: %.1f%%" % self.stats['tc_error_pct'])
            return True
        return False

    def check_alarm(self):
        alarm_temp = self.stats.get('alarm_temp', 0)
        ispoint = self.stats.get('ispoint', self.stats.get('temperature', 0))
        state = self.stats.get('state', '')
        if alarm_temp and alarm_temp > 0:
            if state == 'IDLE' or state == '':
                self.alarm_sent = False
            if not self.alarm_sent and ispoint >= alarm_temp:
                self.send_alert("Kiln reached alarm temperature: %d°" % int(ispoint))
                self.alarm_sent = True

    def run(self):
        log.info("started watching %s" % self.kiln_url)
        while True:
            settings = self.get_settings()

            if not settings.get('watcher_enabled', True):
                log.info("watcher disabled via settings, skipping check")
                time.sleep(self.sleepfor)
                continue

            self.temp_error_limit = settings.get('temp_deviation_limit', self.temp_error_limit)
            tc_error_alerts = settings.get('tc_error_alerts', True)
            ntfy_topic = settings.get('ntfy_topic', None)
            if ntfy_topic:
                self.ntfy_url = "https://ntfy.sh/" + ntfy_topic

            self.stats = self.get_stats()
            if self.has_errors(tc_error_alerts=tc_error_alerts):
                self.bad_checks += 1
            else:
                try:
                    log.info("OK temp=%0.2f target=%0.2f error=%0.2f" % (self.stats['ispoint'], self.stats['setpoint'], self.stats['err']))
                except Exception:
                    pass

            if self.bad_checks >= self.bad_check_limit:
                msg = "error kiln needs help. %s" % json.dumps(self.stats, indent=2, sort_keys=True)
                self.send_alert(msg)
                self.bad_checks = 0

            self.check_alarm()
            time.sleep(self.sleepfor)

if __name__ == "__main__":

    watcher = Watcher(
        kiln_url = "http://localhost:8081/api/stats",
        ntfy_topic = "skutt-kiln",
        bad_check_limit = 6,
        temp_error_limit = 10,
        sleepfor = 10 )

    watcher.run()
