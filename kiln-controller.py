#!/usr/bin/env python

import time
import os
import sys
import logging
import json

import requests
import bottle
import gevent
import geventwebsocket
#from bottle import post, get
from gevent.pywsgi import WSGIServer
from geventwebsocket.handler import WebSocketHandler
from geventwebsocket import WebSocketError

# try/except removed here on purpose so folks can see why things break
import config

logging.basicConfig(level=config.log_level, format=config.log_format)
log = logging.getLogger("kiln-controller")
log.info("Starting kiln controller")

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, script_dir + '/lib/')
profile_path = config.kiln_profiles_directory
settings_path = os.path.join(script_dir, 'settings.json')
runs_path = os.path.join(script_dir, 'runs')

SETTINGS_DEFAULTS = {
    "watcher_enabled": True,
    "tc_error_alerts": True,
    "temp_deviation_limit": 10,
    "ntfy_topic": "skutt-kiln",
    "kwh_rate": config.kwh_rate,
    "currency_type": config.currency_type,
}

def load_settings():
    try:
        with open(settings_path, 'r') as f:
            data = json.load(f)
        merged = dict(SETTINGS_DEFAULTS)
        merged.update(data)
        return merged
    except Exception:
        return dict(SETTINGS_DEFAULTS)

def save_settings(data):
    merged = load_settings()
    merged.update(data)
    with open(settings_path, 'w') as f:
        json.dump(merged, f, indent=2)
    return merged

from oven import SimulatedOven, RealOven, Profile
from ovenWatcher import OvenWatcher

app = bottle.Bottle()

if config.simulate == True:
    log.info("this is a simulation")
    oven = SimulatedOven()
else:
    log.info("this is a real kiln")
    oven = RealOven()
ovenWatcher = OvenWatcher(oven)
# this ovenwatcher is used in the oven class for restarts
oven.set_ovenwatcher(ovenWatcher)

@app.route('/')
def index():
    return bottle.redirect('/picoreflow/index.html')

@app.route('/state')
def state():
    return bottle.redirect('/picoreflow/state.html')

@app.get('/log')
def handle_log():
    log_path = '/var/log/kiln-controller.log'
    per_page = 200
    page = int(bottle.request.query.get('page', 1))

    try:
        with open(log_path, 'r') as f:
            all_lines = f.readlines()
        total_lines = len(all_lines)
        total_pages = max(1, -(-total_lines // per_page))  # ceiling division
        page = max(1, min(page, total_pages))
        # page 1 = last per_page lines, page 2 = lines before that, etc.
        end   = total_lines - (page - 1) * per_page
        start = max(0, end - per_page)
        content = ''.join(all_lines[start:end])
        info = 'Lines %d–%d of %d' % (start + 1, end, total_lines)
    except Exception as e:
        content = 'Could not read log file: %s' % str(e)
        total_pages = 1
        info = ''

    newer = ('<a href="/log?page=%d">&laquo; Newer</a>' % (page - 1)) if page > 1 else '<span style="opacity:0.3">&laquo; Newer</span>'
    older = ('<a href="/log?page=%d">Older &raquo;</a>' % (page + 1)) if page < total_pages else '<span style="opacity:0.3">Older &raquo;</span>'

    bottle.response.content_type = 'text/html; charset=utf-8'
    return '''<!DOCTYPE html>
<html><head><title>Kiln Log</title>
<style>
  body {{ background:#0d1018; color:#c8d8e8; font-family:monospace; font-size:13px; padding:20px; margin:0; }}
  pre {{ white-space:pre-wrap; word-break:break-all; }}
  h2 {{ color:#ee6c20; font-family:sans-serif; margin-top:0; }}
  .nav {{ font-family:sans-serif; font-size:14px; margin-bottom:16px; display:flex; gap:24px; align-items:center; }}
  a {{ color:#ee6c20; text-decoration:none; }} a:hover {{ text-decoration:underline; }}
  .info {{ color:#5a6a7a; font-size:12px; }}
</style></head>
<body>
<h2>Kiln Controller Log</h2>
<div class="nav">{newer} {older} <span class="info">Page {page} of {total_pages} &nbsp;|&nbsp; {info}</span></div>
<pre>{content}</pre>
<div class="nav">{newer} {older}</div>
</body></html>'''.format(newer=newer, older=older, page=page, total_pages=total_pages, info=info, content=content)


@app.get('/api/test_notification')
def handle_test_notification():
    s = load_settings()
    topic = s.get('ntfy_topic', '')
    if not topic:
        return json.dumps({"success": False, "error": "No ntfy topic configured"})
    try:
        requests.post("https://ntfy.sh/" + topic, data="Test notification from Kiln Controller", headers={"Title": "Kiln Test"}, timeout=5)
        return json.dumps({"success": True})
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})

@app.get('/api/settings')
def handle_get_settings():
    bottle.response.content_type = 'application/json'
    return json.dumps(load_settings())

@app.post('/api/settings')
def handle_post_settings():
    try:
        data = bottle.request.json
        if not isinstance(data, dict):
            bottle.response.status = 400
            return json.dumps({"success": False, "error": "expected JSON object"})
        updated = save_settings(data)
        log.info("settings updated: %s" % json.dumps(data))
        return json.dumps({"success": True, "settings": updated})
    except Exception as e:
        bottle.response.status = 500
        return json.dumps({"success": False, "error": str(e)})

@app.get('/api/runs')
def handle_list_runs():
    bottle.response.content_type = 'application/json'
    try:
        files = sorted([f for f in os.listdir(runs_path) if f.endswith('.csv')], reverse=True)
    except Exception:
        files = []
    return json.dumps(files)

@app.get('/api/runs/<filename>')
def handle_download_run(filename):
    if '..' in filename or '/' in filename:
        bottle.abort(400, 'Invalid filename')
    return bottle.static_file(filename, root=runs_path, download=filename)

@app.get('/api/stats')
def handle_api_stats():
    log.info("/api/stats command received")
    state = oven.get_state()
    if hasattr(oven,'pid') and hasattr(oven.pid,'pidstats'):
        state.update(oven.pid.pidstats)
    return json.dumps(state)


def delayed_run(oven, start_at, profile, watcher):
    delay = start_at - time.time()
    if delay > 0:
        gevent.sleep(delay)
    if oven.state != 'SCHEDULED':
        return
    oven.scheduled_start = 0
    oven.run_profile(profile)
    watcher.record(profile)


@app.post('/api')
def handle_api():
    log.info("/api is alive")


    # run a kiln schedule
    if bottle.request.json['cmd'] == 'run':
        wanted = bottle.request.json['profile']
        log.info('api requested run of profile = %s' % wanted)

        # start at a specific minute in the schedule
        # for restarting and skipping over early parts of a schedule
        startat = 0;      
        if 'startat' in bottle.request.json:
            startat = bottle.request.json['startat']

        #Shut off seek if start time has been set
        allow_seek = True
        if startat > 0:
            allow_seek = False

        # get the wanted profile/kiln schedule
        profile = find_profile(wanted)
        if profile is None:
            return { "success" : False, "error" : "profile %s not found" % wanted }

        # FIXME juggling of json should happen in the Profile class
        profile_json = json.dumps(profile)
        profile = Profile(profile_json)
        oven.run_profile(profile, startat=startat, allow_seek=allow_seek)
        ovenWatcher.record(profile)

    if bottle.request.json['cmd'] == 'pause':
        log.info("api pause command received")
        oven.state = 'PAUSED'

    if bottle.request.json['cmd'] == 'resume':
        log.info("api resume command received")
        oven.state = 'RUNNING'

    if bottle.request.json['cmd'] == 'stop':
        log.info("api stop command received")
        oven.abort_run()

    if bottle.request.json['cmd'] == 'schedule':
        log.info("api schedule command received")
        wanted = bottle.request.json['profile']
        start_at = float(bottle.request.json['start_at'])
        profile = find_profile(wanted)
        if profile is None:
            return { "success": False, "error": "profile %s not found" % wanted }
        oven.scheduled_start = start_at
        oven.state = 'SCHEDULED'
        profile_json = json.dumps(profile)
        scheduled_profile = Profile(profile_json)
        gevent.spawn(delayed_run, oven, start_at, scheduled_profile, ovenWatcher)

    if bottle.request.json['cmd'] == 'cancel_schedule':
        log.info("api cancel_schedule command received")
        oven.scheduled_start = 0
        oven.state = 'IDLE'

    if bottle.request.json['cmd'] == 'alarm':
        log.info("api alarm command received")
        oven.alarm_temp = float(bottle.request.json.get('temp', 0))
        log.info("alarm set to %s" % oven.alarm_temp)

    if bottle.request.json['cmd'] == 'memo':
        log.info("api memo command received")
        memo = bottle.request.json['memo']
        log.info("memo=%s" % (memo))

    # get stats during a run
    if bottle.request.json['cmd'] == 'stats':
        log.info("api stats command received")
        if hasattr(oven,'pid'):
            if hasattr(oven.pid,'pidstats'):
                return json.dumps(oven.pid.pidstats)

    return { "success" : True }

def find_profile(wanted):
    '''
    given a wanted profile name, find it and return the parsed
    json profile object or None.
    '''
    #load all profiles from disk
    profiles = get_profiles()
    json_profiles = json.loads(profiles)

    # find the wanted profile
    for profile in json_profiles:
        if profile['name'] == wanted:
            return profile
    return None

@app.route('/picoreflow/:filename#.*#')
def send_static(filename):
    log.debug("serving %s" % filename)
    return bottle.static_file(filename, root=os.path.join(os.path.dirname(os.path.realpath(sys.argv[0])), "public"))


def get_websocket_from_request():
    env = bottle.request.environ
    wsock = env.get('wsgi.websocket')
    if not wsock:
        bottle.abort(400, 'Expected WebSocket request.')
    return wsock


@app.route('/control')
def handle_control():
    wsock = get_websocket_from_request()
    log.info("websocket (control) opened")
    while True:
        try:
            message = wsock.receive()
            if message:
                log.info("Received (control): %s" % message)
                msgdict = json.loads(message)
                if msgdict.get("cmd") == "RUN":
                    log.info("RUN command received")
                    profile_obj = msgdict.get('profile')
                    if profile_obj:
                        profile_json = json.dumps(profile_obj)
                        profile = Profile(profile_json)
                        oven.run_profile(profile)
                        ovenWatcher.record(profile)
                elif msgdict.get("cmd") == "SIMULATE":
                    log.info("SIMULATE command received")
                    #profile_obj = msgdict.get('profile')
                    #if profile_obj:
                    #    profile_json = json.dumps(profile_obj)
                    #    profile = Profile(profile_json)
                    #simulated_oven = Oven(simulate=True, time_step=0.05)
                    #simulation_watcher = OvenWatcher(simulated_oven)
                    #simulation_watcher.add_observer(wsock)
                    #simulated_oven.run_profile(profile)
                    #simulation_watcher.record(profile)
                elif msgdict.get("cmd") == "STOP":
                    log.info("Stop command received")
                    oven.abort_run()
            time.sleep(1)
        except WebSocketError as e:
            log.error(e)
            break
    log.info("websocket (control) closed")


@app.route('/storage')
def handle_storage():
    wsock = get_websocket_from_request()
    log.info("websocket (storage) opened")
    while True:
        try:
            message = wsock.receive()
            if not message:
                break
            log.debug("websocket (storage) received: %s" % message)

            try:
                msgdict = json.loads(message)
            except:
                msgdict = {}

            if message == "GET":
                log.info("GET command received")
                wsock.send(get_profiles())
            elif msgdict.get("cmd") == "DELETE":
                log.info("DELETE command received")
                profile_obj = msgdict.get('profile')
                if delete_profile(profile_obj):
                  msgdict["resp"] = "OK"
                wsock.send(json.dumps(msgdict))
                #wsock.send(get_profiles())
            elif msgdict.get("cmd") == "PUT":
                log.info("PUT command received")
                profile_obj = msgdict.get('profile')
                #force = msgdict.get('force', False)
                force = True
                if profile_obj:
                    #del msgdict["cmd"]
                    if save_profile(profile_obj, force):
                        msgdict["resp"] = "OK"
                    else:
                        msgdict["resp"] = "FAIL"
                    log.debug("websocket (storage) sent: %s" % message)

                    wsock.send(json.dumps(msgdict))
                    wsock.send(get_profiles())
            time.sleep(1) 
        except WebSocketError:
            break
    log.info("websocket (storage) closed")


@app.route('/config')
def handle_config():
    wsock = get_websocket_from_request()
    log.info("websocket (config) opened")
    while True:
        try:
            message = wsock.receive()
            wsock.send(get_config())
        except WebSocketError:
            break
        time.sleep(1)
    log.info("websocket (config) closed")


@app.route('/status')
def handle_status():
    wsock = get_websocket_from_request()
    ovenWatcher.add_observer(wsock)
    log.info("websocket (status) opened")
    while True:
        try:
            message = wsock.receive()
            wsock.send("Your message was: %r" % message)
        except WebSocketError:
            break
        time.sleep(1)
    log.info("websocket (status) closed")


def get_profiles():
    try:
        profile_files = os.listdir(profile_path)
    except:
        profile_files = []
    profiles = []
    for filename in profile_files:
        with open(os.path.join(profile_path, filename), 'r') as f:
            profiles.append(json.load(f))
    profiles = normalize_temp_units(profiles)
    return json.dumps(profiles)


def save_profile(profile, force=False):
    profile=add_temp_units(profile)
    profile_json = json.dumps(profile)
    filename = profile['name']+".json"
    filepath = os.path.join(profile_path, filename)
    if not force and os.path.exists(filepath):
        log.error("Could not write, %s already exists" % filepath)
        return False
    with open(filepath, 'w+') as f:
        f.write(profile_json)
        f.close()
    log.info("Wrote %s" % filepath)
    return True

def add_temp_units(profile):
    """
    always store the temperature in degrees c
    this way folks can share profiles
    """
    if "temp_units" in profile:
        return profile
    profile['temp_units']="c"
    if config.temp_scale=="c":
        return profile
    if config.temp_scale=="f":
        profile=convert_to_c(profile);
        return profile

def convert_to_c(profile):
    newdata=[]
    for (secs,temp) in profile["data"]:
        temp = (5/9)*(temp-32)
        newdata.append((secs,temp))
    profile["data"]=newdata
    return profile

def convert_to_f(profile):
    newdata=[]
    for (secs,temp) in profile["data"]:
        temp = ((9/5)*temp)+32
        newdata.append((secs,temp))
    profile["data"]=newdata
    return profile

def normalize_temp_units(profiles):
    normalized = []
    for profile in profiles:
        if "temp_units" in profile:
            if config.temp_scale == "f" and profile["temp_units"] == "c": 
                profile = convert_to_f(profile)
                profile["temp_units"] = "f"
        normalized.append(profile)
    return normalized

def delete_profile(profile):
    profile_json = json.dumps(profile)
    filename = profile['name']+".json"
    filepath = os.path.join(profile_path, filename)
    os.remove(filepath)
    log.info("Deleted %s" % filepath)
    return True

def get_config():
    s = load_settings()
    return json.dumps({"temp_scale": config.temp_scale,
        "time_scale_slope": config.time_scale_slope,
        "time_scale_profile": config.time_scale_profile,
        "kwh_rate": s.get("kwh_rate", config.kwh_rate),
        "currency_type": s.get("currency_type", config.currency_type)})

def main():
    ip = "0.0.0.0"
    port = config.listening_port
    log.info("listening on %s:%d" % (ip, port))

    server = WSGIServer((ip, port), app,
                        handler_class=WebSocketHandler)
    server.serve_forever()


if __name__ == "__main__":
    main()
