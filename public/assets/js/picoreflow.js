var state = "IDLE";
var state_last = "";
var graph = [ 'profile', 'live'];
var points = [];
var profiles = [];
var time_mode = 0;
var selected_profile = 0;
var selected_profile_name = 'cone-05-long-bisque.json';
var temp_scale = "c";
var time_scale_slope = "s";
var time_scale_profile = "h";
var time_scale_long = "Seconds";
var temp_scale_display = "C";
var kwh_rate = 0.26;
var currency_type = "EUR";

var protocol = 'ws:';
if (window.location.protocol == 'https:') {
    protocol = 'wss:';
}
var host = "" + protocol + "//" + window.location.hostname + ":" + window.location.port;
var ws_status = new WebSocket(host+"/status");
var ws_control = new WebSocket(host+"/control");
var ws_config = new WebSocket(host+"/config");
var ws_storage = new WebSocket(host+"/storage");


if(window.webkitRequestAnimationFrame) window.requestAnimationFrame = window.webkitRequestAnimationFrame;

graph.profile =
{
    label: "Profile",
    data: [],
    points: { show: false },
    color: "#75890c",
    draggable: false
};

graph.live =
{
    label: "Live",
    data: [],
    points: { show: false },
    color: "#d8d3c5",
    draggable: false
};


function updateProfile(id)
{
    selected_profile = id;
    selected_profile_name = profiles[id].name;
    var job_seconds = profiles[id].data.length === 0 ? 0 : parseInt(profiles[id].data[profiles[id].data.length-1][0]);
    var kwh = (3850*job_seconds/3600/1000).toFixed(2);
    var cost =  (kwh*kwh_rate).toFixed(2);
    var job_time = new Date(job_seconds * 1000).toISOString().substr(11, 8);
    $('#sel_prof').html(profiles[id].name);
    $('#sel_prof_eta').html(job_time);
    $('#sel_prof_cost').html(kwh + ' kWh ('+ currency_type +': '+ cost +')');
    graph.profile.data = profiles[id].data;
    graph.plot = $.plot("#graph_container", [ graph.profile, graph.live ] , getOptions());
    updateScheduleDisplay(profiles[id].data);
}

function updateScheduleDisplay(data)
{
    var tbody = $('#schedule_tbody');
    tbody.empty();

    if (!data || data.length === 0) {
        tbody.append('<tr><td colspan="5" style="text-align:center;">No waypoints</td></tr>');
        return;
    }

    for (var i = 0; i < data.length; i++) {
        var secs = parseInt(data[i][0]);
        var temp = data[i][1];

        var hrs  = Math.floor(secs / 3600);
        var mins = Math.floor((secs % 3600) / 60);
        var timeStr = hrs + ':' + (mins < 10 ? '0' : '') + mins;

        var rateStr = '&mdash;';
        var rateClass = '';
        var segStr = '&mdash;';
        if (i > 0) {
            var dt = data[i][0] - data[i-1][0];
            var dtemp = data[i][1] - data[i-1][1];
            if (dt > 0) {
                segStr = Math.round(dt / 60) + ' min';
                var rate = Math.round(dtemp / dt * 3600);
                if (rate > 0) {
                    rateStr = '+' + rate + '&deg;/hr';
                    rateClass = 'rate-up';
                } else if (rate < 0) {
                    rateStr = rate + '&deg;/hr';
                    rateClass = 'rate-down';
                } else {
                    rateStr = 'hold';
                    rateClass = 'rate-hold';
                }
            }
        }

        var row = '<tr>' +
            '<td class="sched-num">' + (i + 1) + '</td>' +
            '<td class="sched-time">' + timeStr + '</td>' +
            '<td class="sched-temp">' + Math.round(temp) + '&deg;</td>' +
            '<td class="sched-seg">' + segStr + '</td>' +
            '<td class="sched-rate ' + rateClass + '">' + rateStr + '</td>' +
            '</tr>';
        tbody.append(row);
    }
}

function deleteProfile()
{
    var profile = { "type": "profile", "data": "", "name": selected_profile_name };
    var delete_struct = { "cmd": "DELETE", "profile": profile };

    var delete_cmd = JSON.stringify(delete_struct);
    console.log("Delete profile:" + selected_profile_name);

    ws_storage.send(delete_cmd);

    ws_storage.send('GET');
    selected_profile_name = profiles[0].name;

    state="IDLE";
    $('#edit').hide();
    $('#profile_selector').show();
    $('#btn_controls').show();
    $('#status').slideDown();
    $('#profile_table').slideUp();
    $('#e2').select2('val', 0);
    graph.profile.points.show = false;
    graph.profile.draggable = false;
    graph.plot = $.plot("#graph_container", [ graph.profile, graph.live ], getOptions());
}


function updateProgress(percentage)
{
    if(state=="RUNNING")
    {
        if(percentage > 100) percentage = 100;
        $('#progressBar').css('width', percentage+'%');
        if(percentage>5) $('#progressBar').html(parseInt(percentage)+'%');
    }
    else
    {
        $('#progressBar').css('width', 0+'%');
        $('#progressBar').html('');
    }
}

function updateProfileTable()
{
    var html = '<h3>Schedule Points</h3><div class="table-responsive" style="scroll: none"><table class="table table-striped">';
        html += '<tr><th style="width: 50px">#</th><th>Target Time in ' + time_scale_long+ '</th><th>Target Temperature in °'+temp_scale_display+'</th><th></th></tr>';

    for(var i=0; i<graph.profile.data.length;i++)
    {
        html += '<tr><td><h4>' + (i+1) + '</h4></td>';
        html += '<td><input type="text" class="form-control" id="profiletable-0-'+i+'" value="'+ timeProfileFormatter(graph.profile.data[i][0],true) + '" style="width: 60px" /></td>';
        html += '<td><input type="text" class="form-control" id="profiletable-1-'+i+'" value="'+ graph.profile.data[i][1] + '" style="width: 60px" /></td>';
        html += '<td>&nbsp;</td></tr>';
    }

    html += '</table></div>';

    $('#profile_table').html(html);

    //Link table to graph
    $(".form-control").change(function(e)
        {
            var id = $(this)[0].id; //e.currentTarget.attributes.id
            var value = parseInt($(this)[0].value);
            var fields = id.split("-");
            var col = parseInt(fields[1]);
            var row = parseInt(fields[2]);

            if (graph.profile.data.length > 0) {
            if (col == 0) {
                graph.profile.data[row][col] = timeProfileFormatter(value,false);
            }
            else {
                graph.profile.data[row][col] = value;
            }

            graph.plot = $.plot("#graph_container", [ graph.profile, graph.live ], getOptions());
            }
            updateProfileTable();

        });
}

function timeProfileFormatter(val, down) {
    var rval = val
    switch(time_scale_profile){
        case "m":
            if (down) {rval = val / 60;} else {rval = val * 60;}
            break;
        case "h":
            if (down) {rval = val / 3600;} else {rval = val * 3600;}
            break;
    }
    return Math.round(rval);
}

function formatDPS(val) {
    var tval = val;
    if (time_scale_slope == "m") {
        tval = val * 60;
    }
    if (time_scale_slope == "h") {
        tval = (val * 60) * 60;
    }
    return Math.round(tval);
}

function hazardTemp(){

    if (temp_scale == "f") {
        return (1500 * 9 / 5) + 32
    }
    else {
        return 1500
    }
}

function timeTickFormatter(val,axis)
{
// hours
if(axis.max>3600) {
  //var hours = Math.floor(val / (3600));
  //return hours;
  return Math.floor(val/3600);
  }

// minutes
if(axis.max<=3600) {
  return Math.floor(val/60);
  }

// seconds
if(axis.max<=60) {
  return val;
  }
}

function runTask()
{
    var cmd =
    {
        "cmd": "RUN",
        "profile": profiles[selected_profile]
    }

    graph.live.data = [];
    graph.plot = $.plot("#graph_container", [ graph.profile, graph.live ] , getOptions());

    ws_control.send(JSON.stringify(cmd));

}

function runTaskSimulation()
{
    var cmd =
    {
        "cmd": "SIMULATE",
        "profile": profiles[selected_profile]
    }

    graph.live.data = [];
    graph.plot = $.plot("#graph_container", [ graph.profile, graph.live ] , getOptions());

    ws_control.send(JSON.stringify(cmd));

}


function abortTask()
{
    var cmd = {"cmd": "STOP"};
    ws_control.send(JSON.stringify(cmd));
}

function enterNewMode()
{
    state="EDIT"
    $('#status').slideUp();
    $('#edit').show();
    $('#profile_selector').hide();
    $('#btn_controls').hide();
    $('#form_profile_name').attr('value', '');
    $('#form_profile_name').attr('placeholder', 'Please enter a name');
    graph.profile.points.show = true;
    graph.profile.draggable = true;
    graph.profile.data = [];
    graph.plot = $.plot("#graph_container", [ graph.profile, graph.live ], getOptions());
    updateProfileTable();
}

function enterEditMode()
{
    state="EDIT"
    $('#status').slideUp();
    $('#edit').show();
    $('#profile_selector').hide();
    $('#btn_controls').hide();
    console.log(profiles);
    $('#form_profile_name').val(profiles[selected_profile].name);
    graph.profile.points.show = true;
    graph.profile.draggable = true;
    graph.plot = $.plot("#graph_container", [ graph.profile, graph.live ], getOptions());
    updateProfileTable();
    toggleTable();
}

function leaveEditMode()
{
    selected_profile_name = $('#form_profile_name').val();
    ws_storage.send('GET');
    state="IDLE";
    $('#edit').hide();
    $('#profile_selector').show();
    $('#btn_controls').show();
    $('#status').slideDown();
    $('#profile_table').slideUp();
    graph.profile.points.show = false;
    graph.profile.draggable = false;
    graph.plot = $.plot("#graph_container", [ graph.profile, graph.live ], getOptions());
}

function newPoint()
{
    if(graph.profile.data.length > 0)
    {
        var pointx = parseInt(graph.profile.data[graph.profile.data.length-1][0])+15;
    }
    else
    {
        var pointx = 0;
    }
    graph.profile.data.push([pointx, Math.floor((Math.random()*230)+25)]);
    graph.plot = $.plot("#graph_container", [ graph.profile, graph.live ], getOptions());
    updateProfileTable();
}

function delPoint()
{
    graph.profile.data.splice(-1,1)
    graph.plot = $.plot("#graph_container", [ graph.profile, graph.live ], getOptions());
    updateProfileTable();
}

function toggleTable()
{
    if($('#profile_table').css('display') == 'none')
    {
        $('#profile_table').slideDown();
    }
    else
    {
        $('#profile_table').slideUp();
    }
}

function saveProfile()
{
    name = $('#form_profile_name').val();
    var rawdata = graph.plot.getData()[0].data
    var data = [];
    var last = -1;

    for(var i=0; i<rawdata.length;i++)
    {
        if(rawdata[i][0] > last)
        {
          data.push([rawdata[i][0], rawdata[i][1]]);
        }
        else
        {
          $.bootstrapGrowl("<span class=\"glyphicon glyphicon-exclamation-sign\"></span> <b>ERROR 88:</b><br/>An oven is not a time-machine", {
            ele: 'body', // which element to append to
            type: 'alert', // (null, 'info', 'error', 'success')
            offset: {from: 'top', amount: 250}, // 'top', or 'bottom'
            align: 'center', // ('left', 'right', or 'center')
            width: 385, // (integer, or 'auto')
            delay: 5000,
            allow_dismiss: true,
            stackup_spacing: 10 // spacing between consecutively stacked growls.
          });

          return false;
        }

        last = rawdata[i][0];
    }

    var profile = { "type": "profile", "data": data, "name": name }
    var put = { "cmd": "PUT", "profile": profile }

    var put_cmd = JSON.stringify(put);

    ws_storage.send(put_cmd);

    leaveEditMode();
}

function get_tick_size() {
//switch(time_scale_profile){
//  case "s":
//    return 1;
//  case "m":
//    return 60;
//  case "h":
//    return 3600;
//  }
return 3600;
}

function getOptions()
{

  var options =
  {

    series:
    {
        lines:
        {
            show: true
        },

        points:
        {
            show: true,
            radius: 5,
            symbol: "circle"
        },

        shadowSize: 3

    },

	xaxis:
    {
      min: 0,
      tickColor: 'rgba(216, 211, 197, 0.2)',
      tickFormatter: timeTickFormatter,
      tickSize: get_tick_size(),
      font:
      {
        size: 14,
        lineHeight: 14,        weight: "normal",
        family: "Digi",
        variant: "small-caps",
        color: "rgba(216, 211, 197, 0.85)"
      }
	},

	yaxis:
    {
      min: 0,
      tickDecimals: 0,
      draggable: false,
      tickColor: 'rgba(216, 211, 197, 0.2)',
      font:
      {
        size: 14,
        lineHeight: 14,
        weight: "normal",
        family: "Digi",
        variant: "small-caps",
        color: "rgba(216, 211, 197, 0.85)"
      }
	},

	grid:
    {
	  color: 'rgba(216, 211, 197, 0.55)',
      borderWidth: 1,
      labelMargin: 10,
      mouseActiveRadius: 50
	},

    legend:
    {
      show: false
    }
  }

  return options;

}



$(document).ready(function()
{

    if(!("WebSocket" in window))
    {
        $('#chatLog, input, button, #examples').fadeOut("fast");
        $('<p>Oh no, you need a browser that supports WebSockets. How about <a href="http://www.google.com/chrome">Google Chrome</a>?</p>').appendTo('#container');
    }
    else
    {

        // Status Socket ////////////////////////////////

        ws_status.onopen = function()
        {
            console.log("Status Socket has been opened");

//            $.bootstrapGrowl("<span class=\"glyphicon glyphicon-exclamation-sign\"></span>Getting data from server",
//            {
//            ele: 'body', // which element to append to
//            type: 'success', // (null, 'info', 'error', 'success')
//            offset: {from: 'top', amount: 250}, // 'top', or 'bottom'
//            align: 'center', // ('left', 'right', or 'center')
//            width: 385, // (integer, or 'auto')
//            delay: 2500,
//            allow_dismiss: true,
//            stackup_spacing: 10 // spacing between consecutively stacked growls.
//            });
        };

        ws_status.onclose = function()
        {
            $.bootstrapGrowl("<span class=\"glyphicon glyphicon-exclamation-sign\"></span> <b>ERROR 1:</b><br/>Status Websocket not available", {
            ele: 'body', // which element to append to
            type: 'error', // (null, 'info', 'error', 'success')
            offset: {from: 'top', amount: 250}, // 'top', or 'bottom'
            align: 'center', // ('left', 'right', or 'center')
            width: 385, // (integer, or 'auto')
            delay: 5000,
            allow_dismiss: true,
            stackup_spacing: 10 // spacing between consecutively stacked growls.
          });
        };

        ws_status.onmessage = function(e)
        {
            x = JSON.parse(e.data);
            if (x.type == "backlog")
            {
                if (x.profile)
                {
                    selected_profile_name = x.profile.name;
                    $.each(profiles,  function(i,v) {
                        if(v.name == x.profile.name) {
                            updateProfile(i);
                            $('#e2').select2('val', i);
                        }
                    });
                }

                $.each(x.log, function(i,v) {
                    graph.live.data.push([v.runtime, v.temperature]);
                    graph.plot = $.plot("#graph_container", [ graph.profile, graph.live ] , getOptions());
                });
            }

            if(state!="EDIT")
            {
                state = x.state;
                if (state!=state_last)
                {
                    if(state_last == "RUNNING" && state != "PAUSED" )
                    {
			console.log(state);
                        $('#target_temp').html('---');
                        updateProgress(0);
                        $.bootstrapGrowl("<span class=\"glyphicon glyphicon-exclamation-sign\"></span> <b>Run completed</b>", {
                        ele: 'body', // which element to append to
                        type: 'success', // (null, 'info', 'error', 'success')
                        offset: {from: 'top', amount: 250}, // 'top', or 'bottom'
                        align: 'center', // ('left', 'right', or 'center')
                        width: 385, // (integer, or 'auto')
                        delay: 0,
                        allow_dismiss: true,
                        stackup_spacing: 10 // spacing between consecutively stacked growls.
                        });
                    }
                }

                if(state=="RUNNING")
                {
                    $("#nav_start").hide();
                    $("#nav_stop").show();

                    graph.live.data.push([x.runtime, x.temperature]);
                    graph.plot = $.plot("#graph_container", [ graph.profile, graph.live ] , getOptions());

                    left = parseInt(x.totaltime-x.runtime);
                    eta = new Date(left * 1000).toISOString().substr(11, 8);

                    updateProgress(parseFloat(x.runtime)/parseFloat(x.totaltime)*100);
                    $('#state').html(eta);
                    $('#target_temp').html(parseInt(x.target));
                    $('#cost').html(x.currency_type + parseFloat(x.cost).toFixed(2));
                  


                }
                else
                {
                    $("#nav_start").show();
                    $("#nav_stop").hide();
                    $('#state').html('<p class="ds-text">'+state+'</p>');
                }

                $('#act_temp').html(parseInt(x.temperature));
                heat_rate = parseInt(x.heat_rate)
                if (heat_rate > 9999) { heat_rate = 9999; }
                if (heat_rate < -9999) { heat_rate = -9999; }
                $('#heat_rate').html(heat_rate);
                if (x.heat > 0.5) { $('#heat').addClass("ds-led-heat-active"); } else { $('#heat').removeClass("ds-led-heat-active"); }
                if (x.cool > 0.5) { $('#cool').addClass("ds-led-cool-active"); } else { $('#cool').removeClass("ds-led-cool-active"); }
                if (x.air > 0.5) { $('#air').addClass("ds-led-air-active"); } else { $('#air').removeClass("ds-led-air-active"); }
                if (x.temperature > hazardTemp()) { $('#hazard').addClass("ds-led-hazard-active"); } else { $('#hazard').removeClass("ds-led-hazard-active"); }
                if ((x.door == "OPEN") || (x.door == "UNKNOWN")) { $('#door').addClass("ds-led-door-open"); } else { $('#door').removeClass("ds-led-door-open"); }

                state_last = state;

            }
        };

        // Config Socket /////////////////////////////////

        ws_config.onopen = function()
        {
            ws_config.send('GET');
        };

        ws_config.onmessage = function(e)
        {
            console.log (e.data);
            x = JSON.parse(e.data);
            temp_scale = x.temp_scale;
            time_scale_slope = x.time_scale_slope;
            time_scale_profile = x.time_scale_profile;
            kwh_rate = x.kwh_rate;
            currency_type = x.currency_type;

            if (temp_scale == "c") {temp_scale_display = "C";} else {temp_scale_display = "F";}


            $('#act_temp_scale').html('º'+temp_scale_display);
            $('#target_temp_scale').html('º'+temp_scale_display);
            $('#heat_rate_temp_scale').html('º'+temp_scale_display);

            switch(time_scale_profile){
                case "s":
                    time_scale_long = "Seconds";
                    break;
                case "m":
                    time_scale_long = "Minutes";
                    break;
                case "h":
                    time_scale_long = "Hours";
                    break;
            }

        }

        // Control Socket ////////////////////////////////

        ws_control.onopen = function()
        {

        };

        ws_control.onmessage = function(e)
        {
            //Data from Simulation
            console.log ("control socket has been opened")
            console.log (e.data);
            x = JSON.parse(e.data);
            graph.live.data.push([x.runtime, x.temperature]);
            graph.plot = $.plot("#graph_container", [ graph.profile, graph.live ] , getOptions());

        }

        // Storage Socket ///////////////////////////////

        ws_storage.onopen = function()
        {
            ws_storage.send('GET');
        };


        ws_storage.onmessage = function(e)
        {
            message = JSON.parse(e.data);

            if(message.resp)
            {
                if(message.resp == "FAIL")
                {
                    if (confirm('Overwrite?'))
                    {
                        message.force=true;
                        console.log("Sending: " + JSON.stringify(message));
                        ws_storage.send(JSON.stringify(message));
                    }
                    else
                    {
                        //do nothing
                    }
                }

                return;
            }

            //the message is an array of profiles
            //FIXME: this should be better, maybe a {"profiles": ...} container?
            profiles = message;
            //delete old options in select
            $('#e2').find('option').remove().end();
            // check if current selected value is a valid profile name
            // if not, update with first available profile name
            var valid_profile_names = profiles.map(function(a) {return a.name;});
            if (
              valid_profile_names.length > 0 &&
              $.inArray(selected_profile_name, valid_profile_names) === -1
            ) {
              selected_profile = 0;
              selected_profile_name = valid_profile_names[0];
            }

            // fill select with new options from websocket
            for (var i=0; i<profiles.length; i++)
            {
                var profile = profiles[i];
                //console.log(profile.name);
                $('#e2').append('<option value="'+i+'">'+profile.name+'</option>');

                if (profile.name == selected_profile_name)
                {
                    selected_profile = i;
                    $('#e2').select2('val', i);
                    updateProfile(i);
                }
            }
        };


        $("#e2").select2(
        {
            placeholder: "Select Profile",
            allowClear: true,
            minimumResultsForSearch: -1
        });


        $("#e2").on("change", function(e)
        {
            updateProfile(e.val);
        });

    }
});

// ============================================================
// SCHEDULE BUILDER
// ============================================================

var sbData = { name: '', startTemp: 65, rows: [] };

function openScheduleBuilder() {
    var curTemp = parseInt($('#act_temp').text());
    if (isNaN(curTemp)) curTemp = (temp_scale === 'f') ? 65 : 18;
    sbData = {
        name: '',
        startTemp: curTemp,
        rows: [
            { to: (temp_scale === 'f') ? 200 : 93, rate: (temp_scale === 'f') ? 300 : 167, hold: 0, mode: 'rate', desc: '' }
        ]
    };
    $('#sb-profile-name').val('');
    $('#sb-start-temp').val(sbData.startTemp);
    $('#sb-start-unit').text('°' + temp_scale_display);
    $('#sb-start-temp').off('change').on('change', function() { sbRenderRows(); });
    sbRenderRows();
    $('#scheduleBuilderModal').modal('show');
}

function openScheduleBuilderForEdit() {
    var profile = profiles[selected_profile];
    if (!profile || !profile.data || profile.data.length < 2) {
        openScheduleBuilder();
        return;
    }

    var labels = profile.segment_labels || [];

    // convert stored °C waypoints to display units
    function toDisplay(c) {
        return (temp_scale === 'f') ? Math.round(c * 9/5 + 32) : Math.round(c);
    }

    var startTemp = toDisplay(profile.data[0][1]);
    var rows = [];

    for (var i = 1; i < profile.data.length; i++) {
        var prevSecs = profile.data[i-1][0];
        var currSecs = profile.data[i][0];
        var prevTemp = toDisplay(profile.data[i-1][1]);
        var currTemp = toDisplay(profile.data[i][1]);
        var dt = currSecs - prevSecs;

        if (currTemp === prevTemp) {
            // hold — add to the last row's hold
            if (rows.length > 0) {
                rows[rows.length - 1].hold = Math.round(dt / 60);
            }
        } else {
            var rate = Math.round(Math.abs(currTemp - prevTemp) / dt * 3600);
            rows.push({ to: currTemp, rate: rate, hold: 0, mode: 'rate', desc: labels[rows.length] || '' });
        }
    }

    sbData = { name: profile.name, startTemp: startTemp, rows: rows };
    $('#sb-profile-name').val(profile.name);
    $('#sb-start-temp').val(startTemp);
    $('#sb-start-unit').text('°' + temp_scale_display);
    $('#sb-start-temp').off('change').on('change', function() { sbRenderRows(); });
    sbRenderRows();
    $('#scheduleBuilderModal').modal('show');
}

function sbFromTemp(idx) {
    if (idx === 0) return parseFloat($('#sb-start-temp').val()) || sbData.startTemp;
    return sbData.rows[idx - 1].to;
}

function sbCalcDuration(from, to, rate) {
    if (!rate || rate <= 0) return 0;
    return Math.abs(to - from) / rate * 60;
}

function sbCalcRate(from, to, durationMin) {
    if (!durationMin || durationMin <= 0) return 0;
    return Math.abs(to - from) / (durationMin / 60);
}

function sbFormatDuration(minutes) {
    var h = Math.floor(minutes / 60);
    var m = Math.round(minutes % 60);
    return h + 'h ' + String(m).padStart(2, '0') + 'm';
}

function sbRenderRows() {
    var html = '';
    var totalMin = 0;

    sbData.rows.forEach(function(row, i) {
        var from = sbFromTemp(i);
        var durationMin = (row.mode === 'rate') ? sbCalcDuration(from, row.to, row.rate) : (row.duration || 0);
        var rate = (row.mode === 'rate') ? row.rate : sbCalcRate(from, row.to, row.duration);
        totalMin += durationMin + (row.hold || 0);

        var rateCell = (row.mode === 'rate')
            ? '<input class="form-control sb-input sb-rate-input" type="number" min="1" value="' + Math.round(rate) + '" data-row="' + i + '">'
            : '<span class="sb-calc" data-row="' + i + '" title="Click to enter rate">' + Math.round(rate) + '</span>';

        var durCell = (row.mode === 'rate')
            ? '<span class="sb-calc" data-row="' + i + '" title="Click to enter duration">' + sbFormatDuration(durationMin) + '</span>'
            : '<input class="form-control sb-input sb-dur-input" type="number" min="1" value="' + Math.round(durationMin) + '" data-row="' + i + '" placeholder="min">';

        html += '<tr data-row="' + i + '">' +
            '<td class="sb-from-cell"><input class="form-control sb-input" type="text" readonly value="' + Math.round(from) + '" style="background:transparent;cursor:default;"></td>' +
            '<td><input class="form-control sb-input sb-to-input" type="number" value="' + row.to + '" data-row="' + i + '"></td>' +
            '<td>' + rateCell + '</td>' +
            '<td>' + durCell + '</td>' +
            '<td><input class="form-control sb-input sb-hold-input" type="number" min="0" value="' + (row.hold || 0) + '" data-row="' + i + '"></td>' +
            '<td><input class="form-control sb-input sb-desc-input" type="text" value="' + (row.desc || '') + '" data-row="' + i + '" placeholder="e.g. candling"></td>' +
            '<td><button class="btn btn-xs sb-delete-btn" data-row="' + i + '">×</button></td>' +
        '</tr>';
    });

    $('#sb-tbody').html(html);

    var totalH = Math.floor(totalMin / 60);
    var totalM = Math.round(totalMin % 60);
    $('#sb-total').text('Total: ' + totalH + 'h ' + String(totalM).padStart(2, '0') + 'm');

    $('#sb-tbody .sb-to-input').on('change', function() {
        var i = parseInt($(this).data('row'));
        sbData.rows[i].to = parseFloat($(this).val()) || 0;
        sbRenderRows();
    });
    $('#sb-tbody .sb-rate-input').on('change', function() {
        var i = parseInt($(this).data('row'));
        sbData.rows[i].rate = parseFloat($(this).val()) || 0;
        sbRenderRows();
    });
    $('#sb-tbody .sb-dur-input').on('change', function() {
        var i = parseInt($(this).data('row'));
        sbData.rows[i].duration = parseFloat($(this).val()) || 0;
        sbRenderRows();
    });
    $('#sb-tbody .sb-hold-input').on('change', function() {
        var i = parseInt($(this).data('row'));
        sbData.rows[i].hold = parseFloat($(this).val()) || 0;
        sbRenderRows();
    });
    $('#sb-tbody .sb-calc').on('click', function() {
        var i = parseInt($(this).data('row'));
        var from = sbFromTemp(i);
        var row = sbData.rows[i];
        if (row.mode === 'rate') {
            row.duration = Math.round(sbCalcDuration(from, row.to, row.rate));
            row.mode = 'duration';
        } else {
            row.rate = Math.round(sbCalcRate(from, row.to, row.duration));
            row.mode = 'rate';
        }
        sbRenderRows();
    });
    $('#sb-tbody .sb-desc-input').on('change', function() {
        var i = parseInt($(this).data('row'));
        sbData.rows[i].desc = $(this).val();
    });
    $('#sb-tbody .sb-delete-btn').on('click', function() {
        var i = parseInt($(this).data('row'));
        sbData.rows.splice(i, 1);
        sbRenderRows();
    });
}

function sbAddRow() {
    var last = sbData.rows[sbData.rows.length - 1];
    var prevTo = last ? last.to : (parseFloat($('#sb-start-temp').val()) || sbData.startTemp);
    sbData.rows.push({ to: prevTo, rate: (temp_scale === 'f') ? 100 : 55, hold: 0, mode: 'rate', desc: '' });
    sbRenderRows();
}

function sbSave() {
    var name = $('#sb-profile-name').val().trim();
    if (!name) { alert('Please enter a schedule name.'); return; }
    if (sbData.rows.length === 0) { alert('Please add at least one segment.'); return; }

    var startTemp = parseFloat($('#sb-start-temp').val()) || sbData.startTemp;
    var startTempC = (temp_scale === 'f') ? (startTemp - 32) * 5 / 9 : startTemp;
    var waypoints = [[0, startTempC]];
    var elapsed = 0;

    sbData.rows.forEach(function(row, i) {
        var from = sbFromTemp(i);
        var durationMin = (row.mode === 'rate') ? sbCalcDuration(from, row.to, row.rate) : (row.duration || 0);
        elapsed += durationMin * 60;
        var toTempC = (temp_scale === 'f') ? (row.to - 32) * 5 / 9 : row.to;
        waypoints.push([Math.round(elapsed), toTempC]);
        if (row.hold && row.hold > 0) {
            elapsed += row.hold * 60;
            waypoints.push([Math.round(elapsed), toTempC]);
        }
    });

    var labels = sbData.rows.map(function(r) { return r.desc || ''; });
    var profile = { "type": "profile", "data": waypoints, "name": name, "temp_units": "c", "segment_labels": labels };
    ws_storage.send(JSON.stringify({ "cmd": "PUT", "profile": profile }));
    $('#scheduleBuilderModal').modal('hide');
    ws_storage.send('GET');
}
