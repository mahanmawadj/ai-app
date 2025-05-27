// Initialize states and globals
let streamActive = false;
var pc = null;
var dc = null, dcInterval = null;

// get DOM elements
var dataChannelLog = document.getElementById('data-channel'),
    iceConnectionLog = document.getElementById('ice-connection-state'),
    iceGatheringLog = document.getElementById('ice-gathering-state'),
    signalingLog = document.getElementById('signaling-state');

// UI Controls
const startButton = document.getElementById('startButton');
const stopButton = document.getElementById('stopButton');
const optionsToggle = document.getElementById('optionsToggle');
const optionsPanel = document.getElementById('optionsPanel');
const sdpToggle = document.getElementById('sdpToggle');
const sdpContent = document.getElementById('sdpContent');

// Model toggles
const detectionToggle = document.getElementById('detectionEnabled');
const classificationToggle = document.getElementById('classificationEnabled');
const poseToggle = document.getElementById('poseEnabled');
const actionToggle = document.getElementById('actionEnabled');
const segmentationToggle = document.getElementById('segmentationEnabled');

// Model settings
const detectionThreshold = document.getElementById('detectionThreshold');
const classificationThreshold = document.getElementById('classificationThreshold');
const poseThreshold = document.getElementById('poseThreshold');
const actionThreshold = document.getElementById('actionThreshold');
const segmentationAlpha = document.getElementById('segmentationAlpha');

// Value displays
const detectionThresholdValue = document.getElementById('detectionThresholdValue');
const classificationThresholdValue = document.getElementById('classificationThresholdValue');
const poseThresholdValue = document.getElementById('poseThresholdValue');
const actionThresholdValue = document.getElementById('actionThresholdValue');
const segmentationAlphaValue = document.getElementById('segmentationAlphaValue');

// Result sections
const resultSections = document.querySelectorAll('.result-section');

// Model interactions
const changeModelBtn = document.getElementById('changeModelBtn');
const modelSelector = document.getElementById('modelSelector');
const currentModelEl = document.getElementById('currentModel');

// Model selectors for other model types
const modelSelectors = document.querySelectorAll('.model-section select');
const changeModelBtns = document.querySelectorAll('.model-section button');
const modelStatusEls = document.querySelectorAll('.model-section .model-status');

// UI Toggles
optionsToggle.addEventListener('click', () => {
    optionsPanel.style.display = optionsPanel.style.display === 'none' ? 'block' : 'none';
});

sdpToggle.addEventListener('click', () => {
    if (sdpContent.classList.contains('show')) {
        sdpContent.classList.remove('show');
        sdpToggle.innerHTML = '<i class="fas fa-chevron-down"></i>';
    } else {
        sdpContent.classList.add('show');
        sdpToggle.innerHTML = '<i class="fas fa-chevron-up"></i>';
    }
});

// Update value displays
detectionThreshold.addEventListener('input', () => {
    detectionThresholdValue.textContent = detectionThreshold.value;
});

classificationThreshold.addEventListener('input', () => {
    classificationThresholdValue.textContent = classificationThreshold.value;
});

poseThreshold.addEventListener('input', () => {
    poseThresholdValue.textContent = poseThreshold.value;
});

actionThreshold.addEventListener('input', () => {
    actionThresholdValue.textContent = actionThreshold.value;
});

segmentationAlpha.addEventListener('input', () => {
    segmentationAlphaValue.textContent = segmentationAlpha.value;
});

// Toggle model states and update results visibility
detectionToggle.addEventListener('change', async () => {
    await toggleModel('detection_enabled', detectionToggle.checked);
    toggleModelSection(detectionToggle, 'detectionBody');
    toggleResultSection('detectionResults', detectionToggle.checked);
});

classificationToggle.addEventListener('change', async () => {
    await toggleModel('classification_enabled', classificationToggle.checked);
    toggleModelSection(classificationToggle, 'classificationBody');
    toggleResultSection('classificationResults', classificationToggle.checked);
});

poseToggle.addEventListener('change', async () => {
    await toggleModel('pose_enabled', poseToggle.checked);
    toggleModelSection(poseToggle, 'poseBody');
    toggleResultSection('poseResults', poseToggle.checked);
});

actionToggle.addEventListener('change', async () => {
    await toggleModel('action_enabled', actionToggle.checked);
    toggleModelSection(actionToggle, 'actionBody');
    toggleResultSection('actionResults', actionToggle.checked);
});

segmentationToggle.addEventListener('change', async () => {
    await toggleModel('segmentation_enabled', segmentationToggle.checked);
    toggleModelSection(segmentationToggle, 'segmentationBody');
    toggleResultSection('segmentationResults', segmentationToggle.checked);
});

// Function to toggle model section visibility
function toggleModelSection(toggle, bodyId) {
    const body = document.getElementById(bodyId);
    if (toggle.checked) {
        // If enabled, show the body (add show class if not already there)
        if (!body.classList.contains('show')) {
            body.classList.add('show');
        }
    } else {
        // If disabled, hide the body (remove show class)
        body.classList.remove('show');
    }
}

// Function to toggle result section visibility
function toggleResultSection(resultId, isVisible) {
    const resultSection = document.getElementById(resultId);
    if (resultSection) {
        if (isVisible) {
            resultSection.classList.add('active');
        } else {
            resultSection.classList.remove('active');
        }
    }
}

// Bind change model events for all model types
changeModelBtns.forEach((btn, index) => {
    btn.addEventListener('click', async () => {
        const section = btn.closest('.model-section');
        const selector = section.querySelector('select');
        const statusEl = section.querySelector('.model-status');

        if (!selector || !statusEl) return;

        const selectedModel = selector.value;
        statusEl.textContent = "Changing...";

        try {
            // For now, we only have the classification model change API implemented
            if (index === 0) {  // Classification model
                const response = await fetch('/api/model/change', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        model_name: selectedModel
                    })
                });

                const data = await response.json();

                if (data.success) {
                    statusEl.textContent = selectedModel;
                } else {
                    statusEl.textContent = "Error: " + data.error;
                }
            } else {
                // Placeholder for other model types (will be implemented later)
                // For now, just update the UI to show it's changed
                setTimeout(() => {
                    statusEl.textContent = selectedModel;
                }, 500);
            }
        } catch (error) {
            console.error('Error changing model:', error);
            statusEl.textContent = "Error changing model";
        }
    });
});

// Start stream button functionality
startButton.addEventListener('click', async () => {
    if (streamActive) return;

    try {
        start();
        startButton.disabled = true;
        stopButton.disabled = false;
        streamActive = true;

        //await initModelStates();
    } catch (error) {
        console.error('Error starting stream:', error);
    }
});

// Stop stream
stopButton.addEventListener('click', async () => {
    if (!streamActive) return;

    try {
        stop();
        startButton.disabled = false;
        stopButton.disabled = true;
        streamActive = false;
    } catch (error) {
        console.error('Error stopping stream:', error);
    }
});

// Toggle model API
async function toggleModel(endpoint, enabled) {
    try {
        const response = await fetch(`/api/${endpoint}`, {
            method: 'PUT',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ [endpoint]: enabled })
        });

        if (!response.ok) {
            console.error(`Failed to set ${endpoint}: ${response.statusText}`);
        }
    } catch (error) {
        console.error(`Error setting ${endpoint}:`, error);
    }
}

async function initModelStates() {
    try {
        // Fetch initial model states
        const endpoints = [
            'detection_enabled',
            'classification_enabled',
            'pose_enabled',
            'action_enabled',
            'segmentation_enabled'
        ];

        for (const endpoint of endpoints) {
            const response = await fetch(`/api/${endpoint}`);
            if (response.ok) {
                const data = await response.json();
                const toggle = document.getElementById(endpoint.replace('_enabled', 'Enabled'));
                if (toggle) {
                    toggle.checked = data[endpoint];

                    // Set the initial visibility of the model section based on the toggle state
                    const bodyId = endpoint.replace('_enabled', 'Body');
                    toggleModelSection(toggle, bodyId);

                    // Set the initial visibility of the result section based on the toggle state
                    const resultId = endpoint.replace('_enabled', 'Results');
                    toggleResultSection(resultId, toggle.checked);
                }
            }
        }
    } catch (error) {
        console.error('Error initializing model states:', error);
    }
}

function createPeerConnection() {
    var config = {
        sdpSemantics: 'unified-plan'
    };

    if (document.getElementById('use-stun').checked) {
        config.iceServers = [{ urls: ['stun:stun.l.google.com:19302'] }];
    }

    pc = new RTCPeerConnection(config);

    // register some listeners to help debugging
    pc.addEventListener('icegatheringstatechange', () => {
        if (iceGatheringLog) {
            iceGatheringLog.textContent += ' -> ' + pc.iceGatheringState;
        }
    }, false);
    if (iceGatheringLog) {
        iceGatheringLog.textContent = pc.iceGatheringState;
    }

    pc.addEventListener('iceconnectionstatechange', () => {
        if (iceConnectionLog) {
            iceConnectionLog.textContent += ' -> ' + pc.iceConnectionState;
        }
    }, false);
    if (iceConnectionLog) {
        iceConnectionLog.textContent = pc.iceConnectionState;
    }

    pc.addEventListener('signalingstatechange', () => {
        if (signalingLog) {
            signalingLog.textContent += ' -> ' + pc.signalingState;
        }
    }, false);
    if (signalingLog) {
        signalingLog.textContent = pc.signalingState;
    }

    // connect audio / video
    pc.addEventListener('track', (evt) => {
        if (evt.track.kind == 'video')
            document.getElementById('video').srcObject = evt.streams[0];
        else
            document.getElementById('audio').srcObject = evt.streams[0];
    });

    return pc;
}

function enumerateInputDevices() {
    const populateSelect = (select, devices) => {
        let counter = 1;
        devices.forEach((device) => {
            const option = document.createElement('option');
            option.value = device.deviceId;
            option.text = device.label || ('Device #' + counter);
            select.appendChild(option);
            counter += 1;
        });
    };

    navigator.mediaDevices.enumerateDevices().then((devices) => {
        populateSelect(
            document.getElementById('audio-input'),
            devices.filter((device) => device.kind == 'audioinput')
        );
        populateSelect(
            document.getElementById('video-input'),
            devices.filter((device) => device.kind == 'videoinput')
        );
    }).catch((e) => {
        console.error('Error enumerating devices:', e);
    });
}

function negotiate() {
    return pc.createOffer().then((offer) => {
        return pc.setLocalDescription(offer);
    }).then(() => {
        // wait for ICE gathering to complete
        return new Promise((resolve) => {
            if (pc.iceGatheringState === 'complete') {
                resolve();
            } else {
                function checkState() {
                    if (pc.iceGatheringState === 'complete') {
                        pc.removeEventListener('icegatheringstatechange', checkState);
                        resolve();
                    }
                }
                pc.addEventListener('icegatheringstatechange', checkState);
            }
        });
    }).then(() => {
        var offer = pc.localDescription;
        var codec;

        codec = document.getElementById('audio-codec').value;
        if (codec !== 'default') {
            offer.sdp = sdpFilterCodec('audio', codec, offer.sdp);
        }

        codec = document.getElementById('video-codec').value;
        if (codec !== 'default') {
            offer.sdp = sdpFilterCodec('video', codec, offer.sdp);
        }

        document.getElementById('offer-sdp').textContent = offer.sdp;
        return fetch('/offer', {
            body: JSON.stringify({
                sdp: offer.sdp,
                type: offer.type,
                video_transform: document.getElementById('video-transform').value
            }),
            headers: {
                'Content-Type': 'application/json'
            },
            method: 'POST'
        });
    }).then((response) => {
        return response.json();
    }).then((answer) => {
        document.getElementById('answer-sdp').textContent = answer.sdp;
        return pc.setRemoteDescription(answer);
    }).catch((e) => {
        console.error('Negotiation error:', e);
    });
}

function start() {
    pc = createPeerConnection();
    var time_start = null;
    const current_stamp = () => {
        if (time_start === null) {
            time_start = new Date().getTime();
            return 0;
        } else {
            return new Date().getTime() - time_start;
        }
    };

    if (document.getElementById('use-datachannel').checked) {
        var parameters = JSON.parse(document.getElementById('datachannel-parameters').value);

        dc = pc.createDataChannel('chat', parameters);
        dc.addEventListener('close', () => {
            clearInterval(dcInterval);
            dataChannelLog.textContent += '- close\n';
        });
        dc.addEventListener('open', () => {
            dataChannelLog.textContent += '- open\n';
            dcInterval = setInterval(() => {
                var message = 'ping ' + current_stamp();
                dataChannelLog.textContent += '> ' + message + '\n';
                dc.send(message);
            }, 1000);
        });
        dc.addEventListener('message', (evt) => {
            dataChannelLog.textContent += '< ' + evt.data + '\n';

            if (evt.data.substring(0, 4) === 'pong') {
                var elapsed_ms = current_stamp() - parseInt(evt.data.substring(5), 10);
                dataChannelLog.textContent += ' RTT ' + elapsed_ms + ' ms\n';
            }
        });
    }

    // Build media constraints
    const constraints = {
        audio: false,
        video: false
    };

    if (document.getElementById('use-audio').checked) {
        const audioConstraints = {};

        const device = document.getElementById('audio-input').value;
        if (device) {
            audioConstraints.deviceId = { exact: device };
        }

        constraints.audio = Object.keys(audioConstraints).length ? audioConstraints : true;
    }

    if (document.getElementById('use-video').checked) {
        const videoConstraints = {};

        const device = document.getElementById('video-input').value;
        if (device) {
            videoConstraints.deviceId = { exact: device };
        }

        const resolution = document.getElementById('video-resolution').value;
        if (resolution) {
            const dimensions = resolution.split('x');
            videoConstraints.width = parseInt(dimensions[0], 0);
            videoConstraints.height = parseInt(dimensions[1], 0);
        }

        constraints.video = Object.keys(videoConstraints).length ? videoConstraints : true;
    }

    // Acquire media and start negotiation
    if (constraints.audio || constraints.video) {
        navigator.mediaDevices.getUserMedia(constraints).then((stream) => {
            stream.getTracks().forEach((track) => {
                pc.addTrack(track, stream);
            });
            return negotiate();
        }, (err) => {
            console.error('Could not acquire media:', err);
        });
    } else {
        negotiate();
    }
}

function stop() {
    // close data channel
    if (dc) {
        dc.close();
    }

    // close transceivers
    if (pc.getTransceivers) {
        pc.getTransceivers().forEach((transceiver) => {
            if (transceiver.stop) {
                transceiver.stop();
            }
        });
    }

    // close local audio / video
    pc.getSenders().forEach((sender) => {
        sender.track.stop();
    });

    // close peer connection
    setTimeout(() => {
        pc.close();
    }, 500);
}

function sdpFilterCodec(kind, codec, realSdp) {
    var allowed = [];
    var rtxRegex = new RegExp('a=fmtp:(\\d+) apt=(\\d+)\r$');
    var codecRegex = new RegExp('a=rtpmap:([0-9]+) ' + escapeRegExp(codec));
    var videoRegex = new RegExp('(m=' + kind + ' .*?)( ([0-9]+))*\\s*$');

    var lines = realSdp.split('\n');

    var isKind = false;
    for (var i = 0; i < lines.length; i++) {
        if (lines[i].startsWith('m=' + kind + ' ')) {
            isKind = true;
        } else if (lines[i].startsWith('m=')) {
            isKind = false;
        }

        if (isKind) {
            var match = lines[i].match(codecRegex);
            if (match) {
                allowed.push(parseInt(match[1]));
            }

            match = lines[i].match(rtxRegex);
            if (match && allowed.includes(parseInt(match[2]))) {
                allowed.push(parseInt(match[1]));
            }
        }
    }

    var skipRegex = 'a=(fmtp|rtcp-fb|rtpmap):([0-9]+)';
    var sdp = '';

    isKind = false;
    for (var i = 0; i < lines.length; i++) {
        if (lines[i].startsWith('m=' + kind + ' ')) {
            isKind = true;
        } else if (lines[i].startsWith('m=')) {
            isKind = false;
        }

        if (isKind) {
            var skipMatch = lines[i].match(skipRegex);
            if (skipMatch && !allowed.includes(parseInt(skipMatch[2]))) {
                continue;
            } else if (lines[i].match(videoRegex)) {
                sdp += lines[i].replace(videoRegex, '$1 ' + allowed.join(' ')) + '\n';
            } else {
                sdp += lines[i] + '\n';
            }
        } else {
            sdp += lines[i] + '\n';
        }
    }

    return sdp;
}

function escapeRegExp(string) {
    return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'); // $& means the whole matched string
}

// Fetch current model on page load
async function fetchCurrentModel() {
    try {
        const response = await fetch('/api/model');
        const data = await response.json();
        currentModelEl.textContent = data.model_name;
        // Set the select value
        if (modelSelector.querySelector(`option[value="${data.model_name}"]`)) {
            modelSelector.value = data.model_name;
        }
    } catch (error) {
        console.error('Error fetching current model:', error);
        currentModelEl.textContent = "Unknown";
    }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    enumerateInputDevices();
    fetchCurrentModel();
    initModelStates();

    // Check 'Use Video' by default
    document.getElementById('use-video').checked = true;

    // Initialize SDP panel display
    if (sdpContent) {
        sdpContent.classList.add('show');
    }
});