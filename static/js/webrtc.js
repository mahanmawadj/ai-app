/**
 * WebRTC client implementation for TensorRT inference
 */

// Configure RTCPeerConnection with STUN servers
const rtcConfig = {
    iceServers: [
        { urls: 'stun:stun.l.google.com:19302' },
        { urls: 'stun:stun1.l.google.com:19302' }
    ]
};

/**
 * Start WebRTC connection and stream
 * @param {HTMLVideoElement} videoElement - Video element to display the stream
 * @returns {RTCPeerConnection} - WebRTC peer connection
 */
async function startWebRTC(videoElement) {
    // Create peer connection
    const pc = new RTCPeerConnection(rtcConfig);
    
    // Set up event handlers
    setupPeerConnectionEvents(pc, videoElement);
    
    // Create data channel (optional, for future use)
    const dataChannel = pc.createDataChannel('control');
    setupDataChannel(dataChannel);
    
    // Add local video track (optional, if sending local camera)
    // await addLocalStream(pc);
    
    // Create and send offer
    const offer = await pc.createOffer({
        offerToReceiveAudio: false,
        offerToReceiveVideo: true
    });
    
    await pc.setLocalDescription(offer);
    
    // Send offer to server and get answer
    const answer = await sendOfferToServer(pc.localDescription);
    
    // Set remote description
    await pc.setRemoteDescription(new RTCSessionDescription(answer));
    
    return pc;
}

/**
 * Set up event handlers for peer connection
 * @param {RTCPeerConnection} pc - WebRTC peer connection
 * @param {HTMLVideoElement} videoElement - Video element to display the stream
 */
function setupPeerConnectionEvents(pc, videoElement) {
    // ICE connection state change
    pc.oniceconnectionstatechange = () => {
        console.log(`ICE connection state: ${pc.iceConnectionState}`);
    };
    
    // Connection state change
    pc.onconnectionstatechange = () => {
        console.log(`Connection state: ${pc.connectionState}`);
        
        if (pc.connectionState === 'disconnected' || 
            pc.connectionState === 'failed' || 
            pc.connectionState === 'closed') {
            
            console.log('WebRTC connection closed or failed');
            
            // Clear video element
            if (videoElement.srcObject) {
                const tracks = videoElement.srcObject.getTracks();
                tracks.forEach(track => track.stop());
                videoElement.srcObject = null;
            }
        }
    };
    
    // Track event (receiving remote track)
    pc.ontrack = (event) => {
        console.log('Received remote track');
        
        if (event.streams && event.streams[0]) {
            videoElement.srcObject = event.streams[0];
        } else {
            // Create a new MediaStream if no stream is provided
            let newStream = new MediaStream();
            newStream.addTrack(event.track);
            videoElement.srcObject = newStream;
        }
    };
    
    // ICE candidate event
    pc.onicecandidate = (event) => {
        if (event.candidate) {
            console.log('New ICE candidate:', event.candidate.candidate);
        } else {
            console.log('ICE gathering complete');
        }
    };
}

/**
 * Set up event handlers for data channel
 * @param {RTCDataChannel} dataChannel - WebRTC data channel
 */
function setupDataChannel(dataChannel) {
    // Data channel open
    dataChannel.onopen = () => {
        console.log('Data channel open');
    };
    
    // Data channel close
    dataChannel.onclose = () => {
        console.log('Data channel closed');
    };
    
    // Data channel message
    dataChannel.onmessage = (event) => {
        console.log('Received message:', event.data);
        
        try {
            const message = JSON.parse(event.data);
            handleDataChannelMessage(message);
        } catch (error) {
            console.error('Error parsing message:', error);
        }
    };
}

/**
 * Handle data channel message
 * @param {Object} message - Message object
 */
function handleDataChannelMessage(message) {
    // Handle different message types
    if (message.type === 'stats') {
        console.log('Received stats:', message.data);
    } else if (message.type === 'event') {
        console.log('Received event:', message.data);
    }
}

/**
 * Add local camera stream to peer connection
 * @param {RTCPeerConnection} pc - WebRTC peer connection
 */
async function addLocalStream(pc) {
    try {
        // Get user media (camera)
        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 1280 },
                height: { ideal: 720 }
            },
            audio: false
        });
        
        // Add tracks to peer connection
        stream.getTracks().forEach(track => {
            pc.addTrack(track, stream);
        });
        
        console.log('Local stream added to peer connection');
        
        return stream;
    } catch (error) {
        console.error('Error getting user media:', error);
        return null;
    }
}

/**
 * Send offer to server and get answer
 * @param {RTCSessionDescription} offer - WebRTC offer
 * @returns {Object} - WebRTC answer
 */
async function sendOfferToServer(offer) {
    try {
        const response = await fetch('/api/webrtc_offer', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                sdp: offer.sdp,
                type: offer.type
            })
        });
        
        if (!response.ok) {
            throw new Error(`Server responded with status ${response.status}`);
        }
        
        return await response.json();
    } catch (error) {
        console.error('Error sending offer to server:', error);
        throw error;
    }
}

/**
 * Stop WebRTC connection
 * @param {RTCPeerConnection} pc - WebRTC peer connection
 */
async function stopWebRTC(pc) {
    if (!pc) return;
    
    // Close data channels
    pc.getDataChannels().forEach(channel => {
        channel.close();
    });
    
    // Close peer connection
    pc.close();
    
    console.log('WebRTC connection stopped');
}