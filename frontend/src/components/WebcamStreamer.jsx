import React, { useRef, useEffect, useState } from 'react';

const BACKEND_URL = import.meta.env.VITE_API_URL ? import.meta.env.VITE_API_URL.replace(/\/$/, '') : 'http://localhost:8000';
const API_PREDICT_URL = `${BACKEND_URL}/api/v1/mask-detection/predict`;

export default function WebcamStreamer({ isStreaming, onPredictionUpdate, isMirrored = false }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const offscreenCanvasRef = useRef(null);
  
  const isStreamingRef = useRef(isStreaming);
  const isProcessingRef = useRef(false);
  const facesRef = useRef([]);
  const sourceDimRef = useRef({ width: 640, height: 360 });
  const [errorMsg, setErrorMsg] = useState(null);

  useEffect(() => {
    isStreamingRef.current = isStreaming;
  }, [isStreaming]);

  useEffect(() => {
    offscreenCanvasRef.current = document.createElement('canvas');
  }, []);

  // Camera initialization
  useEffect(() => {
    let stream = null;

    if (isStreaming) {
      setErrorMsg(null);
      navigator.mediaDevices.getUserMedia({ 
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' } 
      })
      .then((userStream) => {
        stream = userStream;
        if (videoRef.current) {
          videoRef.current.srcObject = userStream;
          videoRef.current.play().catch(e => console.warn("Autoplay handler:", e));
        }
      })
      .catch((err) => {
        console.error("Camera access error:", err);
        setErrorMsg("Unable to access video camera. Please verify camera permissions in your browser.");
      });
    } else {
      facesRef.current = [];
      if (videoRef.current && videoRef.current.srcObject) {
        videoRef.current.srcObject.getTracks().forEach(track => track.stop());
        videoRef.current.srcObject = null;
      }
    }

    return () => {
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }
    };
  }, [isStreaming]);

  // 60 FPS Smooth Canvas Render Loop
  useEffect(() => {
    let animId = null;

    const drawLoop = () => {
      if (isStreamingRef.current && videoRef.current && canvasRef.current) {
        const video = videoRef.current;
        const canvas = canvasRef.current;

        if (video.videoWidth && video.videoHeight) {
          if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
          }

          const ctx = canvas.getContext('2d');
          ctx.clearRect(0, 0, canvas.width, canvas.height);
          
          drawBoundingBoxes(ctx, facesRef.current, video.videoWidth, video.videoHeight, isMirrored);
        }
      }

      if (isStreamingRef.current) {
        animId = requestAnimationFrame(drawLoop);
      }
    };

    if (isStreaming) {
      animId = requestAnimationFrame(drawLoop);
    }

    return () => {
      if (animId) cancelAnimationFrame(animId);
    };
  }, [isStreaming, isMirrored]);

  // Async Non-blocking API Prediction Loop
  useEffect(() => {
    let timerId = null;

    const runAsyncPredict = async () => {
      if (!isStreamingRef.current) return;

      if (videoRef.current && videoRef.current.readyState >= 2 && !isProcessingRef.current) {
        isProcessingRef.current = true;

        try {
          const video = videoRef.current;
          const offCanvas = offscreenCanvasRef.current;
          
          const vWidth = video.videoWidth || 640;
          const vHeight = video.videoHeight || 360;
          const targetW = 640;
          const targetH = Math.round(targetW * (vHeight / vWidth));

          offCanvas.width = targetW;
          offCanvas.height = targetH;
          const offCtx = offCanvas.getContext('2d');
          
          offCtx.drawImage(video, 0, 0, targetW, targetH);
          const base64Img = offCanvas.toDataURL('image/jpeg', 0.55);

          const response = await fetch(API_PREDICT_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: base64Img })
          });

          if (response.ok) {
            const data = await response.json();
            sourceDimRef.current = {
              width: data.source_width || targetW,
              height: data.source_height || targetH
            };

            facesRef.current = data.faces || [];
            if (onPredictionUpdate) {
              onPredictionUpdate(data);
            }
          }
        } catch (err) {
          // Backend loading
        } finally {
          isProcessingRef.current = false;
        }
      }

      if (isStreamingRef.current) {
        timerId = setTimeout(runAsyncPredict, 110);
      }
    };

    if (isStreaming) {
      runAsyncPredict();
    }

    return () => {
      if (timerId) clearTimeout(timerId);
    };
  }, [isStreaming]);

  // Render bounding boxes & 3-state label pills (SAFE / UNSAFE / CHECKING)
  const drawBoundingBoxes = (ctx, faces, videoW, videoH, mirrored) => {
    if (!faces || faces.length === 0) return;

    const scaleX = videoW / (sourceDimRef.current.width || 640);
    const scaleY = videoH / (sourceDimRef.current.height || 360);

    faces.forEach((face) => {
      let bx = face.x * scaleX;
      let by = face.y * scaleY;
      let bw = face.width * scaleX;
      let bh = face.height * scaleY;

      if (mirrored) {
        bx = videoW - (bx + bw);
      }

      let color = '#FF2E54';
      let statusIcon = '🔴';
      let statusText = 'UNSAFE';

      if (face.label === 'SAFE') {
        color = '#00FF87';
        statusIcon = '🟢';
        statusText = 'SAFE';
      } else if (face.label === 'CHECKING') {
        color = '#FACC15';
        statusIcon = '🟡';
        statusText = 'CHECKING';
      }

      const confPercent = (face.confidence * 100).toFixed(1);

      // 1. Draw Bounding Box Rectangle
      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.strokeRect(bx, by, bw, bh);

      // 2. Corner Accents
      const cornerLen = Math.min(22, bw / 4);
      ctx.lineWidth = 5;
      ctx.beginPath();
      ctx.moveTo(bx, by + cornerLen); ctx.lineTo(bx, by); ctx.lineTo(bx + cornerLen, by);
      ctx.moveTo(bx + bw - cornerLen, by); ctx.lineTo(bx + bw, by); ctx.lineTo(bx + bw, by + cornerLen);
      ctx.moveTo(bx, by + bh - cornerLen); ctx.lineTo(bx, by + bh); ctx.lineTo(bx + cornerLen, by + bh);
      ctx.moveTo(bx + bw - cornerLen, by + bh); ctx.lineTo(bx + bw, by + bh); ctx.lineTo(bx + bw, by + bh - cornerLen);
      ctx.stroke();

      // 3. Draw Individual Result Label Pill (🟢 SAFE — 97.4% / 🔴 UNSAFE — 94.2% / 🟡 CHECKING — 65.0%)
      const labelStr = `${statusIcon} ${statusText} — ${confPercent}%`;
      ctx.font = 'bold 15px "Outfit", sans-serif';
      const textMetrics = ctx.measureText(labelStr);
      const pillWidth = textMetrics.width + 20;
      const pillHeight = 28;

      let pillX = bx;
      let pillY = by - pillHeight - 6;

      if (pillY < 5) {
        pillY = by + 6; // Inside top of box if near camera top edge
      }

      // Pill Background Box
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.roundRect(pillX, pillY, pillWidth, pillHeight, 6);
      ctx.fill();

      // Pill Text
      ctx.fillStyle = '#0B0E14';
      ctx.fillText(labelStr, pillX + 10, pillY + 19);
    });
  };

  return (
    <div className="camera-container">
      <video
        ref={videoRef}
        className={`video-element ${isMirrored ? 'mirrored' : ''}`}
        autoPlay
        playsInline
        muted
        style={{ display: isStreaming ? 'block' : 'none' }}
      />
      
      <canvas
        ref={canvasRef}
        className="canvas-overlay"
        style={{ display: isStreaming ? 'block' : 'none' }}
      />
      
      {!isStreaming && (
        <div style={{ textAlign: 'center', padding: '2rem' }}>
          <div style={{ fontSize: '3rem', marginBottom: '1rem' }}>📷</div>
          <h3 style={{ fontSize: '1.2rem', color: '#9CA3AF' }}>Camera Feed Stopped</h3>
          <p style={{ fontSize: '0.9rem', color: '#6B7280', marginTop: '0.5rem' }}>
            Click <strong>Start Camera</strong> to launch live face mask verification.
          </p>
        </div>
      )}

      {errorMsg && (
        <div style={{ position: 'absolute', background: 'rgba(255,46,84,0.9)', color: 'white', padding: '1rem', borderRadius: '12px', textAlign: 'center', margin: '1rem', zIndex: 20 }}>
          ⚠️ {errorMsg}
        </div>
      )}
    </div>
  );
}
