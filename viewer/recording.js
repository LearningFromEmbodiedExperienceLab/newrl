const MAX_CHUNK_BYTES = 4 * 1024 * 1024;
const MAX_PENDING_BYTES = 64 * 1024 * 1024;

async function post(path, body, options = {}) {
  const response = await fetch(`/recordings/${path}`, {method: "POST", body, ...options});
  const result = await response.json();
  if (!response.ok) throw new Error(result.error ?? `HTTP ${response.status}`);
  return result;
}

export function setupRecording(canvas, onComplete) {
  const button = document.querySelector("#record");
  const status = document.querySelector("#record-status");
  const formats = [
    ["video/mp4;codecs=avc1", "mp4"],
    ["video/mp4;codecs=avc1.42002a", "mp4"],
    ["video/webm;codecs=vp9", "webm"],
    ["video/webm;codecs=vp8", "webm"],
    ["video/webm", "webm"],
  ];
  const format = canvas.captureStream && globalThis.MediaRecorder
    ? formats.find(([mime]) => MediaRecorder.isTypeSupported(mime))
    : undefined;
  if (!format) {
    status.textContent = "Video recording is unavailable in this browser.";
    return () => false;
  }
  const [mimeType, extension] = format;
  let active;
  button.disabled = false;

  function complete(recording, message, cancel = false) {
    if (active !== recording) return;
    active = undefined;
    if (cancel) {
      recording.abort.abort();
      if (recording.recorder?.state === "recording") recording.recorder.stop();
      if (recording.id) post(`${recording.id}/cancel`).catch(() => {});
    }
    recording.stream?.getTracks().forEach(track => track.stop());
    button.textContent = "Record";
    button.disabled = false;
    status.textContent = message;
    onComplete();
  }

  button.addEventListener("click", async () => {
    button.disabled = true;
    if (active) {
      if (active.recorder.state !== "inactive") active.recorder.stop();
      status.textContent = "Saving…";
      return;
    }

    const recording = {abort: new AbortController(), uploads: Promise.resolve(), pendingBytes: 0};
    active = recording;
    status.textContent = "Starting…";
    const fail = error => complete(recording, `Recording failed. ${error.message}`, true);
    try {
      const {id} = await post(`start/${extension}`);
      recording.id = id;
      const stream = recording.stream = canvas.captureStream(30);
      const recorder = recording.recorder = new MediaRecorder(stream, {
        mimeType, videoBitsPerSecond: 12_000_000,
      });
      recorder.addEventListener("error", event => fail(event.error));
      recorder.addEventListener("dataavailable", ({data}) => {
        if (active !== recording || !data.size) return;
        recording.pendingBytes += data.size;
        if (recording.pendingBytes > MAX_PENDING_BYTES) {
          fail(new Error("Video uploads could not keep pace with recording."));
          return;
        }
        recording.uploads = recording.uploads.then(async () => {
          if (active !== recording) return;
          for (let offset = 0; offset < data.size; offset += MAX_CHUNK_BYTES) {
            await post(`${id}/chunk`, data.slice(offset, offset + MAX_CHUNK_BYTES), {
              signal: recording.abort.signal,
            });
          }
          recording.pendingBytes -= data.size;
        }).catch(fail);
      });
      recorder.addEventListener("stop", async () => {
        if (active !== recording) return;
        button.disabled = true;
        status.textContent = "Saving…";
        await recording.uploads;
        if (active !== recording) return;
        try {
          const {path} = await post(`${id}/finish`);
          complete(recording, `Saved ${path}`);
        } catch (error) {
          fail(error);
        }
      });
      recorder.start(1000);
      button.textContent = "Stop & Save";
      button.disabled = false;
      status.textContent = `Recording ${extension.toUpperCase()}…`;
    } catch (error) {
      fail(error);
    }
  });

  addEventListener("pagehide", () => {
    if (active?.id) post(`${active.id}/cancel`, undefined, {keepalive: true}).catch(() => {});
  });
  return () => active !== undefined;
}
