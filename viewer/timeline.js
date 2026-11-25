export function setupTimeline({setPaused, showFrame, showLive}) {
  document.querySelector("#timeline").hidden = false;
  const branches = document.querySelector("#history-branch");
  const slider = document.querySelector("#history-position");
  const play = document.querySelector("#history-play");
  const live = document.querySelector("#history-live");
  const fork = document.querySelector("#history-fork");
  const status = document.querySelector("#history-status");
  const ruler = document.querySelector("#history-ruler");
  const rangeLabel = document.querySelector("#history-range");
  const splitSide = document.querySelector("#history-split-side");
  let index;
  let historical = false;
  let cursorTick;
  let selectedFrame;
  let request = 0;
  let timer;
  let playing = false;
  let seekAbort;
  let pause;
  let forking = false;
  let rulerKey;

  const selectedRange = () => index?.branches.find(branch => branch.id === Number(branches.value));
  const seconds = time => `${time.toFixed(2)} s`;

  // A reset adds one frame at the terminal frame's tick. Tabs and storage
  // still address distinct frame IDs while the ruler uses simulation ticks.
  function tickAtFrame(range, id) {
    const duplicates = range.splits.filter(split => split.id > range.first && split.id <= id).length;
    return range.firstTick + id - range.first - duplicates;
  }

  function frameAtTick(range, tick) {
    const duplicates = range.splits.filter(split => split.id > range.first && split.tick <= tick).length;
    return range.first + tick - range.firstTick + duplicates;
  }

  function drawRuler(range) {
    const key = `${range?.id}:${range?.first}:${range?.last}:${ruler.clientWidth}`;
    if (key === rulerKey) return;
    rulerKey = key;
    ruler.replaceChildren();
    if (!range) return;
    const {firstTick, lastTick, splits} = range;
    const first = firstTick * index.controlDt;
    const last = lastTick * index.controlDt;
    const span = last - first;
    const position = t => span > 0 ? `${100 * (t - first) / span}%` : "0%";

    if (span > 0) {
      const minStep = Math.max(index.controlDt, span / Math.max(1, Math.floor(ruler.clientWidth / 84)));
      const magnitude = 10 ** Math.floor(Math.log10(minStep));
      const step = [1, 2, 5, 10].find(n => n * magnitude >= minStep) * magnitude;
      const precision = Math.max(0, Math.ceil(-Math.log10(step)));
      for (let i = Math.ceil(first / step); i * step <= last; i++) {
        const time = i * step;
        const mark = document.createElement("div");
        mark.className = "time-tick";
        mark.style.left = position(time);
        const label = document.createElement("span");
        label.textContent = `${time.toFixed(precision)} s`;
        mark.append(label);
        ruler.append(mark);
      }
    }
    for (const split of splits) {
      const marker = document.createElement("button");
      marker.type = "button";
      marker.className = "episode-marker";
      marker.tabIndex = -1;
      marker.style.left = position(split.tick * index.controlDt);
      marker.title = `Episode split at ${seconds(split.tick * index.controlDt)}`;
      marker.setAttribute("aria-label", marker.title);
      marker.addEventListener("click", () => {
        if (forking) return;
        stopHistory();
        seek(split.id);
      });
      ruler.append(marker);
    }
  }

  function updateControls() {
    const range = selectedRange();
    slider.disabled = !range || forking;
    branches.disabled = forking;
    live.disabled = !historical || forking;
    fork.disabled = !historical || !selectedFrame?.branchable || playing || forking;
    play.disabled = !range || forking;
    play.textContent = playing ? "Pause history" : "Play history";
    splitSide.hidden = !selectedFrame?.split || !range || selectedFrame.split.id <= range.first;
    splitSide.disabled = playing || forking;
    if (selectedFrame?.split) {
      splitSide.textContent = selectedFrame.branchable ? "Show terminal" : "Show reset";
    }
    drawRuler(range);
    if (range) {
      slider.min = range.firstTick;
      slider.max = range.lastTick;
      if (historical) {
        slider.value = cursorTick;
      } else {
        slider.value = range.lastTick;
        status.textContent = `Live · ${seconds(range.lastTick * index.controlDt)}`;
      }
      const first = seconds(range.firstTick * index.controlDt);
      const last = seconds(range.lastTick * index.controlDt);
      const retained = seconds((range.lastTick - range.firstTick) * index.controlDt);
      rangeLabel.textContent = `${first}–${last} · ${retained} retained`;
      slider.setAttribute("aria-valuetext", seconds(Number(slider.value) * index.controlDt));
    } else {
      rangeLabel.textContent = "";
    }
  }

  function stopHistory() {
    playing = false;
    clearTimeout(timer);
    updateControls();
  }

  async function seek(id, {jump = true} = {}) {
    const range = selectedRange();
    if (!range || id < range.first || id > range.last) return;
    const current = ++request;
    historical = true;
    cursorTick = tickAtFrame(range, id);
    selectedFrame = undefined;
    seekAbort?.abort();
    const {signal} = seekAbort = new AbortController();
    updateControls();
    try {
      pause ??= setPaused(true);
      await pause;
      if (signal.aborted) return;
      const response = await fetch(`/timeline/frame/${id}`, {cache: "no-store", signal});
      if (!response.ok) throw new Error("That frame is no longer in history.");
      const frame = await response.json();
      if (current !== request) return;
      const branchable = response.headers.get("X-Timeline-Branchable") === "1";
      const resetId = branchable ? id : id + 1;
      const split = range.splits.find(split => split.id === resetId);
      selectedFrame = {id, branchable, split};
      showFrame(frame, jump);
      const label = !branchable ? " · Terminal frame" : split ? " · Episode reset" : "";
      status.textContent = `${seconds(frame.time)}${label}`;
      updateControls();
    } catch (error) {
      if (current !== request) return;
      pause = undefined;
      stopHistory();
      status.textContent = error.message;
    }
  }

  async function advance(id, jump) {
    const start = performance.now();
    await seek(id, {jump});
    if (!playing || selectedFrame?.id !== id) return;
    const range = selectedRange();
    if (!range || id >= range.last) {
      stopHistory();
      return;
    }
    timer = setTimeout(
      () => advance(id + 1, false),
      Math.max(0, index.controlDt * 1000 - (performance.now() - start))
    );
  }

  play.addEventListener("click", () => {
    if (playing) {
      stopHistory();
      return;
    }
    const range = selectedRange();
    if (!range) return;
    playing = true;
    const current = selectedFrame?.id ?? frameAtTick(range, Number(slider.value));
    advance(!historical || current >= range.last ? range.first : current, true);
  });
  slider.addEventListener("input", () => {
    const tick = Number(slider.value);
    stopHistory();
    const range = selectedRange();
    if (range) seek(frameAtTick(range, tick));
  });
  branches.addEventListener("change", () => {
    stopHistory();
    const range = selectedRange();
    if (range) seek(range.first);
  });
  splitSide.addEventListener("click", () => {
    const {split, branchable} = selectedFrame;
    stopHistory();
    seek(branchable ? split.id - 1 : split.id);
  });
  function returnToLive() {
    request++;
    seekAbort?.abort();
    historical = false;
    pause = undefined;
    selectedFrame = undefined;
    stopHistory();
    branches.value = index.liveBranch;
    updateControls();
    showLive();
  }
  live.addEventListener("click", returnToLive);
  fork.addEventListener("click", async () => {
    if (!selectedFrame?.branchable) return;
    forking = true;
    updateControls();
    try {
      const response = await fetch(`/timeline/branch/${selectedFrame.id}`, {method: "POST"});
      if (!response.ok) throw new Error((await response.json()).error);
      returnToLive();
    } catch (error) {
      status.textContent = error.message;
    } finally {
      forking = false;
      updateControls();
    }
  });

  async function refresh() {
    try {
      const response = await fetch("/timeline.json", {cache: "no-store"});
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      index = await response.json();
      const selection = historical ? branches.value : String(index.liveBranch);
      const ids = index.branches.map(branch => String(branch.id));
      if (ids.join(",") !== Array.from(branches.options, option => option.value).join(",")) {
        branches.replaceChildren(...ids.map(id => new Option(id === "0" ? "Original run" : `Branch ${id}`, id)));
      }
      branches.value = ids.includes(selection) ? selection : index.liveBranch;
      updateControls();
    } catch (error) {
      status.textContent = error.message;
    } finally {
      setTimeout(refresh, 500);
    }
  }
  refresh();
  new ResizeObserver(updateControls).observe(ruler);
  return {
    isHistorical: () => historical,
    returnToLive,
    toggleHistory: () => play.click(),
  };
}
