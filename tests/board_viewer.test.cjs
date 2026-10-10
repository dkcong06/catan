// Exercise the actual inline script with controlled DOM, canvas, and fetch adapters.
// Browser layout/appearance is verified separately in the live viewer.
const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const { resolve } = require("node:path");
const { test } = require("node:test");
const { createContext, runInContext } = require("node:vm");

const html = readFileSync(resolve(__dirname, "../scripts/board_viewer.html"), "utf8");
const source = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const points = (n) => Array.from({ length: n }, (_, i) => [i % 6, Math.floor(i / 6)]);
const fixture = (seed) => ({
  seed,
  tile_positions: points(19), vertex_positions: points(54), edge_positions: points(72),
  port_positions: points(9),
  tile_vertices: Array.from({ length: 19 }, () => [0, 1, 2, 3, 4, 5]),
  edge_vertices: Array.from({ length: 72 }, () => [0, 1]),
  port_vertices: Array.from({ length: 9 }, (_, i) => [i * 2, i * 2 + 1]),
  resources: Array(19).fill("wood"), numbers: Array(19).fill(5), robber_tile: 0,
  ports: Array.from({ length: 9 }, (_, i) => ({ type: i < 4 ? "generic" : "wood", ratio: i < 4 ? 3 : 2 })),
});

function harness() {
  const labels = [];
  const drawing = new Proxy({ fillText: (text) => labels.push(text) }, {
    get: (target, name) => target[name] ?? (() => {}),
  });
  const elements = {};
  for (const id of ["seed", "seed-form", "random-seed", "status", "board-title", "board-canvas"]) {
    elements[id] = {
      value: "0", textContent: "", style: {}, events: {},
      get valueAsNumber() { return this.value === "" ? NaN : Number(this.value); },
      addEventListener(name, callback) { this.events[name] = callback; },
      getContext: () => drawing,
    };
  }
  const requests = [];
  const sandbox = createContext({
    document: { getElementById: (id) => elements[id] },
    window: { devicePixelRatio: 2 }, AbortController, Uint32Array,
    crypto: { getRandomValues(array) { array[0] = 4294967295; return array; } },
    fetch(url, options) {
      return new Promise((resolve, reject) => requests.push({ url, options, resolve, reject }));
    },
  });
  runInContext(source, sandbox, { filename: "board_viewer.html" });
  return { elements, requests, labels };
}

const flush = () => new Promise((resolve) => setImmediate(resolve));
async function respond(request, seed, ok = true) {
  request.resolve({ ok, json: async () => fixture(seed) });
  await flush();
}

test("initial render draws ports and vertices at high pixel density", async () => {
  const { elements, requests, labels } = harness();
  assert.equal(requests[0].url, "/api/board?seed=0");
  await respond(requests[0], 0);
  assert.equal(elements["status"].textContent, "19 tiles, 54 vertices, 72 edges, 9 ports");
  assert.equal(labels.filter((text) => /^v\d+$/.test(text)).length, 54);
  assert.equal(labels.filter((text) => text === "3:1" || text === "2:1").length, 9);
  assert.ok(elements["board-canvas"].width > 0 && elements["board-canvas"].height > 0);
});

test("valid scientific notation is normalized to an integer API seed", async () => {
  const { elements, requests } = harness();
  await respond(requests[0], 0);
  elements.seed.value = "1e3";
  let prevented = false;
  elements["seed-form"].events.submit({ preventDefault() { prevented = true; } });
  assert.equal(prevented, true);
  assert.equal(requests[1].url, "/api/board?seed=1000");
  await respond(requests[1], 1000);
  assert.equal(elements["board-title"].textContent, "Board setup (seed 1000)");
});

test("a late response cannot overwrite the most recently requested board", async () => {
  const { elements, requests } = harness();
  await respond(requests[0], 0);
  const submit = () => elements["seed-form"].events.submit({ preventDefault() {} });
  elements.seed.value = "1";
  submit();
  elements.seed.value = "2";
  submit();
  assert.equal(requests[1].options.signal.aborted, true);
  await respond(requests[2], 2);
  // Simulate a transport that finishes JSON parsing despite aborting the fetch.
  await respond(requests[1], 1);
  assert.equal(elements["board-title"].textContent, "Board setup (seed 2)");
});

test("an aborted response is silent and real errors recover on the next request", async () => {
  const { elements, requests } = harness();
  elements.seed.value = "42";
  elements["seed-form"].events.submit({ preventDefault() {} });
  requests[0].reject(Object.assign(new Error("aborted"), { name: "AbortError" }));
  await respond(requests[1], 42, false);
  assert.equal(elements.status.textContent, "The board could not be loaded.");
  elements["random-seed"].events.click();
  assert.equal(requests[2].url, "/api/board?seed=4294967295");
  await respond(requests[2], 4294967295);
  assert.equal(elements["board-title"].textContent, "Board setup (seed 4294967295)");
  assert.equal(elements.status.textContent, "19 tiles, 54 vertices, 72 edges, 9 ports");
});

test("a late HTTP failure cannot replace the latest successful status", async () => {
  const { elements, requests } = harness();
  elements.seed.value = "42";
  elements["seed-form"].events.submit({ preventDefault() {} });
  await respond(requests[1], 42);
  await respond(requests[0], 0, false);
  assert.equal(elements.status.textContent, "19 tiles, 54 vertices, 72 edges, 9 ports");
});

for (const seed of ["", "-1", "1.5", "4294967296"]) {
  test(`invalid numeric seed ${JSON.stringify(seed)} sends no request`, async () => {
    const { elements, requests } = harness();
    await respond(requests[0], 0);
    elements.seed.value = seed;
    elements["seed-form"].events.submit({ preventDefault() {} });
    await flush();
    assert.equal(requests.length, 1);
    assert.equal(elements.status.textContent, "Enter a seed from 0 to 4294967295.");
  });
}
