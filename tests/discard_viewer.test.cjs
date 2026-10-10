const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const { resolve } = require("node:path");
const { test } = require("node:test");
const { createContext, runInContext } = require("node:vm");

const html = readFileSync(resolve(__dirname, "../scripts/discard_viewer.html"), "utf8");
const source = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const initial = () => ({ session: "demo", revision: 0, phase: "ROLL", current_player: 0,
  discard_player: null, discard_pending: [0, 0, 0, 0], last_roll: null,
  hands: [[4, 3, 2, 1, 0], [1, 2, 0, 3, 1], [0, 1, 3, 0, 5], [2, 2, 2, 2, 0]],
  bank: [12, 11, 12, 13, 13] });
const pending = () => ({ ...initial(), revision: 1, phase: "DISCARD", last_roll: 7,
  discard_player: 0, discard_pending: [5, 0, 4, 4] });

function harness() {
  const elements = {};
  for (const id of ["seven", "reset", "discard-form", "resource-choices", "confirm",
    "phase-title", "phase-detail", "selection-count", "message", "bank",
    ...Array.from({ length: 5 }, (_, i) => `choose-${i}`),
    ...Array.from({ length: 5 }, (_, i) => `held-${i}`),
    ...Array.from({ length: 4 }, (_, i) => `player-${i}`),
    ...Array.from({ length: 4 }, (_, i) => `player-status-${i}`)]) {
    elements[id] = { value: "0", textContent: "", disabled: false, hidden: false, max: "",
      events: {}, classList: { toggle() {} },
      get valueAsNumber() { return this.value === "" ? NaN : Number(this.value); },
      addEventListener(name, callback) { this.events[name] = callback; } };
  }
  const requests = [];
  const context = createContext({ document: { getElementById: (id) => elements[id] },
    fetch(url, options) { return new Promise((resolve, reject) => requests.push({ url, options, resolve, reject })); }
  });
  runInContext(source, context, { filename: "discard_viewer.html" });
  const choose = (counts) => counts.forEach((n, i) => { elements[`choose-${i}`].value = String(n); elements[`choose-${i}`].events.input(); });
  const submit = () => elements["discard-form"].events.submit({ preventDefault() {} });
  return { elements, requests, choose, submit };
}

const flush = () => new Promise((resolve) => setImmediate(resolve));
async function respond(request, data, ok = true) {
  request.resolve({ ok, json: async () => data });
  await flush();
}

async function startDiscarding() {
  const h = harness();
  await respond(h.requests[0], initial());
  h.elements.seven.events.click();
  await respond(h.requests[1], pending());
  return h;
}

test("sample load and seven create an explicit manual choice", async () => {
  const h = harness();
  assert.equal(h.requests[0].url, "/api/discard/new");
  assert.equal(h.elements.reset.disabled, true);
  await respond(h.requests[0], initial());
  assert.equal(h.elements.seven.disabled, false);
  assert.equal(h.elements["discard-form"].hidden, true);
  h.elements.seven.events.click();
  assert.equal(JSON.parse(h.requests[1].options.body).revision, 0);
  await respond(h.requests[1], pending());
  assert.equal(h.elements["phase-title"].textContent, "Player 1: choose 5 cards");
  assert.equal(h.elements["held-0"].textContent, "In hand: 4");
  assert.equal(h.elements["choose-0"].max, "4");
  assert.equal(h.elements["discard-form"].hidden, false);
  assert.equal(h.elements.confirm.disabled, true);
});

test("exact selected counts are submitted once and next hand clears the selectors", async () => {
  const h = await startDiscarding();
  h.choose([4, 0, 0, 1, 0]);
  assert.equal(h.elements["selection-count"].textContent, "5 of 5 cards selected");
  assert.equal(h.elements.confirm.disabled, false);
  h.submit(); h.submit();
  assert.equal(h.requests.length, 3);
  assert.equal(h.elements.confirm.disabled, true);
  assert.deepEqual(JSON.parse(h.requests[2].options.body), { session: "demo", revision: 1, player: 0, resources: [4, 0, 0, 1, 0] });
  const next = pending(); next.revision = 2; next.discard_player = 2;
  next.discard_pending[0] = 0; next.hands[0] = [0, 3, 2, 0, 0];
  next.bank = [16, 11, 12, 14, 13];
  await respond(h.requests[2], next);
  assert.equal(h.elements["phase-title"].textContent, "Player 3: choose 4 cards");
  assert.equal(h.elements["choose-0"].value, "0");
  assert.equal(h.elements["selection-count"].textContent, "0 of 4 cards selected");
  assert.equal(h.elements["player-status-1"].textContent, "7 cards");
  assert.match(h.elements.bank.textContent, /16 wood/);
});

for (const counts of [[3, 0, 0, 1, 0], [4, 1, 0, 1, 0], [5, 0, 0, 0, 0],
  [-1, 3, 2, 1, 0], [1.5, 2.5, 0, 1, 0], ["", 3, 1, 1, 0]]) {
  test(`invalid selection ${JSON.stringify(counts)} makes no request`, async () => {
    const h = await startDiscarding(); h.choose(counts); h.submit();
    assert.equal(h.elements.confirm.disabled, true);
    assert.equal(h.requests.length, 2);
  });
}

test("server validation and network failures retain the choices and allow retry", async () => {
  const h = await startDiscarding(); h.choose([4, 0, 0, 1, 0]); h.submit();
  await respond(h.requests[2], { error: "Choose exactly five cards." }, false);
  assert.equal(h.elements.message.textContent, "Choose exactly five cards.");
  assert.equal(h.elements["choose-0"].value, "4");
  assert.equal(h.elements.confirm.disabled, false);
  h.submit(); h.requests[3].reject(new Error("Connection lost")); await flush();
  assert.equal(h.elements.message.textContent, "Connection lost");
  assert.equal(h.elements["resource-choices"].disabled, false);
});

test("a stale request recovers the server hand and revision", async () => {
  const h = await startDiscarding(); h.choose([4, 0, 0, 1, 0]); h.submit();
  const next = pending(); next.revision = 2; next.discard_player = 2; next.discard_pending[0] = 0;
  await respond(h.requests[2], { error: "Outdated choice", current: next }, false);
  assert.equal(h.elements["phase-title"].textContent, "Player 3: choose 4 cards");
  assert.equal(h.elements["choose-0"].value, "0");
  h.choose([0, 0, 0, 0, 4]); h.submit();
  assert.equal(JSON.parse(h.requests[3].options.body).revision, 2);
});

test("finishing discards hides the form and reset starts a new round", async () => {
  const h = await startDiscarding(); h.choose([4, 0, 0, 1, 0]); h.submit();
  const done = { ...initial(), phase: "ROBBER_MOVE", revision: 4 };
  await respond(h.requests[2], done);
  assert.equal(h.elements["discard-form"].hidden, true);
  assert.equal(h.elements["phase-title"].textContent, "All discards complete");
  assert.match(h.elements["phase-detail"].textContent, /Player 1 can now move the robber/);
  assert.equal(h.elements.seven.disabled, true);
  h.elements.reset.events.click();
  assert.deepEqual(JSON.parse(h.requests[3].options.body), { seed: 0 });
  await respond(h.requests[3], initial());
  assert.equal(h.elements.seven.disabled, false);
});

test("initial load failure leaves reset available", async () => {
  const h = harness();
  await respond(h.requests[0], { error: "Demo unavailable" }, false);
  assert.equal(h.elements.message.textContent, "Demo unavailable");
  assert.equal(h.elements.reset.disabled, false);
  assert.equal(h.elements.seven.disabled, true);
});
