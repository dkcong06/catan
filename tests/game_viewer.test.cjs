const test = require("node:test");
const assert = require("node:assert/strict");
const { ReplayTimeline } = require("../scripts/game_viewer.js");

test("rewinding changes playback only and preserves the recorded future", () => {
  const replay = new ReplayTimeline({step:0});
  replay.append({step:1,event:{label:"first"}});
  replay.append({step:2,event:{label:"second"}});
  assert.equal(replay.seek(0).step, 0);
  assert.equal(replay.frames.length, 3);
  assert.equal(replay.seek(1).event.label, "first");
  assert.equal(replay.seek(2).event.label, "second");
  replay.append({step:3,terminated:true,winner:1});
  assert.equal(replay.current.winner, 1);
});

test("duplicate/out-of-order frames and invalid seeks cannot corrupt history", () => {
  const replay = new ReplayTimeline({step:0});
  replay.append({step:1});
  for(const frame of [{step:1},{step:3},null])assert.throws(()=>replay.append(frame));
  for(const index of [-1,2,0.5,NaN])assert.throws(()=>replay.seek(index));
  assert.equal(replay.frames.length,2);
  assert.equal(replay.current.step,1);
});

test("terminal wins and action caps both close the replay", () => {
  for(const closure of [{terminated:true},{truncated:true}]){
    const replay=new ReplayTimeline({step:0});
    replay.append({step:1,...closure});
    assert.throws(()=>replay.append({step:2}));
    assert.equal(replay.seek(0).step,0);
  }
});

test("finite 5000-action timeline accepts the final frame and rejects overflow", () => {
  const replay=new ReplayTimeline({step:0});
  for(let i=1;i<=5000;i++)replay.append({step:i});
  assert.equal(replay.frames.length,5001);
  assert.throws(()=>replay.append({step:5001}));
  assert.throws(()=>new ReplayTimeline({step:1}));
});
