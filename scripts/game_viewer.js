"use strict";

class ReplayTimeline {
  constructor(initial) {
    if (!initial || initial.step !== 0) throw new Error("Replay must start at action zero");
    this.frames = [initial];
    this.position = 0;
  }
  append(frame) {
    if (!frame || frame.step !== this.frames.length || this.frames.length > 5000)
      throw new Error("Outdated or oversized replay frame");
    const last = this.frames[this.frames.length - 1];
    if (last.terminated || last.truncated) throw new Error("Game has finished");
    this.frames.push(frame);
    this.position = this.frames.length - 1;
  }
  seek(index) {
    if (!Number.isInteger(index) || index < 0 || index >= this.frames.length)
      throw new Error("Replay position outside recorded game");
    this.position = index;
    return this.current;
  }
  get current() { return this.frames[this.position]; }
}

if (typeof module !== "undefined") module.exports = { ReplayTimeline };

if (typeof document !== "undefined") {
  const el = (id) => document.getElementById(id);
  const seats = ["#2589c8", "#dc6755", "#e2a63b", "#696c7d"];
  const terrains = {wood:"#287b53",sheep:"#a0c46d",wheat:"#eacb67",brick:"#c47755",ore:"#a7b5b7",desert:"#ddcfb0"};
  const resources = ["Wood","Sheep","Wheat","Brick","Ore"];
  const cardNames = ["Knight","Victory point","Road Building","Year of Plenty","Monopoly"];
  let names = [], game = null, replay = null, playing = false, busy = false, generation = 0;
  let selectedVertex = null;
  const canvas = el("board"), context = canvas.getContext("2d");

  function text(node, value) { node.textContent = value; }
  function option(value, label) {
    const item = document.createElement("option");
    item.value = value; text(item, label); return item;
  }
  function settings() {
    const count = Number(el("players").value);
    const old = Array.from(el("bots").querySelectorAll("select")).map((item) => item.value);
    el("bots").replaceChildren();
    for (let p = 0; p < count; p++) {
      const label = document.createElement("label"), select = document.createElement("select");
      text(label, "Player " + (p + 1));
      select.id = "bot-" + p; select.setAttribute("aria-label", "Player " + (p + 1) + " bot");
      for (const name of names) select.append(option(name, name.replaceAll("_", " ")));
      select.value = old[p] || (p === 0 && names.includes("original") ? "original" : "tactical");
      label.append(select); el("bots").append(label);
    }
  }
  function stop() { playing = false; generation++; text(el("play"), "Play"); }
  function pending(value) {
    busy = value;
    for (const id of ["new", "next", "players"]) el(id).disabled = value;
  }
  async function post(route, body) {
    const response = await fetch(route, {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
    const value = await response.json();
    if (!response.ok) throw new Error(value.error || "Request failed");
    return value;
  }
  async function newGame() {
    if (busy) return;
    stop(); pending(true); text(el("error"), ""); text(el("status"),"Loading selected bots and board…");
    try {
      const count = Number(el("players").value);
      const seed = Number(el("seed").value);
      if (!Number.isInteger(seed) || seed < 0 || seed > 4294967295) throw new Error("Use an integer seed from 0 to 4294967295");
      game = await post("/api/game/new", {
        seed:seed,players:count,bots:Array.from({length:count},(_,p)=>el("bot-"+p).value),
        deterministic:el("deterministic").checked
      });
      replay = new ReplayTimeline(game.frame); selectedVertex = null;
      text(el("vertex"),"Select a vertex on the board.");
      el("perspective").replaceChildren(option("-1", "Public information"));
      for (let p=0;p<count;p++) el("perspective").append(option(String(p), "P"+(p+1)+" · "+game.bots[p]));
      render();
    } catch (error) { text(el("error"), error.message); }
    finally { pending(false); }
  }
  async function advance() {
    if (busy || !replay) return false;
    if (replay.position < replay.frames.length - 1) {
      replay.seek(replay.position + 1); render(); return true;
    }
    if (replay.current.terminated || replay.current.truncated) { stop(); return false; }
    pending(true);
    try {
      const response = await post("/api/game/step",{session:game.session,revision:game.revision});
      if (response.revision === game.revision) { stop(); return false; }
      replay.append(response.frame); game.revision = response.revision; render();
      if (el("pause-win").checked && response.frame.event && response.frame.event.missed_win) stop();
      return true;
    } catch (error) { stop(); text(el("error"), error.message); return false; }
    finally { pending(false); }
  }
  async function play() {
    if (playing) { stop(); return; }
    if (!replay || busy) return;
    playing = true; const ownGeneration = ++generation; text(el("play"), "Pause");
    while (playing && generation === ownGeneration) {
      if (!await advance()) break;
      await new Promise((resolve)=>setTimeout(resolve, Number(el("speed").value)));
    }
    if (generation === ownGeneration) stop();
  }
  function xy(point) { return [500 + point[0] * 60, 290 + point[1] * 60]; }
  function line(first, second, color, width) {
    context.beginPath();context.moveTo(first[0],first[1]);context.lineTo(second[0],second[1]);
    context.strokeStyle=color;context.lineWidth=width;context.lineCap="round";context.stroke();
  }
  function label(value, point, size, color) {
    context.fillStyle=color;context.font="600 "+size+"px system-ui";context.textAlign="center";
    context.textBaseline="middle";context.fillText(String(value),point[0],point[1]);
  }
  function draw(frame) {
    context.clearRect(0,0,1000,590);
    if (!game) { label("Start a game to watch the bots", [500,290],22,"#6a8173"); return; }
    const geometry = game.geometry, board = game.board;
    geometry.tile_vertices.forEach((vertices,tile)=>{
      context.beginPath();
      vertices.forEach((v,i)=>{const p=xy(geometry.vertex_positions[v]);if(i===0)context.moveTo(...p);else context.lineTo(...p);});
      context.closePath();context.fillStyle=terrains[board.resources[tile]];context.fill();
      context.lineWidth=3;context.strokeStyle="#fafbf6";context.stroke();
      const center=xy(geometry.tile_positions[tile]);
      if(board.numbers[tile]) {
        context.beginPath();context.arc(center[0],center[1],20,0,Math.PI*2);
        context.fillStyle="#fff8e9";context.fill();
        label(board.numbers[tile],center,22,[6,8].includes(board.numbers[tile])?"#b44430":"#263d35");
      }
      label(board.resources[tile], [center[0],center[1]+39],12,"#19372c");
      if(frame.robber===tile) {
        context.beginPath();context.arc(center[0]+37,center[1]-18,11,0,Math.PI*2);
        context.fillStyle="#20322e";context.fill(); label("R",[center[0]+37,center[1]-18],10,"white");
      }
    });
    geometry.port_positions.forEach((point,p)=>{
      const center=xy(point), ends=geometry.port_vertices[p].map(v=>xy(geometry.vertex_positions[v]));
      ends.forEach(end=>line(center,end,"#7b9a89",1.5));
      label(board.ports[p]==="generic"?"3:1":board.ports[p]+" 2:1",center,12,"#425f50");
    });
    frame.edges.forEach((owner,e)=>{
      if(owner<0)return;
      const ends=geometry.edge_vertices[e].map(v=>xy(geometry.vertex_positions[v]));
      line(ends[0],ends[1],"#243c32",11);line(ends[0],ends[1],seats[owner],7);
    });
    const action=frame.event && frame.event.action;
    if(action!==null && action!==undefined && action<72){
      const ends=geometry.edge_vertices[action].map(v=>xy(geometry.vertex_positions[v]));
      line(ends[0],ends[1],"#fff2a3",3);
    }
    frame.vertices.forEach((owner,v)=>{
      if(owner<0)return;
      const point=xy(geometry.vertex_positions[v]),size=frame.buildings[v]===2?11:8;
      context.fillStyle=seats[owner];context.strokeStyle="#263c33";context.lineWidth=2;
      context.beginPath();context.moveTo(point[0]-size,point[1]+size);
      context.lineTo(point[0]-size,point[1]-size/2);context.lineTo(point[0],point[1]-size*1.5);
      context.lineTo(point[0]+size,point[1]-size/2);context.lineTo(point[0]+size,point[1]+size);
      context.closePath();context.fill();context.stroke();
      if(frame.buildings[v]===2)label("C",point,10,"#fff");
    });
    if(selectedVertex!==null){
      const point=xy(geometry.vertex_positions[selectedVertex]);
      context.beginPath();context.arc(point[0],point[1],17,0,Math.PI*2);context.strokeStyle="#172f26";context.lineWidth=2;context.stroke();
    }
  }
  function render() {
    if(!replay)return;
    const frame=replay.current,event=frame.event;
    draw(frame);el("scores").replaceChildren();
    frame.players.forEach((player,p)=>{
      const card=document.createElement("div");card.className="score"+(p===frame.actor?" active":"");
      card.style.setProperty("--seat",seats[p]);
      const title=document.createElement("span");text(title,"PLAYER "+(p+1));
      const bot=document.createElement("div");bot.className="bot";text(bot,game.bots[p]);
      const vp=document.createElement("strong");text(vp,player.public_vp+" ");
      const subtitle=document.createElement("span");text(subtitle,"public VP");
      const detail=document.createElement("div");detail.className="sub";
      text(detail,player.resources+" resources · "+player.cards+" dev cards");
      const road=document.createElement("div");road.className="sub";text(road,"Road "+player.road+" · Knights "+player.knights);
      card.append(title,bot,vp,subtitle,detail,road);el("scores").append(card);
    });
    el("scores").style.gridTemplateColumns="repeat("+frame.players.length+",1fr)";
    text(el("phase"),frame.phase.replaceAll("_"," ")+" · P"+(frame.actor+1));
    text(el("roll"),"Dice "+(frame.roll===null?"—":frame.roll));
    text(el("awards"),"Longest road "+(frame.longest<0?"—":"P"+(frame.longest+1))+" · Largest army "+(frame.army<0?"—":"P"+(frame.army+1)));
    text(el("status"),"Seed "+game.seed+" · "+(frame.terminated?"P"+(frame.winner+1)+" wins":frame.truncated?"Action limit reached; no winner":replay.position<replay.frames.length-1?"Recorded action — playback does not change the game":"Live match ready"));
    text(el("position"),replay.position+" / "+(replay.frames.length-1)+" actions");
    el("timeline").max=String(replay.frames.length-1);el("timeline").value=String(replay.position);
    text(el("actor"),event?"P"+(event.player+1)+" · "+event.phase.replaceAll("_"," "):"Initial board");
    text(el("move"),event?event.label:"Opening placement");
    el("alert").hidden=!(event&&event.winning.length);
    if(event&&event.winning.length)text(el("alert"),event.missed_win?"Missed a guaranteed win.":"Chose a verified winning action/road plan.");
    el("alternatives").replaceChildren();
    if(event){
      for(const item of event.top){
        const row=document.createElement("div");row.className="alternative"+(item.winning?" good":"");
        const probability=document.createElement("strong");text(probability,(item.probability*100).toFixed(1)+"%");
        row.append(probability,document.createTextNode(item.label+(item.winning?" · winning":"")));el("alternatives").append(row);
      }
      if(event.missed_win)for(const item of event.winning){
        const row=document.createElement("div");row.className="alternative good";text(row,"Available win: "+item.label);el("alternatives").append(row);
      }
    }
    text(el("value"),event&&event.value!==null?"Critic estimate "+event.value.toFixed(3)+" (not a calibrated win probability)":"Reference heuristic · no neural probability ranking");
    const perspective=Number(el("perspective").value);
    if(perspective<0)text(el("hand"),"Choose a player to inspect its private hand and held development cards.");
    else {
      const hand=frame.hands[perspective].map((n,i)=>resources[i]+": "+n).join(" · ");
      const cards=frame.cards[perspective].map((n,i)=>n?cardNames[i]+": "+n+(frame.new_cards[perspective][i]?" ("+frame.new_cards[perspective][i]+" new)":""):"").filter(Boolean).join(" · ");
      text(el("hand"),"Own VP: "+frame.players[perspective].own_vp+". "+hand+". Development cards: "+(cards||"none")+".");
    }
    el("log").replaceChildren();
    const start=Math.max(1,replay.position-149);
    for(let i=start;i<=replay.position;i++){
      const item=replay.frames[i].event;if(!item)continue;
      const button=document.createElement("button");button.className=(i===replay.position?"current ":"")+(item.missed_win?"flag":"");
      text(button,i+" · P"+(item.player+1)+" · "+item.label+(item.missed_win?" ⚑":""));
      button.addEventListener("click",()=>{stop();replay.seek(i);render();});el("log").append(button);
    }
    el("log").scrollTop=el("log").scrollHeight;
  }
  el("new").addEventListener("click",newGame);
  el("players").addEventListener("change",settings);
  el("play").addEventListener("click",play);
  el("next").addEventListener("click",()=>{stop();advance();});
  el("back").addEventListener("click",()=>{stop();if(replay){replay.seek(Math.max(0,replay.position-1));render();}});
  el("latest").addEventListener("click",()=>{stop();if(replay){replay.seek(replay.frames.length-1);render();}});
  el("timeline").addEventListener("input",()=>{stop();if(replay){replay.seek(Number(el("timeline").value));render();}});
  el("perspective").addEventListener("change",render);
  el("export").addEventListener("click",()=>{
    if(!replay)return;
    const data={schema:1,seed:game.seed,bots:game.bots,board:game.board,geometry:game.geometry,
      deterministic:game.deterministic,policy_seed:game.policy_seed,max_actions:game.max_actions,
      debug_private_information:true,frames:replay.frames};
    const url=URL.createObjectURL(new Blob([JSON.stringify(data)],{type:"application/json"}));
    const link=document.createElement("a");link.href=url;link.download="catan-"+game.seed+"-replay.json";link.click();
    setTimeout(()=>URL.revokeObjectURL(url),1000);
  });
  canvas.addEventListener("click",(event)=>{
    if(!game)return;
    const rect=canvas.getBoundingClientRect(),x=(event.clientX-rect.left)*1000/rect.width,y=(event.clientY-rect.top)*590/rect.height;
    let best=null,distance=22;
    game.geometry.vertex_positions.forEach((point,v)=>{const p=xy(point),d=Math.hypot(p[0]-x,p[1]-y);if(d<distance){distance=d;best=v;}});
    if(best===null)return;
    selectedVertex=best;
    const tiles=game.geometry.tile_vertices.map((vertices,t)=>vertices.includes(best)?t:-1).filter(t=>t>=0);
    const pips=tiles.reduce((n,t)=>n+(game.board.numbers[t]===0?0:6-Math.abs(7-game.board.numbers[t])),0);
    const ports=game.geometry.port_vertices.map((vertices,p)=>vertices.includes(best)?game.board.ports[p]:null).filter(Boolean);
    text(el("vertex"),"Vertex "+best+" · "+tiles.map(t=>game.board.resources[t]+" "+(game.board.numbers[t]||"desert")).join(" / ")+". Gross production "+pips+"/36 per roll before robber/bank shortages. Port: "+(ports.join(", ")||"none")+".");
    draw(replay.current);
  });
  async function experimentProgress() {
    try {
      const response=await fetch("/api/experiment");
      if(!response.ok)return;
      const info=await response.json();
      let message=info.status;
      if(info.replicate!==undefined)message+=" · replicate "+(info.replicate+1);
      if(info.arm)message+=" · "+info.arm;
      if(info.update!==undefined)message+=" · cohort "+info.update+(info.total_cohorts?"/"+info.total_cohorts:"");
      if(info.passes!==undefined)message+=" · teaching pass "+info.passes;
      if(info.practice_games!==undefined)message+=". Practice wins "+info.learner_wins+"/"+info.practice_games;
      const score=info.validation||(info.warmup&&info.warmup.validation);
      if(score&&score.winning_argmax!==null)message+=". Winning choices on tactical validation: "+(score.winning_argmax*100).toFixed(1)+"%";
      text(el("experiment"),message);
    } catch (_) { text(el("experiment"),"Experiment progress unavailable"); }
  }
  experimentProgress();setInterval(experimentProgress,5000);
  fetch("/api/game/bots").then(response=>response.json()).then(data=>{names=data.bots;settings();draw(null);}).catch(error=>text(el("error"),error.message));
}
