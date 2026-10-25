/* Parameter Studio. All observations and state changes go through the local engine. */
"use strict";
const $ = (id) => document.getElementById(id);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
const NS = "http://www.w3.org/2000/svg";
const COLORS = ["#b8a0ff", "#c4e78c", "#f1ac79", "#87c9d1", "#df99c0", "#e0cf8f"];
const VIEW_NAMES = {world:"World", timeline:"Timeline", microscope:"Microscope", data:"Data lab", journal:"Journal", artifacts:"Artifacts",nodes:"Node lab",plugins:"Plugins",transformers:"Transformers"};
let studio = {workspace:{view:{x:0,y:0,zoom:1}},runs:[]};
let capabilities = null;
let view = "world";
let selectedId = null;
let selectionPending = false;
let lastSelectedId = null;
let camera = {x:0,y:0,zoom:1};
let cameraLoaded = false;
let metric = "w";
let pinnedStep = null;
let pinnedRecord = null;
let pinnedRunId = null;
let recordRequest = 0;
let pollBusy = false;
let actionBusy = false;
let actionQueue = Promise.resolve();
let stateGeneration = 0;
let workspaceNameDirty = false;
let workspaceNameTimer = null;
let configJSONDirty = false;
let lastChartSignature = "";
let lastMicroscopeSignature = "";
let lastSideSignature = "";
let lastCheckpointSignature = "";
let lastEventsSignature = "";
let configRunId = null;
let configDraft = null;
let datasetRunId = null;
let datasetDirty = false;
let datasetEvidence = null;
let notesRunId = null;
let workspaceNotesLoaded = false;
let editBits = null;
let bitDirty = false;
let nodeDrag = null;
let panDrag = null;
let cameraSaveTimer = null;
let dataPreviewTimer = null;
let textCallback = null;
let firstRender = true;
let lastSaveError = "";
let optimizerEditRunId = null;
const experimentDrafts = new Map();
let studioDrafts = {schema_version:"spobs-ui-drafts/1",runs:{}};
let draftSaveTimer = null;
let draftsLoaded = false;
let draftsDirty = false;
let draftVersion = 0;
let draftsRevision = 0;
let draftsConflict = false;
let draftsConflictDialog = null;
let importingProject = false;
let workspaceEpoch = null;
let workspaceEpochGeneration = 0;
let workspaceReplaced = false;
let workspaceReplacementDialog = null;
function serializeDrafts(){
  captureExperimentDraft(selectedId);
  const runs={};
  for(const [id,draft] of experimentDrafts){if(!studio.runs.some(item=>item.id===id))continue;const value={};for(const [local,wire] of [["config","config"],["configJSON","config_json"],["configJSONDirty","config_json_dirty"],["csv","csv"],["notes","notes"],["bits","bits"],["bitDirty","bit_dirty"],["optimizer","optimizer_json"]])if(draft[local]!==undefined)value[wire]=draft[local];runs[id]=value;}
  return {schema_version:"spobs-ui-drafts/1",runs,workspace_notes:workspaceNotesLoaded?$("workspace-notes").value:studio.workspace.notes||"",...(window.NodeLab?.serializeDrafts?.()||{})};
}
function markDraftsDirty(){if(!draftsLoaded)return;draftsDirty=true;draftVersion++;clearTimeout(draftSaveTimer);if(!draftsConflict&&!importingProject&&!workspaceReplaced)draftSaveTimer=setTimeout(()=>flushDrafts().catch(error=>toast(`Draft save: ${error.message}`,"error")),1000);}
async function flushDrafts(){
  clearTimeout(draftSaveTimer);if(!draftsLoaded||!draftsDirty)return;
  if(draftsConflict){showDraftConflict();throw new Error("Another tab changed the saved drafts. Resolve the draft conflict to save this tab's edits.");}
  return enqueueMutation(async()=>{
    if(!draftsDirty)return;
    if(importingProject)return;
    if(draftsConflict){showDraftConflict();throw new Error("Resolve the editor-draft conflict before saving.");}
    const version=draftVersion,payload=serializeDrafts();actionBusy=true;stateGeneration++;
    try{const response=await request("/api/action",{method:"POST",headers:{"Content-Type":"application/json"},body:encodeJSON({action:"workspace.drafts",drafts:payload,expected_revision:draftsRevision})});draftsRevision=response.state?.ui_drafts_revision??response.result?.revision??draftsRevision+1;studioDrafts=payload;if(response.state)acceptState(response.state);if(version===draftVersion)draftsDirty=false;}
    catch(error){if(error.status===409&&error.type!=="StudioEpochError"){draftsConflict=true;clearTimeout(draftSaveTimer);showDraftConflict();throw new Error("Another tab changed the saved drafts. This tab's editors have been retained; choose which drafts to save.");}throw error;}
    finally{actionBusy=false;renderTransport();}
  });
}
async function loadDrafts(replaceEditors=false){
  clearTimeout(draftSaveTimer);const payload=await request("/api/ui/drafts");studioDrafts=payload.drafts||payload;draftsRevision=payload.revision??0;experimentDrafts.clear();
  for(const [id,value]of Object.entries(studioDrafts.runs||{})){const draft={};for(const [wire,local]of [["config","config"],["config_json","configJSON"],["config_json_dirty","configJSONDirty"],["csv","csv"],["notes","notes"],["bits","bits"],["bit_dirty","bitDirty"],["optimizer_json","optimizer"]])if(value[wire]!==undefined)draft[local]=value[wire];experimentDrafts.set(id,draft);}
  workspaceNotesLoaded=studioDrafts.workspace_notes!==undefined;$("workspace-notes").value=studioDrafts.workspace_notes??studio.workspace.notes??"";
  draftsLoaded=true;draftsDirty=false;draftsConflict=false;window.NodeLab?.applyDrafts?.(studioDrafts,{replace:replaceEditors});
}
function showDraftConflict(){
  if(draftsConflictDialog?.open)return;
  const dialog=element("dialog","studio-dialog"),heading=element("h2","","Choose the editor drafts to keep."),description=element("p","","Another tab saved editor drafts after this tab loaded them. Your current text is still here. Keep this tab's edits replaces the saved draft set with this tab's editors. Load saved edits replaces this tab's unapplied buffers with the saved set. Applied experiments and numerical history are separate."),error=element("p","error-box");error.hidden=true;
  const actions=element("div","graph-dialog-actions"),keep=element("button","button primary","Keep this tab's edits"),reload=element("button","button","Load saved edits");keep.type=reload.type="button";
  keep.addEventListener("click",async()=>{keep.disabled=reload.disabled=true;try{const latest=await request("/api/ui/drafts");draftsRevision=latest.revision??0;draftsConflict=false;draftsDirty=true;await flushDrafts();dialog.close();toast("This tab's editor drafts were saved explicitly.","success");}catch(problem){draftsConflict=true;error.hidden=false;error.textContent=problem.message;}finally{keep.disabled=reload.disabled=false;}});
  reload.addEventListener("click",async()=>{keep.disabled=reload.disabled=true;try{await loadDrafts(true);configRunId=null;datasetRunId=null;notesRunId=null;lastSelectedId=null;datasetDirty=false;bitDirty=false;editBits=null;lastMicroscopeSignature="";acceptState(studio);if(window.NodeLab)await window.NodeLab.reload(true);dialog.close();toast("Saved editor drafts loaded. Applied experiment state was retained.","success");}catch(problem){error.hidden=false;error.textContent=problem.message;}finally{keep.disabled=reload.disabled=false;}});
  actions.append(keep,reload);dialog.append(heading,description,error,actions);document.body.append(dialog);draftsConflictDialog=dialog;dialog.addEventListener("close",()=>{draftsConflictDialog=null;dialog.remove();});dialog.showModal();
}
function handleWorkspaceReplacement(){
  if(!workspaceReplaced){workspaceReplaced=true;draftsDirty=true;stateGeneration++;clearTimeout(draftSaveTimer);clearTimeout(cameraSaveTimer);clearTimeout(workspaceNameTimer);window.NodeLab?.quiesce();draftsConflictDialog?.close();}
  if(workspaceReplacementDialog?.open)return;
  const dialog=element("dialog","studio-dialog"),heading=element("h2","","The server workspace changed."),description=element("p","","Another tab opened a project. This tab's editor buffers and graph are still here, and its writes are paused so they cannot overwrite the replacement project. Download a local draft recovery file before reloading if you want to keep these edits. That file contains local editor drafts and the current graph; numerical histories remain part of their original project."),error=element("p","error-box");error.hidden=true;
  const actions=element("div","graph-dialog-actions"),download=element("button","button","Download this tab's drafts"),reload=element("button","button primary","Reload saved project");download.type=reload.type="button";
  download.addEventListener("click",()=>{const recovery={schema_version:"spobs-local-recovery/1",workspace_name:studio.workspace.name,drafts:serializeDrafts(),graph:window.NodeLab?.getDraft?.()||null};const encode=window.GraphModel?.stringify||((value)=>encodeJSON(value,null,2));downloadBlob(new Blob([encode(recovery,2)],{type:"application/json"}),"parameter-studio-local-drafts.json");});
  reload.addEventListener("click",async()=>{reload.disabled=download.disabled=true;try{await enqueueMutation(async()=>{importingProject=true;stateGeneration++;await window.NodeLab?.quiesce();const next=await request("/api/state",{acceptWorkspaceEpoch:true,omitWorkspaceEpoch:true});experimentDrafts.clear();cameraLoaded=false;configRunId=null;datasetRunId=null;notesRunId=null;lastSelectedId=null;datasetDirty=false;bitDirty=false;editBits=null;workspaceNotesLoaded=false;workspaceNameDirty=false;lastMicroscopeSignature="";await loadDrafts(true);acceptState(next);if(window.NodeLab)await window.NodeLab.reload(true);});dialog.close();toast("Replacement project loaded. Experiments are shown in their saved state.","success");}catch(problem){error.hidden=false;error.textContent=problem.message;}finally{importingProject=false;window.NodeLab?.resume();reload.disabled=download.disabled=false;}});
  actions.append(download,reload);dialog.append(heading,description,error,actions);document.body.append(dialog);workspaceReplacementDialog=dialog;dialog.addEventListener("close",()=>{workspaceReplacementDialog=null;dialog.remove();});dialog.showModal();
}
function captureExperimentDraft(id){
  if(!id)return;const previous=experimentDrafts.get(id)||{};
  if(configRunId===id&&configDraft){previous.config=GraphModel.clone(configDraft);previous.configJSON=$("config-json").value;previous.configJSONDirty=configJSONDirty;}
  if(datasetRunId===id&&datasetDirty)previous.csv=$("dataset-csv").value;
  if(notesRunId===id)previous.notes=$("run-notes").value;
  if(optimizerEditRunId===id)previous.optimizer=$("optimizer-state-json")?.value;
  if(bitDirty){previous.bits=editBits;previous.bitDirty=true;}
  else if(pinnedStep===null){delete previous.bits;delete previous.bitDirty;}
  experimentDrafts.set(id,previous);
}
const run = () => studio.runs.find((entry) => entry.id === selectedId) || null;
const colorOf = (id) => COLORS[Math.max(0, studio.runs.findIndex((entry) => entry.id === id)) % COLORS.length];
const finite = (value) => typeof value === "number" && Number.isFinite(value);
const scalarText=value=>Object.is(value,-0)?"-0":String(value);
function encodeJSON(value,replacer=null,space=0){return window.GraphModel?.stringify?window.GraphModel.stringify(value,space):JSON.stringify(value,replacer,space);}
const extent = (values) => values.reduce((bounds,value)=>[Math.min(bounds[0],value),Math.max(bounds[1],value)],[Infinity,-Infinity]);
const fmt = (value, digits = 6) => {
  if (value === null || value === undefined) return "—";
  if (!finite(value)) return scalarText(value);
  if (value === 0) return Object.is(value,-0) ? "−0" : "0";
  if (Math.abs(value) < .0001 || Math.abs(value) >= 1e6) return value.toExponential(Math.max(1, digits - 2));
  return Number(value.toPrecision(digits)).toString();
};
const titleCase = (value) => String(value || "idle").replaceAll("_", " ").replace(/\b\w/g, c => c.toUpperCase());
const slug = (value) => String(value || "experiment").replace(/[^a-z0-9_-]+/gi,"-").replace(/^-|-$/g, "").slice(0,80) || "experiment";
const isRunning = (entry) => String(entry?.state).toLowerCase() === "running";
const isFinished = (entry) => ["finished","aborted","stopped"].includes(String(entry?.state).toLowerCase());
function element(tag, cls, text) { const node = document.createElement(tag); if (cls) node.className = cls; if (text !== undefined) node.textContent = text; return node; }
function svgElement(tag, attrs = {}, text) { const node = document.createElementNS(NS, tag); for (const [key, value] of Object.entries(attrs)) node.setAttribute(key,scalarText(value)); if(text!==undefined) node.textContent=text; return node; }
function toast(message, kind = "") { const node=element("div",`toast ${kind}`,message); $("toasts").append(node); setTimeout(()=>node.remove(),kind==="error"?9000:4200); }
function updateConnection(ok) { $("connection").textContent=ok?"Connected":"Reconnecting"; $("connection").className=`connection ${ok?"online":"offline"}`; }
async function request(path, options = {}) {
  const {jsonParser,binary=false,acceptWorkspaceEpoch=false,omitWorkspaceEpoch=false,...fetchOptions}=options;
  if(workspaceReplaced&&!acceptWorkspaceEpoch){handleWorkspaceReplacement();throw new Error("The server workspace changed. Reload it before continuing.");}
  const generation=workspaceEpochGeneration,headers=new Headers(fetchOptions.headers||{});if(workspaceEpoch&&!omitWorkspaceEpoch)headers.set("X-Studio-Epoch",workspaceEpoch);
  let response;
  try { response=await fetch(path,{cache:"no-store",...fetchOptions,headers}); }
  catch (error) { updateConnection(false); throw new Error(`Cannot reach the local studio engine. ${error.message}`); }
  if(generation!==workspaceEpochGeneration&&!acceptWorkspaceEpoch)throw new Error("The project changed while this request was in flight; its stale response was discarded.");
  if(binary&&response.ok){updateConnection(true);return response;}
  const type=response.headers.get("content-type") || "";
  const parser=jsonParser||((path.startsWith("/api/graph/")||path==="/api/ui/drafts")&&window.GraphModel?.parseTransportJSON)||JSON.parse;
  const text=await response.text(),data=type.includes("json")?parser(text):text;
  if (!response.ok || (data && typeof data==="object" && data.ok===false)){const error=new Error(data?.error || data?.message || `Request failed (${response.status})`);error.status=response.status;error.type=data?.type;if(data?.type==="StudioEpochError"){handleWorkspaceReplacement();error.workspaceReplaced=true;}throw error;}
  const epoch=data?.workspace_epoch||data?.state?.workspace_epoch;
  if(epoch){if(acceptWorkspaceEpoch){if(workspaceEpoch!==epoch)workspaceEpochGeneration++;workspaceEpoch=epoch;workspaceReplaced=false;}else if(!workspaceEpoch)workspaceEpoch=epoch;else if(epoch!==workspaceEpoch){handleWorkspaceReplacement();throw new Error("The server workspace changed. Local editor buffers were retained.");}}
  if(workspaceReplaced&&!acceptWorkspaceEpoch)throw new Error("The server workspace changed. Reload it before continuing.");
  updateConnection(true);
  return data;
}
function act(action, data = {}, options = {}) {
  if(workspaceReplaced){handleWorkspaceReplacement();return Promise.resolve(null);}
  if(importingProject){toast("A project is opening. Wait for it to finish before changing this workspace.");return Promise.resolve(null);}
  if(selectionPending&&((action.startsWith("run.")&&!["run.select","run.create"].includes(action))||action.startsWith("checkpoint."))){toast("Wait for the selected experiment to load before using experiment controls.");return Promise.resolve(null);}
  const captured={run_id:selectedId,...data};
  const operation=()=>performAct(action,captured,options);
  return enqueueMutation(operation);
}
function enqueueMutation(operation){
  const result=actionQueue.then(operation,operation);
  actionQueue=result.catch(()=>null);
  return result;
}
async function performAct(action, data = {}, options = {}) {
  actionBusy=true;
  stateGeneration++;
  const buttons=options.button?[options.button]:[];
  buttons.forEach(button=>button.disabled=true);
  try {
    const response=await request("/api/action",{method:"POST",headers:{"Content-Type":"application/json"},body:encodeJSON({action,run_id:selectedId,...data})});
    if(response.state) acceptState(response.state); else await refresh();
    if(options.toast) toast(options.toast,"success");
    return response.result || response;
  } catch(error) { toast(error.message,"error"); return null; }
  finally { actionBusy=false; buttons.forEach(button=>button.disabled=false); renderTransport(); }
}
async function refresh() {
  if(pollBusy||actionBusy||workspaceReplaced) return;
  pollBusy=true;
  const generation=stateGeneration;
  try { const next=await request("/api/state");if(generation===stateGeneration)acceptState(next); }
  catch(error) { if(firstRender) $("workspace-summary").textContent="Waiting for the local engine…"; }
  finally {pollBusy=false;}
}
function acceptState(next) {
  if(!next || !next.workspace || !Array.isArray(next.runs)) return;
  studio=next;
  selectedId=next.workspace.active_run_id || next.runs[0]?.id || null;
  if(!cameraLoaded) {camera={x:0,y:0,zoom:1,...next.workspace.view};cameraLoaded=true;applyCamera();}
  if(selectedId!==lastSelectedId) {
    captureExperimentDraft(lastSelectedId);
    pinnedStep=null;pinnedRecord=null;pinnedRunId=null;bitDirty=false;editBits=null;
    configRunId=null;datasetRunId=null;notesRunId=null;datasetDirty=false;optimizerEditRunId=null;datasetEvidence=null;
    lastMicroscopeSignature="";lastChartSignature="";lastEventsSignature="";
    lastSelectedId=selectedId;
    const draft=experimentDrafts.get(selectedId);
    if(draft?.bitDirty){editBits=draft.bits;bitDirty=true;}
    if(draft?.csv!==undefined){$("dataset-csv").value=draft.csv;datasetRunId=selectedId;datasetDirty=true;}
  }
  if(pinnedStep!==null && run() && pinnedStep>=run().step){pinnedStep=null;pinnedRecord=null;pinnedRunId=null;lastChartSignature="";lastMicroscopeSignature="";}
  if(document.activeElement!==$("workspace-name")&&!workspaceNameDirty) $("workspace-name").value=next.workspace.name;
  $("run-count").textContent=String(next.runs.length);
  $("workspace-summary").textContent=`${next.runs.length} experiment${next.runs.length===1?"":"s"} · ${next.runs.reduce((sum,item)=>sum+(item.step||0),0).toLocaleString()} real steps`;
  if(next.autosave_error && next.autosave_error!==lastSaveError) {lastSaveError=next.autosave_error;toast(`Local autosave: ${next.autosave_error}`,"error");}
  renderSidebar();renderWorld();renderInspector();renderTransport();renderActiveView();
  firstRender=false;
}
function renderSidebar() {
  const signature=encodeJSON(studio.runs.map(entry=>[entry.id,entry.name,entry.step,entry.state,entry.parent_id]))+selectedId;
  if(signature===lastSideSignature) return;
  lastSideSignature=signature;
  const fragment=document.createDocumentFragment();
  studio.runs.forEach(entry=>{
    const button=element("button",`run-list-item ${entry.id===selectedId?"selected":""}`);
    button.title=`${entry.name} · ${titleCase(entry.state)} · Step ${entry.step}`;
    button.setAttribute("aria-pressed",String(entry.id===selectedId));
    const dot=element("span","run-color");dot.style.background=colorOf(entry.id);button.append(dot);
    const info=element("div");info.append(element("strong","",entry.name),element("small","",`${entry.parent_id?"⑂ ":""}${titleCase(entry.config?.optimizer)} · ${entry.step} steps`));button.append(info);
    if(isRunning(entry)) button.append(element("span","running-dot"));
    button.addEventListener("click",()=>selectRun(entry.id));fragment.append(button);
  });
  $("run-list").replaceChildren(fragment);
}
function setSelectionPending(value){selectionPending=value;document.body.classList.toggle("selection-pending",value);for(const node of $$(".inspector,.transport,#view-timeline,#view-microscope,#view-data,#view-artifacts,#run-notes,#save-run-notes")){node.toggleAttribute("inert",value);node.setAttribute("aria-busy",String(value));}const status=$("run-selection-status");if(status){status.hidden=!value;status.textContent=value?"Selecting experiment…":"";}}
async function selectRun(id) { if(id===selectedId&&!selectionPending)return;if(selectionPending){toast("Wait for the current experiment selection to finish.");return;}setSelectionPending(true);try{return await act("run.select",{run_id:id});}finally{setSelectionPending(false);renderTransport();} }
function makeNode(entry) {
  const node=element("article","world-node");node.dataset.id=entry.id;node.tabIndex=0;node.setAttribute("aria-label",`Experiment ${entry.name}`);
  const top=element("div","node-top");const dot=element("span","run-color");top.append(dot,element("span","node-label"),element("span","node-type"));
  const value=element("div","node-parameter");
  const chart=svgElement("svg",{class:"node-chart",viewBox:"0 0 267 40","aria-hidden":"true"});
  const bottom=element("div","node-bottom");bottom.append(element("span","node-status"),element("span","node-step"));
  const play=element("button","node-play","▶");play.title="Play or pause this experiment";play.setAttribute("aria-label","Play or pause experiment");play.addEventListener("pointerdown",e=>e.stopPropagation());play.addEventListener("click",e=>{e.stopPropagation();togglePlay(entry.id);});bottom.append(play);
  node.append(top,value,chart,bottom);
  node.addEventListener("pointerdown",event=>beginNodeDrag(event,node));
  node.addEventListener("keydown",event=>{if(event.key==="Enter"){event.preventDefault();selectRun(entry.id);}});
  return node;
}
function renderWorld() {
  const nodes=$("world-nodes");
  const ids=new Set(studio.runs.map(entry=>entry.id));
  [...nodes.children].forEach(node=>{if(!ids.has(node.dataset.id))node.remove();});
  studio.runs.forEach(entry=>{
    let node=[...nodes.children].find(child=>child.dataset.id===entry.id);
    if(!node){node=makeNode(entry);nodes.append(node);}
    node.classList.toggle("selected",entry.id===selectedId);node.dataset.state=String(entry.state).toLowerCase();node.style.setProperty("--node-color",colorOf(entry.id));
    const position=nodeDrag?.id===entry.id?nodeDrag.position:entry.position || {x:80,y:270};
    node.style.left=`${position.x}px`;node.style.top=`${position.y}px`;
    node.querySelector(".run-color").style.background=colorOf(entry.id);
    node.querySelector(".node-label").textContent=entry.name;
    node.querySelector(".node-type").textContent=entry.config?.optimizer || "parameter";
    const parameter=node.querySelector(".node-parameter");parameter.replaceChildren(document.createTextNode(fmt(entry.parameter,7)),element("small","","w"));
    node.querySelector(".node-status").textContent=titleCase(entry.state);
    node.querySelector(".node-step").textContent=`STEP ${entry.step} / ${entry.config?.steps || "—"}`;
    const play=node.querySelector(".node-play");play.textContent=isRunning(entry)?"Ⅱ":"▶";play.disabled=isFinished(entry);play.title=isFinished(entry)?"Run complete. Restart for fresh history, or configure a new experiment.":"Play or pause this experiment";
    const signature=`${entry.step}:${entry.parameter_uint32??floatHex(entry.parameter)}:${entry.latest?.record_hash || ""}`;
    const chart=node.querySelector("svg");
    if(chart.dataset.signature!==signature){drawSpark(chart,entry.series || [],267,40,colorOf(entry.id),true);chart.dataset.signature=signature;}
  });
  const maxX=Math.max(80,...studio.runs.map(entry=>(entry.position?.x || 0)+310));
  const last=studio.runs[studio.runs.length-1];
  $("canvas-new").style.left=`${maxX+15}px`;$("canvas-new").style.top=`${last?.position?.y || 275}px`;
  $("world-empty").hidden=studio.runs.length>0;
  renderConnections();
}
function renderConnections() {
  const svg=$("world-connections");const fragment=document.createDocumentFragment();
  studio.runs.forEach(entry=>{
    const parent=studio.runs.find(item=>item.id===entry.parent_id);if(!parent)return;
    const from=nodeDrag?.id===parent.id?nodeDrag.position:parent.position;
    const to=nodeDrag?.id===entry.id?nodeDrag.position:entry.position;
    if(!from||!to)return;
    const x1=from.x+267+2000,y1=from.y+98+2000,x2=to.x+2000,y2=to.y+98+2000;
    const reach=Math.max(45,Math.abs(x2-x1)*.5);
    fragment.append(svgElement("path",{d:`M${x1},${y1} C${x1+reach},${y1} ${x2-reach},${y2} ${x2},${y2}`,class:"connection-path"}));
    fragment.append(svgElement("circle",{cx:x1,cy:y1,r:3,fill:colorOf(parent.id),opacity:.7}));
    fragment.append(svgElement("circle",{cx:x2,cy:y2,r:3,fill:colorOf(entry.id),opacity:.7}));
  });svg.replaceChildren(fragment);
}
function applyCamera() {
  $("world-transform").style.transform=`translate(${camera.x}px,${camera.y}px) scale(${camera.zoom})`;
  $("world-grid").style.backgroundSize=`${21*camera.zoom}px ${21*camera.zoom}px`;
  $("world-grid").style.backgroundPosition=`${camera.x}px ${camera.y}px`;
  $("zoom-reset").textContent=`${Math.round(camera.zoom*100)}%`;
}
function saveCamera() {clearTimeout(cameraSaveTimer);cameraSaveTimer=setTimeout(()=>act("workspace.update",{view:{...camera}},{concurrent:true}),450);}
function zoomBy(factor, px, py) {
  const rect=$("world-viewport").getBoundingClientRect();const x=px??rect.width/2,y=py??rect.height/2;
  const next=Math.min(2.5,Math.max(.2,camera.zoom*factor));const ratio=next/camera.zoom;
  camera.x=x-(x-camera.x)*ratio;camera.y=y-(y-camera.y)*ratio;camera.zoom=next;applyCamera();saveCamera();
}
function fitWorld() {
  if(!studio.runs.length)return;
  const viewport=$("world-viewport").getBoundingClientRect();
  const minX=Math.min(...studio.runs.map(entry=>entry.position?.x||0));
  const minY=Math.min(...studio.runs.map(entry=>entry.position?.y||0));
  const maxX=Math.max(...studio.runs.map(entry=>(entry.position?.x||0)+267));
  const maxY=Math.max(...studio.runs.map(entry=>(entry.position?.y||0)+195));
  const topSafe=20;
  const availableH=Math.max(125,viewport.height-topSafe-60);
  const zoom=Math.min(1.15,Math.max(.2,Math.min((viewport.width-70)/(maxX-minX),availableH/(maxY-minY))));
  camera={zoom,x:(viewport.width-(maxX-minX)*zoom)/2-minX*zoom,y:topSafe+(availableH-(maxY-minY)*zoom)/2-minY*zoom};
  applyCamera();saveCamera();
}
function focusRun(id=selectedId){const entry=studio.runs.find(item=>item.id===id);if(!entry?.position)return;const rect=$("world-viewport").getBoundingClientRect();if(!rect.width||!rect.height)return;camera.x=rect.width/2-(entry.position.x+133.5)*camera.zoom;camera.y=Math.max(105,(rect.height-35)/2)-(entry.position.y+97.5)*camera.zoom;applyCamera();saveCamera();}
function beginNodeDrag(event,node) {
  if(event.button!==0||event.target.closest("button"))return;
  event.preventDefault();event.stopPropagation();
  const entry=studio.runs.find(item=>item.id===node.dataset.id);if(!entry)return;
  nodeDrag={id:entry.id,startX:event.clientX,startY:event.clientY,origin:{...entry.position},position:{...entry.position},node,moved:false};
  node.setPointerCapture(event.pointerId);
}
window.addEventListener("pointermove",event=>{
  if(nodeDrag){
    const dx=(event.clientX-nodeDrag.startX)/camera.zoom,dy=(event.clientY-nodeDrag.startY)/camera.zoom;
    nodeDrag.moved ||= Math.abs(dx)+Math.abs(dy)>4;
    nodeDrag.position={x:nodeDrag.origin.x+dx,y:nodeDrag.origin.y+dy};
    nodeDrag.node.style.left=`${nodeDrag.position.x}px`;nodeDrag.node.style.top=`${nodeDrag.position.y}px`;renderConnections();
  }else if(panDrag){camera.x=panDrag.x+event.clientX-panDrag.clientX;camera.y=panDrag.y+event.clientY-panDrag.clientY;applyCamera();}
});
window.addEventListener("pointerup",async()=>{
  if(nodeDrag){const drag=nodeDrag;nodeDrag=null;if(drag.moved)await act("run.update",{run_id:drag.id,position:drag.position});else await selectRun(drag.id);}
  if(panDrag){panDrag=null;saveCamera();}
});
$("world-viewport").addEventListener("pointerdown",event=>{
  if(event.button!==0||event.target.closest(".world-node,button"))return;
  panDrag={x:camera.x,y:camera.y,clientX:event.clientX,clientY:event.clientY};$("world-viewport").setPointerCapture(event.pointerId);
});
$("world-viewport").addEventListener("wheel",event=>{event.preventDefault();const rect=$("world-viewport").getBoundingClientRect();zoomBy(Math.exp(-event.deltaY*.001),event.clientX-rect.left,event.clientY-rect.top);},{passive:false});
function drawSpark(svg,series,width,height,color,showEmpty=false) {
  const points=series.filter(entry=>finite(entry.w));svg.replaceChildren();
  if(!points.length){svg.append(svgElement("path",{d:`M15 ${height-12}H${width-15}`,class:"empty-line",stroke:color,fill:"none","stroke-width":1}));if(showEmpty)svg.append(svgElement("text",{x:15,y:14,class:"node-empty"},"A trajectory waiting to happen"));return;}
  let lo=Math.min(...points.map(point=>point.w)),hi=Math.max(...points.map(point=>point.w));
  if(lo===hi){lo-=Math.max(.01,Math.abs(lo)*.01);hi+=Math.max(.01,Math.abs(hi)*.01);}
  const coords=points.map((point,index)=>[12+index/Math.max(1,points.length-1)*(width-24),height-7-(point.w-lo)/(hi-lo)*(height-14)]);
  const path=coords.map(([x,y],index)=>`${index?"L":"M"}${x.toFixed(2)},${y.toFixed(2)}`).join(" ");
  svg.append(svgElement("path",{d:`${path} L${width-12},${height} L12,${height} Z`,fill:color,opacity:.055}));
  svg.append(svgElement("path",{d:path,stroke:color,"stroke-width":1.5,fill:"none","stroke-linejoin":"round"}));
  const end=coords[coords.length-1];svg.append(svgElement("circle",{cx:end[0],cy:end[1],r:2,fill:color}));
}
function renderInspector() {
  const entry=run();const active=!!entry;
  $("inspector-run-name").textContent=entry?.name || "Select an experiment";
  $("parameter-value").textContent=fmt(entry?.parameter,8);
  const currentHex=floatHex(entry?.parameter);
  const intervened=!!entry?.live_intervention;
  $("parameter-hex").textContent=`${currentHex} · float32${intervened?" · manually changed":""}`;
  $("loss-value").textContent=fmt(entry?.loss,5);$("gradient-value").textContent=fmt(entry?.gradient,5);
  $("run-status").textContent=titleCase(entry?.state || "idle").toUpperCase();$("run-status").className=`status-badge ${isRunning(entry)?"running":""}`;
  if(document.activeElement!==$("run-speed"))$("run-speed").value=String(Math.min(120,entry?.speed||30));
  $("speed-label").textContent=`${fmt(entry?.speed||30,4)} steps/s`;
  drawSpark($("inspector-spark"),entry?.series || [],240,57,colorOf(selectedId));
  const config=entry?.config || {};
  $("selected-context").replaceChildren();
  if(entry){
    for(const [label,value] of [["ENGINE",`${titleCase(config.optimizer)} · ${config.model?.use_tanh?"tanh":"linear"}`],["DATA",`${config.dataset?.samples || "—"} samples · batch ${config.dataset?.batch_size || "—"}`],["PRECISION",config.run_fp64_reference_track?"float32 + independent float64":"float32 · reference track disabled"],["ORIGIN",entry.parent_id?`Branch of ${studio.runs.find(item=>item.id===entry.parent_id)?.name || "deleted experiment"}`:"Root experiment"]]){
      const line=element("div");line.append(element("span","",`${label}  `),element("strong","",value));$("selected-context").append(line);
    }
    if(intervened)$("selected-context").append(element("div","orange-text","Live state contains a manual intervention. Loss and gradient describe the last optimizer step."));
  }
  $("run-error").hidden=!entry?.error;$("run-error").textContent=entry?.error || "";
  for(const id of ["run-steps","fork-run","checkpoint-run","reset-run","delete-run","rename-run","apply-dataset","save-run-notes","artifact-export","verify-run","artifact-checkpoint","transport-checkpoint","restart-run"])$(id).disabled=!active;
  $("config-form").querySelector("button[type=submit]").disabled=!active;
  if(configRunId!==selectedId && capabilities){configRunId=selectedId;const draft=experimentDrafts.get(selectedId);configDraft=GraphModel.clone(draft?.config||entry?.config||capabilities.default_config);renderConfig();if(draft?.configJSONDirty){$("config-json").value=draft.configJSON;configJSONDirty=true;}}
}
function renderTransport() {
  const entry=run();
  $("transport-name").textContent=entry?.name || "No experiment";
  $("transport-state").textContent=entry?`${titleCase(entry.state)} · ${titleCase(entry.config?.optimizer)}`:"Ready when you are";
  $("transport-color").style.background=colorOf(selectedId);
  $("play-run").textContent=isRunning(entry)?"Ⅱ":"▶";
  $("play-run").setAttribute("aria-label",isRunning(entry)?"Pause selected experiment":"Play selected experiment");
  $("play-run").disabled=!entry || isFinished(entry) || actionBusy;
  $("play-run").title=isFinished(entry)?"Run complete. Restart for fresh history, or configure a new experiment.":"Play / pause · Space";
  $("step-run").disabled=!entry || isFinished(entry) || actionBusy;
  $("transport-reset").disabled=!entry;
  $("current-step").textContent=String(entry?.step || 0);$("total-steps").textContent=`/ ${entry?.config?.steps || 0}`;
  const percentage=entry?Math.min(100,entry.step/entry.config.steps*100):0;
  $("progress-fill").style.width=`${percentage}%`;
  $("transport-hint").textContent=entry?.error?"Stopped · inspect the recorded engine error":isFinished(entry)?"Trajectory complete. Restart, or configure a new experiment.":isRunning(entry)?"Live computation · every optimizer step recorded":"One real parameter. Every step observed.";
}
function switchView(next) {
  if(!VIEW_NAMES[next])return;view=next;location.hash=next;
  document.body.classList.toggle("graph-mode",["nodes","plugins","transformers"].includes(next));
  $$(".view").forEach(node=>node.classList.toggle("active",node.id===`view-${next}`));
  $$(".nav-item").forEach(node=>{node.classList.toggle("active",node.dataset.view===next);node.setAttribute("aria-current",node.dataset.view===next?"page":"false");});
  $("view-label").textContent=VIEW_NAMES[next];$("fit-world").style.display=next==="world"?"":"none";
  lastChartSignature="";lastMicroscopeSignature="";renderActiveView();
  window.dispatchEvent(new CustomEvent("studio:view",{detail:next}));
}
function renderActiveView() {
  if(view==="timeline")renderTimeline();
  if(view==="microscope")renderMicroscope();
  if(view==="data" && (!selectedId||datasetRunId!==selectedId))loadDataset();
  else if(view==="data"&&datasetDirty){drawDataset();drawLandscape();}
  if(view==="journal")renderJournal();
  if(view==="artifacts")renderArtifacts();
}
function currentRecord() {return pinnedStep!==null && pinnedRunId===selectedId?pinnedRecord:run()?.latest || null;}
function metricCell(label,value,detail) {const item=element("div","metric-item");item.append(element("label","",label),element("strong","",fmt(value,7)));if(detail)item.append(element("small","",detail));return item;}
function renderTimeline() {
  const entry=run();
  const signature=encodeJSON(studio.runs.map(item=>[item.id,item.step,item.latest?.record_hash,item.name]))+selectedId+metric+$("compare-runs").checked+pinnedStep;
  if(signature!==lastChartSignature){lastChartSignature=signature;drawTimeline();}
  const series=entry?.series||[];const max=series.length?series[series.length-1].step:0;const min=series.length?series[0].step:0;
  $("sampling-note").textContent=entry?.series_sampled?`Showing ${series.length.toLocaleString()} observed points from ${entry.series_total.toLocaleString()} records. Every exact record remains available with the time-travel slider and in exports.`:"Record indices start at 0. The transport counts completed optimizer updates. Every plotted point is a recorded observation.";
  $("scrub-step").min=String(min);$("scrub-step").max=String(max);$("scrub-step").disabled=!series.length;
  if(document.activeElement!==$("scrub-step"))$("scrub-step").value=String(pinnedStep??max);
  $("scrub-label").textContent=series.length?`Record ${pinnedStep??max} ${pinnedStep===null?"· following live":"· historical record"}`:"No recorded steps";
  $("follow-live").textContent=pinnedStep===null?"● Following live":"Follow live";
  const record=currentRecord();
  $("timeline-detail").replaceChildren(metricCell("Parameter",record?.parameter_after,"stored float32"),metricCell("Loss",record?.loss,"recorded objective"),metricCell("Gradient",record?.gradient,"autograd observation"),metricCell("Actual update",record?.actual_delta,"stored after − before"),metricCell("Learning rate",record?.learning_rate,"at this optimizer step"),metricCell("FP64 reference",record?.fp64_track_value,"independent track"));
  renderEvents($("timeline-events"),entry?.events || [],30);$("event-count").textContent=`${entry?.events?.length || 0} recent`;
}
function drawTimeline() {
  const svg=$("timeline-chart");const rect=svg.getBoundingClientRect();const width=Math.max(280,rect.width),height=Math.max(200,rect.height);svg.setAttribute("viewBox",`0 0 ${width} ${height}`);svg.replaceChildren();
  const entries=$("compare-runs").checked?studio.runs:run()?[run()]:[];
  const all=entries.flatMap(entry=>(entry.series||[]).filter(point=>finite(point[metric])));
  const pad={left:58,right:23,top:25,bottom:36};
  const plotW=width-pad.left-pad.right,plotH=height-pad.top-pad.bottom;
  let min=all.length?all.reduce((lowest,point)=>Math.min(lowest,point[metric]),Infinity):0,max=all.length?all.reduce((highest,point)=>Math.max(highest,point[metric]),-Infinity):1;
  if(min===max){const amount=Math.max(Math.abs(min)*.05,.01);min-=amount;max+=amount;}else{const amount=(max-min)*.07;min-=amount;max+=amount;}
  const minStep=all.length?all.reduce((lowest,point)=>Math.min(lowest,point.step),Infinity):0,maxStep=all.length?all.reduce((highest,point)=>Math.max(highest,point.step),-Infinity):10;
  const x=(step)=>pad.left+(step-minStep)/Math.max(1,maxStep-minStep)*plotW;
  const y=(value)=>pad.top+(max-value)/(max-min)*plotH;
  for(let i=0;i<=4;i++){const value=min+(max-min)*i/4,py=y(value);svg.append(svgElement("line",{x1:pad.left,y1:py,x2:width-pad.right,y2:py,stroke:"#37303e","stroke-width":.6,"stroke-dasharray":"3 5"}));svg.append(svgElement("text",{x:pad.left-10,y:py+3,"text-anchor":"end",fill:"#8f7e9d","font-size":8},fmt(value,4)));}
  for(let i=0;i<=4;i++){const step=Math.round(minStep+(maxStep-minStep)*i/4),px=pad.left+plotW*i/4;svg.append(svgElement("text",{x:px,y:height-15,"text-anchor":"middle",fill:"#8f7e9d","font-size":8},String(step)));}
  svg.append(svgElement("text",{x:width-pad.right,y:height-4,"text-anchor":"end",fill:"#6e5e7c","font-size":7},"OPTIMIZER STEP"));
  entries.forEach(entry=>{
    const points=(entry.series||[]).filter(point=>finite(point[metric]));if(!points.length)return;
    const path=points.map((point,index)=>`${index?"L":"M"}${x(point.step)},${y(point[metric])}`).join(" ");
    svg.append(svgElement("path",{d:path,fill:"none",stroke:colorOf(entry.id),"stroke-width":entry.id===selectedId?2:1.4,"stroke-linejoin":"round",opacity:entry.id===selectedId?1:.7}));
    if(points.length===1)svg.append(svgElement("circle",{cx:x(points[0].step),cy:y(points[0][metric]),r:3,fill:colorOf(entry.id)}));
    if(pinnedStep!==null && entry.id===selectedId){const point=points.find(item=>item.step===pinnedStep);if(point){svg.append(svgElement("line",{x1:x(point.step),y1:pad.top,x2:x(point.step),y2:height-pad.bottom,stroke:colorOf(entry.id),"stroke-dasharray":"3 4",opacity:.45}));svg.append(svgElement("circle",{cx:x(point.step),cy:y(point[metric]),r:4,fill:colorOf(entry.id),stroke:"#29222f","stroke-width":2}));}}
  });
  $("chart-empty").style.display=all.length?"none":"flex";
  $("chart-legend").replaceChildren(...entries.map(entry=>{const label=element("span");const line=element("i");line.style.setProperty("--line-color",colorOf(entry.id));label.append(line,document.createTextNode(entry.name));return label;}));
  if(all.length){const hit=svgElement("rect",{x:pad.left,y:pad.top,width:plotW,height:plotH,fill:"transparent",cursor:"crosshair"});hit.addEventListener("pointerdown",event=>{const bounds=svg.getBoundingClientRect();const requested=Math.round(minStep+((event.clientX-bounds.left-pad.left)/plotW)*(maxStep-minStep));pinRecord(Math.min(maxStep,Math.max(minStep,requested)));});svg.append(hit);}
}
async function pinRecord(step) {
  const entry=run();if(!entry)return;const serial=++recordRequest;
  if(bitDirty)captureExperimentDraft(selectedId);
  pinnedStep=step;pinnedRunId=entry.id;pinnedRecord=null;bitDirty=false;
  if(entry.latest?.step===step)pinnedRecord=entry.latest;
  else try {const response=await request(`/api/runs/${encodeURIComponent(entry.id)}/record?step=${step}`);if(serial!==recordRequest||entry.id!==selectedId)return;pinnedRecord=response.record||response;}
  catch(error){toast(error.message,"error");}
  lastChartSignature="";lastMicroscopeSignature="";renderActiveView();
}
function renderEvents(container,events,limit=50) {
  const signature=encodeJSON(events.slice(-limit));if(container.dataset.signature===signature)return;container.dataset.signature=signature;
  const rows=events.slice(-limit).reverse().map(event=>{const row=element("div","event-row");row.append(element("span","",`#${event.step}`));const info=element("div");info.append(element("b","",titleCase(event.code)),element("p","",event.detail));row.append(info);return row;});
  container.replaceChildren(...(rows.length?rows:[element("div","empty-message","The journal is quiet. Start, branch, intervene, or capture a checkpoint to leave a trail.")]));
}
function floatHex(value) {if(!finite(value))return "—";const buffer=new ArrayBuffer(4);const data=new DataView(buffer);data.setFloat32(0,value,false);return "0x"+data.getUint32(0,false).toString(16).padStart(8,"0");}
function clearAppliedBitDraft(){bitDirty=false;const draft=experimentDrafts.get(selectedId);if(draft){delete draft.bits;delete draft.bitDirty;}}
function hexNumber(word) {const buffer=new ArrayBuffer(4);const data=new DataView(buffer);data.setUint32(0,word>>>0,false);return data.getFloat32(0,false);}
function renderMicroscope() {
  const entry=run(),record=currentRecord();
  const signature=encodeJSON([entry?.id,entry?.step,record?.record_hash,record?.parameter_after_uint32,entry?.parameter,entry?.parameter_uint32,pinnedStep,entry?.live_optimizer_state,entry?.live_learning_rate,entry?.live_optimizer_initialized,entry?.live_intervention]);
  if(signature===lastMicroscopeSignature)return;lastMicroscopeSignature=signature;
  const value=pinnedStep!==null?record?.parameter_after:entry?.parameter;
  if(!bitDirty){const word=pinnedStep!==null?record?.parameter_after_uint32:entry?.parameter_uint32;editBits=Number.isInteger(word)&&word>=0&&word<=0xffffffff?word>>>0:finite(value)?parseInt(floatHex(value).slice(2),16):null;}
  renderBits();
  const liveMemory=entry?.live_optimizer_state||{},historical=pinnedStep!==null;
  $("optimizer-label").textContent=entry?`${titleCase(entry.config.optimizer)} · ${historical?`historical step ${pinnedStep}`:entry.live_optimizer_initialized?"actual live memory":"live memory not initialized"}`:"—";
  const memoryMetrics=historical?[metricCell("First moment",record?.moment1,"recorded memory m"),metricCell("Second moment",record?.moment2,"recorded memory v"),metricCell("Learning rate",record?.learning_rate,"at recorded step")]:[metricCell(entry?.config.optimizer==="sgd"?"Momentum buffer":"First moment",liveMemory.exp_avg??liveMemory.momentum_buffer,"actual live optimizer state"),metricCell("Second moment",liveMemory.exp_avg_sq,"actual live optimizer state"),metricCell("Next step LR",entry?.live_learning_rate??entry?.config.lr,"live optimizer rate prepared for the following step")];
  $("optimizer-metrics").replaceChildren(...memoryMetrics,metricCell("Adaptive update",record?.optimizer_adaptive_update,"last recorded contribution"),metricCell("Weight decay",record?.optimizer_weight_decay_update,"last recorded contribution"),metricCell("Rounding error",record?.fp32_rounding_error_abs,"last recorded storage error"));
  if(optimizerEditRunId!==selectedId){optimizerEditRunId=selectedId;const state=entry?.config.optimizer==="sgd"?{momentum_buffer:liveMemory.momentum_buffer??0}:{step:liveMemory.step??0,exp_avg:liveMemory.exp_avg??0,exp_avg_sq:liveMemory.exp_avg_sq??0};if(entry?.config.amsgrad)state.max_exp_avg_sq=liveMemory.max_exp_avg_sq??0;$("optimizer-state-json").value=experimentDrafts.get(selectedId)?.optimizer??encodeJSON(state,null,2);}
  const stages=[];
  if(record){
    const labels={a:"Before forward",b:"After forward",c:"After backward",d:"Before optimizer",e:"After optimizer",f:"After scheduler"};
    for(const key of ["a","b","c","d","e","f"]){const word=record[`snapshot_${key}_uint32`];if(word===null||word===undefined)continue;const card=element("div","stage");card.append(element("b","",`${key.toUpperCase()} · ${labels[key]}`),element("span","",fmt(hexNumber(word),8)),element("small","",`0x${Number(word).toString(16).padStart(8,"0")}`));stages.push(card);}
  }
  $("record-stages").replaceChildren(...(stages.length?stages:[element("div","empty-message","Step the optimizer to capture the parameter before and after each training stage.")]));
  $("full-record").textContent=record?encodeJSON(record,null,2):"No training steps recorded yet. The live parameter exists, but there is no optimizer-step record.";
}
function renderBits() {
  const editor=$("bit-editor");editor.replaceChildren();
  for(let index=31;index>=0;index--){const value=editBits===null?0:(editBits>>>index)&1;const button=element("button",`bit ${index===31?"sign":index>=23?"exponent":"fraction"} ${value?"on":""}`,scalarText(value));button.disabled=editBits===null;button.title=`${index===31?"Sign":index>=23?"Exponent":"Fraction"} bit ${index} · click to flip`;button.setAttribute("aria-label",`Bit ${index}, ${value}, click to flip`);button.addEventListener("click",()=>{editBits=(editBits^(1<<index))>>>0;bitDirty=true;markDraftsDirty();renderBits();});editor.append(button);}
  const value=editBits===null?null:hexNumber(editBits);
  $("bits-hex").textContent=editBits===null?"—":`0x${editBits.toString(16).padStart(8,"0")}`;
  $("bit-value").textContent=editBits===null?"Select an experiment":`${fmt(value,10)} ${bitDirty?"· edited, not yet applied":pinnedStep!==null?`· recorded step ${pinnedStep}`:"· current live parameter"}`;
  $("apply-bits").disabled=!bitDirty||!finite(value);
  $("apply-bits").title=!finite(value)?"The engine requires a finite float32 parameter.":pinnedStep!==null?"Applies these edited bits to the live experiment, not to the historical record.":"Apply these bits to the current parameter; pauses playback.";
}
function renderJournal() {
  if(!workspaceNotesLoaded){$("workspace-notes").value=studio.workspace.notes||"";workspaceNotesLoaded=true;}
  if(notesRunId!==selectedId){$("run-notes").value=experimentDrafts.get(selectedId)?.notes??run()?.notes??"";notesRunId=selectedId;}
  $("journal-run-name").textContent=run()?.name||"No experiment";renderEvents($("journal-events"),run()?.events||[],100);
}
function renderArtifacts() {
  const entry=run();const signature=encodeJSON(entry?.checkpoints || [])+selectedId;
  if(signature===lastCheckpointSignature)return;lastCheckpointSignature=signature;
  const rows=(entry?.checkpoints||[]).map(checkpoint=>{
    const row=element("div","checkpoint-row");row.append(element("span","checkpoint-icon","◇"));const info=element("div");const time=new Date(checkpoint.created_at);info.append(element("strong","",checkpoint.name),element("small","",`Step ${checkpoint.step} · ${Number.isNaN(time.getTime())?checkpoint.created_at:time.toLocaleString()}`));row.append(info);
    const controls=element("div","checkpoint-actions");const fork=element("button","button tiny","⑂ Restore as branch");fork.title="Create a new run from this exact checkpoint. Current history remains available.";fork.addEventListener("click",async()=>{const result=await act("checkpoint.restore",{checkpoint_id:checkpoint.id},{toast:"Checkpoint restored in a new branch."});if(result?.run_id){await selectRun(result.run_id);switchView("world");focusRun(result.run_id);}});const remove=element("button","icon-button","×");remove.title="Delete checkpoint";remove.setAttribute("aria-label",`Delete checkpoint ${checkpoint.name}`);remove.addEventListener("click",()=>askText({title:"Delete this checkpoint?",description:`“${checkpoint.name}” at step ${checkpoint.step} will be removed. The experiment's recorded history is retained.`,confirm:"Delete checkpoint",hideInput:true},()=>act("checkpoint.delete",{checkpoint_id:checkpoint.id},{toast:"Checkpoint deleted."})));controls.append(fork,remove);row.append(controls);return row;
  });$("checkpoint-list").replaceChildren(...(rows.length?rows:[element("div","empty-message","No checkpoints captured for this experiment. Capture a moment, then return to it as a new branch.")]));
}
async function loadDataset(force=false) {
  const entry=run();if(!entry){$("dataset-csv").value="";$("dataset-count").textContent="Create an experiment to inspect its data.";$("data-chart").replaceChildren();$("data-stats").replaceChildren();$("landscape-chart").replaceChildren();return;}
  if(datasetDirty&&!force)return;datasetRunId=entry.id;
  $("dataset-count").textContent="Loading actual engine data…";
  try {const data=await request(`/api/runs/${encodeURIComponent(entry.id)}/dataset`);if(entry.id!==selectedId)return;
    $("dataset-csv").value=data.csv || `x,target\n${(data.inputs||[]).map((x,index)=>`${scalarText(x)},${scalarText(data.targets[index])}`).join("\n")}\n`;
    datasetEvidence={runId:entry.id,step:entry.step,parameter:data.landscape?.parameters?.[Math.floor((data.landscape?.parameters?.length||0)/2)]??entry.parameter,...data};datasetDirty=false;const draft=experimentDrafts.get(entry.id);if(draft)delete draft.csv;markDraftsDirty();drawDataset();drawLandscape();
  }catch(error){datasetRunId=null;toast(error.message,"error");}
}
function parseCSV() {
  const lines=$("dataset-csv").value.trim().split(/\r?\n/).filter(line=>line.trim());
  if(lines.length&&/[a-z]/i.test(lines[0])&&!/^\s*[-+.\d]/.test(lines[0]))lines.shift();
  return lines.map(line=>line.split(",").map(cell=>Number(cell.trim()))).filter(values=>values.length===2&&values.every(Number.isFinite));
}
function drawDataset() {
  const points=parseCSV();$("dataset-count").textContent=`${points.length.toLocaleString()} numeric input/target rows${datasetDirty?" · edited, not applied":" · from current experiment"}`;
  const svg=$("data-chart");svg.replaceChildren();svg.setAttribute("viewBox","0 0 330 250");
  if(!points.length){svg.append(svgElement("text",{x:165,y:125,"text-anchor":"middle",fill:"#9a85ac","font-size":11},"Add numeric x,target rows to preview"));$("data-stats").replaceChildren();return;}
  const xs=points.map(p=>p[0]),ys=points.map(p=>p[1]);const predictions=!datasetDirty&&datasetEvidence?.runId===selectedId?(datasetEvidence.predictions||[]).filter(finite):[];let [minX,maxX]=extent(xs),[minY,maxY]=extent(ys.concat(predictions));if(minX===maxX){minX-=1;maxX+=1;}if(minY===maxY){minY-=1;maxY+=1;}
  const x=(value)=>40+(value-minX)/(maxX-minX)*270,y=(value)=>215-(value-minY)/(maxY-minY)*185;
  for(let i=0;i<=4;i++){const py=30+i*185/4;svg.append(svgElement("line",{x1:40,x2:310,y1:py,y2:py,stroke:"#45364f","stroke-width":.5,"stroke-dasharray":"3 4"}));svg.append(svgElement("text",{x:31,y:py+3,"text-anchor":"end",fill:"#8b739c","font-size":7},fmt(maxY-(maxY-minY)*i/4,3)));}
  for(let index=0;index<points.length;index+=Math.max(1,Math.floor(points.length/2500))){const point=points[index];svg.append(svgElement("circle",{cx:x(point[0]),cy:y(point[1]),r:2,fill:"#c4e78c",opacity:.6}));}
  if(!datasetDirty&&datasetEvidence?.runId===selectedId&&datasetEvidence.predictions){for(let index=0;index<points.length;index+=Math.max(1,Math.floor(points.length/1500))){const predicted=datasetEvidence.predictions[index];if(finite(predicted))svg.append(svgElement("circle",{cx:x(points[index][0]),cy:y(predicted),r:1.7,fill:"#b8a0ff",opacity:.6}));}}
  svg.append(svgElement("text",{x:175,y:241,"text-anchor":"middle",fill:"#8b739c","font-size":8},"INPUT x"));
  $("data-stats").replaceChildren(...[["INPUT RANGE",extent(xs).map(value=>fmt(value,4)).join(" → ")],["TARGET RANGE",extent(ys).map(value=>fmt(value,4)).join(" → ")]].map(([label,value])=>{const cell=element("div","",label);cell.append(element("strong","",value));return cell;}));
}
function drawLandscape(){
  const svg=$("landscape-chart");svg.replaceChildren();svg.setAttribute("viewBox","0 0 680 220");
  const landscape=datasetEvidence?.landscape;const points=(landscape?.parameters||[]).map((value,index)=>[value,landscape.losses[index]]).filter(point=>point.every(finite));
  if(!points.length){svg.append(svgElement("text",{x:340,y:110,fill:"#9b84ad","text-anchor":"middle","font-size":11},"Reload actual dataset to compute this diagnostic slice"));return;}
  const minX=Math.min(...points.map(point=>point[0])),maxX=Math.max(...points.map(point=>point[0]));let minY=Math.min(...points.map(point=>point[1])),maxY=Math.max(...points.map(point=>point[1]));if(minY===maxY){minY-=.01;maxY+=.01;}const x=value=>54+(value-minX)/Math.max(1e-20,maxX-minX)*600,y=value=>183-(value-minY)/(maxY-minY)*160;
  for(let index=0;index<=4;index++){const py=23+index*40;svg.append(svgElement("line",{x1:54,y1:py,x2:654,y2:py,stroke:"#45364f","stroke-width":.6,"stroke-dasharray":"3 5"}));svg.append(svgElement("text",{x:43,y:py+3,"text-anchor":"end",fill:"#8b739c","font-size":8},fmt(maxY-(maxY-minY)*index/4,4)));const px=54+index*150;svg.append(svgElement("text",{x:px,y:205,"text-anchor":"middle",fill:"#8b739c","font-size":8},fmt(minX+(maxX-minX)*index/4,4)));}
  svg.append(svgElement("path",{d:points.map((point,index)=>`${index?"L":"M"}${x(point[0])},${y(point[1])}`).join(" "),stroke:"#f1ac79","stroke-width":2,fill:"none","stroke-linejoin":"round"}));
  if(finite(datasetEvidence.parameter)){const px=x(datasetEvidence.parameter);svg.append(svgElement("line",{x1:px,x2:px,y1:23,y2:183,stroke:"#b8a0ff","stroke-dasharray":"4 4",opacity:.7}));svg.append(svgElement("text",{x:Math.min(604,Math.max(90,px+7)),y:33,fill:"#cbb3df","font-size":8},`slice center w = ${fmt(datasetEvidence.parameter,5)}`));}
  $("landscape-description").textContent=`${landscape.objective || "Full dataset objective"}. Computed from the actual dataset when loaded. Use Reload to compute a new slice around the live parameter. This full-data diagnostic differs from an individual minibatch training loss.`;
}
const CONFIG_GROUPS = [
  ["Parameter & training",["name","seed","initial_value","steps","accumulation_micro_batches","zero_grad_set_to_none","stop_on_nonfinite","live_speed_steps_per_second"]],
  ["Optimizer",["optimizer","lr","betas","eps","weight_decay","momentum","dampening","nesterov","amsgrad","maximize"]],
  ["Learning-rate schedule",["lr_schedule","lr_final_factor"]],
  ["Model",["model"]],["Dataset",["dataset"]],["Landscape probes",["landscape"]],["Validation",["validation"]],["Stability criteria",["stability"]],["Instrumentation",["run_fp64_reference_track","record_full_state_snapshots"]]
];
const SELECT_OPTIONS={optimizer:["sgd","adam","adamw"],lr_schedule:["constant","cosine","exponential"],walk:["linear","alternating"],dtype:["float64","float32"]};
function humanKey(key){return key.replaceAll("_"," ").replace(/\b(lr|fp64|fd|ulp)\b/g,part=>part.toUpperCase());}
function parseConfigJSON(text){
  if(!window.GraphModel)throw new Error("The configuration parser is still loading. Try again in a moment.");
  return GraphModel.parseConfigValue(text);
}
function renderConfig() {
  if(!configDraft)return;for(const owner of [configDraft,configDraft.dataset])if(owner&&owner.seed!==undefined)owner.seed=String(owner.seed);configJSONDirty=false;const fragment=document.createDocumentFragment();
  const addField=(fieldset,path,value)=>{
    const key=path.split(".").at(-1);const wrapper=element("div",`config-field ${key==="name"?"full":""}`);const label=element("label","",humanKey(key));const id=`config-${path.replaceAll(".","-")}`;label.htmlFor=id;let input;
    if(SELECT_OPTIONS[key]){input=element("select");SELECT_OPTIONS[key].forEach(option=>input.append(new Option(titleCase(option),option)));input.value=scalarText(value);}
    else{input=element("input");if(key==="seed"){input.type="text";input.inputMode="numeric";input.value=scalarText(value);}
      else if(typeof value==="boolean"){input.type="checkbox";input.checked=value;}
      else if(typeof value==="number"){input.type="number";input.step=Number.isInteger(value)&&!["initial_value","lr","eps","momentum","weight_decay","dampening","input_min","input_max","output_scale","input_scale"].includes(key)?"1":"any";input.value=scalarText(value);}
      else{input.type="text";input.value=Array.isArray(value)?encodeJSON(value):scalarText(value);}}
    input.id=id;input.dataset.path=path;input.dataset.kind=key==="seed"?"seed":Array.isArray(value)?"array":typeof value;input.title=key==="seed"?"Exact decimal integer seed; all 64 bits are preserved":path;
    if(key==="record_full_state_snapshots"){input.disabled=true;input.title="Full A–E state snapshots are required for numerical verification.";}
    const capture=()=>{try{if(typeof value==="number"&&(!input.value.trim()||!Number.isFinite(Number(input.value))))return;setConfigPath(path,input.type==="checkbox"?input.checked:typeof value==="number"?Number(input.value):Array.isArray(value)?JSON.parse(input.value):input.value);$("config-json").value=encodeJSON(configDraft,null,2);configJSONDirty=false;}catch(error){if(document.activeElement!==input)toast(`Invalid ${humanKey(key)}: ${error.message}`,"error");}};
    input.addEventListener("input",capture);input.addEventListener("change",capture);
    wrapper.append(label,input);fieldset.append(wrapper);
  };
  CONFIG_GROUPS.forEach(([name,keys])=>{const fieldset=element("fieldset","config-group");fieldset.append(element("legend","",name));keys.forEach(key=>{const value=configDraft[key];if(value===undefined)return;if(value&&typeof value==="object"&&!Array.isArray(value))Object.entries(value).forEach(([nested,subvalue])=>addField(fieldset,`${key}.${nested}`,subvalue));else addField(fieldset,key,value);});fragment.append(fieldset);});
  $("config-fields").replaceChildren(fragment);$("config-json").value=encodeJSON(configDraft,null,2);
}
function setConfigPath(path,value){const parts=path.split(".");let obj=configDraft;parts.slice(0,-1).forEach(part=>obj=obj[part]);obj[parts.at(-1)]=value;}
function setInspector(name){$$("[data-inspector]").forEach(button=>button.classList.toggle("active",button.dataset.inspector===name));$("inspector-controls").hidden=name!=="controls";$("inspector-configuration").hidden=name!=="configuration";}
async function togglePlay(id=selectedId){const entry=studio.runs.find(item=>item.id===id);if(!entry)return;if(isFinished(entry)){toast("This trajectory has used its step budget. Restart with a clean history, or configure a new experiment.");return;}await act(isRunning(entry)?"run.pause":"run.play",{run_id:id});}
async function createCheckpoint(){const entry=run();if(!entry)return;askText({title:"Keep this moment.",description:"Capture the parameter, optimizer, fixed data, reference track, and scheduler state. Restore it later in a new branch.",value:`Step ${entry.step}`,confirm:"Capture checkpoint"},name=>act("checkpoint.create",{name},{toast:"Checkpoint captured. Find it in Artifacts."}));}
async function forkRun(){const entry=run();if(!entry)return;askText({title:"Follow a new direction.",description:"Continue from this exact state in a new experiment. Your original trajectory stays here.",value:`${entry.name} · branch`,confirm:"Create branch"},async name=>{const result=await act("run.fork",{name},{toast:"A new direction is ready."});if(result?.run_id){await selectRun(result.run_id);switchView("world");focusRun(result.run_id);}});}
function resetRun(){const entry=run();if(!entry)return;askText({title:"Reset this experiment?",description:`“${entry.name}” will restore its initial parameter, clear optimizer memory, and restart regime analysis. Its immutable training history and existing step budget are retained. Capture a checkpoint first to keep this exact live state.`,confirm:"Reset experiment",hideInput:true},()=>act("run.reset",{},{toast:"Parameter reset. History retained; the existing step budget is unchanged."}));}
function restartRun(){const entry=run();if(!entry)return;askText({title:"Restart with a clean history?",description:`“${entry.name}” will return to its initial parameter and clear its recorded training history. Its current configuration and custom data are retained. Download evidence or capture a checkpoint first to preserve this trajectory.`,confirm:"Restart experiment",hideInput:true},()=>act("run.restart",{},{toast:"Experiment restarted with a clean training history."}));}
function deleteRun(){const entry=run();if(!entry)return;askText({title:"Delete this experiment?",description:`“${entry.name}”, its records, and its checkpoints will be removed from this workspace. Other branches remain available.`,confirm:"Delete experiment",hideInput:true},()=>act("run.delete",{},{toast:"Experiment deleted."}));}
function askText(options,callback){textCallback=callback;$("text-title").textContent=options.title;$("text-description").textContent=options.description||"";$("text-input").value=options.value||"";$("text-input-label").hidden=!!options.hideInput;$("text-input").required=!options.hideInput;$("text-submit").textContent=options.confirm||"Save";$("text-dialog").showModal();if(!options.hideInput){$("text-input").focus();$("text-input").select();}else $("text-submit").focus();}
function showNew(){if(!capabilities){toast("The engine is still connecting. Try again in a moment.");return;}$("new-name").value="";$("new-dialog").showModal();$("new-name").focus();}
function renderPresets(){const options=$("preset-options");options.replaceChildren();Object.entries(capabilities.presets||{}).forEach(([name,preset],index)=>{const label=element("label","preset-option");const radio=element("input");radio.type="radio";radio.name="preset";radio.value=name;radio.checked=name==="adamw"||(index===0&&!capabilities.presets.adamw);label.append(radio,element("b","",name.replaceAll("_"," ")),element("small","",preset.description||"Real optimizer experiment"));options.append(label);});}
async function saveProject(){
  if(workspaceReplaced){handleWorkspaceReplacement();return;}
  try{if(window.NodeLab)await window.NodeLab.flush({allowInvalid:true});}catch(error){toast(`Project not downloaded: ${error.message}`,"error");return;}
  try{markDraftsDirty();await flushDrafts();}catch(error){toast(`Project not downloaded: ${error.message}`,"error");return;}
  clearTimeout(cameraSaveTimer);clearTimeout(workspaceNameTimer);
  await saveWorkspaceName();
  if(cameraLoaded)await act("workspace.update",{view:{...camera}});
  await enqueueMutation(async()=>{
    actionBusy=true;stateGeneration++;
    try{const response=await request("/api/project",{binary:true});downloadBlob(await response.blob(),`${slug(studio.workspace.name)}.parameter-studio.json`);toast("Complete workspace downloaded.","success");}
    catch(error){toast(error.message,"error");}
    finally{actionBusy=false;renderTransport();}
  });
}
async function importProject(text){
  clearTimeout(cameraSaveTimer);clearTimeout(workspaceNameTimer);clearTimeout(draftSaveTimer);clearTimeout(dataPreviewTimer);
  return enqueueMutation(async()=>{
    actionBusy=true;stateGeneration++;importingProject=true;
    try{
      if(window.NodeLab)await window.NodeLab.quiesce();
      const result=await request("/api/project",{method:"POST",headers:{"Content-Type":"application/json"},body:text,acceptWorkspaceEpoch:true});
      experimentDrafts.clear();cameraLoaded=false;configRunId=null;datasetRunId=null;workspaceNotesLoaded=false;workspaceNameDirty=false;lastCheckpointSignature="";lastSelectedId=null;
      await loadDrafts(true);
      if(result.state)acceptState(result.state);else acceptState(await request("/api/state"));
      if(window.NodeLab)await window.NodeLab.reload(true);
      switchView("world");toast("Project opened. All imported experiments are paused.","success");
    }catch(error){toast(error.message,"error");}
    finally{importingProject=false;window.NodeLab?.resume();if(draftsDirty&&!draftsConflict)markDraftsDirty();actionBusy=false;renderTransport();}
  });
}
function downloadBlob(blob,filename){const url=URL.createObjectURL(blob);const link=element("a");link.href=url;link.download=filename;document.body.append(link);link.click();link.remove();setTimeout(()=>URL.revokeObjectURL(url),10000);}
async function exportRun(){if(selectionPending){toast("Wait for the selected experiment to load before downloading its evidence.");return;}const entry=run();if(!entry)return;try{const response=await request(`/api/runs/${encodeURIComponent(entry.id)}/export`,{binary:true});downloadBlob(await response.blob(),`${slug(entry.name)}-evidence.zip`);toast("Recorded experiment evidence exported.","success");}catch(error){toast(error.message,"error");}}
function downloadChart(){if(!(run()?.series||[]).length){toast("Record some optimizer steps before exporting a trajectory.");return;}const svg=$("timeline-chart").cloneNode(true);svg.setAttribute("xmlns",NS);svg.querySelectorAll("rect[fill=transparent]").forEach(node=>node.remove());const bg=svgElement("rect",{width:"100%",height:"100%",fill:"#1b1b20"});svg.prepend(bg);downloadBlob(new Blob([new XMLSerializer().serializeToString(svg)],{type:"image/svg+xml"}),`${slug(run().name)}-${metric}.svg`);}
const COMMANDS=[
  ...Object.entries(VIEW_NAMES).map(([key,name],index)=>({label:`Go to ${name}`,key:String(index+1),run:()=>switchView(key)})),
  {label:"New experiment",key:"+",run:showNew},{label:"Play / pause experiment",key:"Space",run:()=>togglePlay()},
  {label:"Advance one optimizer step",key:"S",run:()=>act("run.step",{count:1})},{label:"Branch experiment",key:"F",run:forkRun},
  {label:"Reset parameter, retain history",key:"↺",run:resetRun},{label:"Restart with a clean training history",key:"↻",run:restartRun},
  {label:"Capture checkpoint",key:"◇",run:createCheckpoint},{label:"Configure experiment",key:"↗",run:()=>{setInspector("configuration");if(innerWidth<=600)document.body.classList.add("inspector-open");}},
  {label:"Fit world to experiments",key:"⌖",run:()=>{switchView("world");fitWorld();}},{label:"Download workspace project",key:"Ctrl S",run:saveProject},{label:"Open saved project",key:"↑",run:()=>$("project-file").click()},{label:"Export experiment evidence ZIP",key:"↓",run:exportRun},{label:"Studio guide and shortcuts",key:"?",run:()=>$("help-dialog").showModal()}
];
function renderCommands(){const query=$("command-search").value.toLowerCase();$("command-results").replaceChildren(...COMMANDS.filter(item=>item.label.toLowerCase().includes(query)).map(item=>{const button=element("button","command-result");button.append(document.createTextNode(item.label),element("small","",item.key));button.addEventListener("click",()=>{$("command-dialog").close();item.run();});return button;}));}
function showCommands(){$("command-search").value="";renderCommands();$("command-dialog").showModal();$("command-search").focus();}

// Extended instruments are deliberately ordinary controls with explicit live-state semantics.
$("loss-value").previousElementSibling.textContent="LAST RECORDED LOSS";
$("gradient-value").previousElementSibling.textContent="LAST RECORDED GRADIENT";
const draftNotice=element("p","muted small","Unapplied configuration, CSV, bit, optimizer, and manifest edits are saved as drafts. Apply controls explicitly to change numerical state; project downloads retain both saved state and editor drafts.");
$("view-artifacts").querySelector(".page-heading").append(draftNotice);
const helpItems=$("help-dialog").querySelector(".help-items");
for(const [name,description]of [["Node lab","Connect typed nodes, edit literal inputs and JavaScript, queue immutable workflow snapshots, and inspect actual outputs. Click or drag ports, or use the connection form. Templates provide complete examples."],["Plugins","Build custom nodes, install or edit portable manifests, enable or disable add-ons, and turn declarative panel values into editable Object nodes. Scripts run in an isolated JavaScript engine with time and memory limits."]]){const item=element("p");item.append(element("b","",name),document.createTextNode(description));helpItems.append(item);}
for(const item of $$(".shortcut-list span"))if(item.textContent.includes("Change view"))item.querySelector("kbd").textContent="1–9";
const metricSelect=element("select","metric-select");metricSelect.id="more-metrics";metricSelect.setAttribute("aria-label","Additional recorded metric");metricSelect.append(new Option("More observations…",""));for(const [key,label] of [["moment1","First moment m"],["moment2","Second moment v"],["lr","Learning rate"],["fp64","FP64 reference"]])metricSelect.append(new Option(label,key));$("metric-tabs").after(metricSelect);metricSelect.addEventListener("change",()=>{if(!metricSelect.value)return;metric=metricSelect.value;$$("[data-metric]").forEach(button=>button.classList.remove("active"));lastChartSignature="";renderTimeline();});
$$("[data-metric]").forEach(button=>button.addEventListener("click",()=>{metricSelect.value="";}));
const samplingNote=element("div","sampling-note");samplingNote.id="sampling-note";$("chart-legend").after(samplingNote);
const optimizerEditor=element("details","optimizer-edit");optimizerEditor.append(element("summary","","Intervene in optimizer memory"),element("p","","Change live optimizer state with finite numeric values. Playback pauses and a manual intervention is recorded. Before Adam's first step, provide all three fields: step, exp_avg, exp_avg_sq (plus max_exp_avg_sq for AMSGrad). SGD accepts momentum_buffer. Historical records stay immutable."));const stateLabel=element("label","","Optimizer state JSON");stateLabel.htmlFor="optimizer-state-json";const stateTextarea=element("textarea","code-editor");stateTextarea.id="optimizer-state-json";stateTextarea.spellcheck=false;stateTextarea.setAttribute("aria-label","Optimizer state JSON");const stateButton=element("button","button","Apply live optimizer intervention ↗");stateButton.type="button";stateButton.addEventListener("click",async()=>{try{const state=JSON.parse(stateTextarea.value);if(!state||typeof state!=="object"||Array.isArray(state)||!Object.keys(state).length)throw new Error("Provide a non-empty optimizer state object.");if(Object.values(state).some(value=>!finite(value)))throw new Error("Every optimizer state value must be a finite number.");const result=await act("run.intervene",{optimizer_state:state},{button:stateButton,toast:"Live optimizer memory changed. Playback paused."});if(result){pinnedStep=null;lastMicroscopeSignature="";renderMicroscope();}}catch(error){toast(error.message,"error");}});optimizerEditor.append(stateLabel,stateTextarea,stateButton);$("optimizer-metrics").after(optimizerEditor);
const dataLegend=element("div","data-preview-legend");for(const label of ["Target samples","Model predictions when loaded"]){const item=element("span");item.append(element("i"),document.createTextNode(label));dataLegend.append(item);}$("data-chart").after(dataLegend);
const landscapePanel=element("div","data-landscape");const landscapeHeading=element("div","section-heading","LOSS LANDSCAPE");landscapeHeading.append(element("span","","Real full-data diagnostic slice"));const landscapeChart=svgElement("svg",{id:"landscape-chart",role:"img","aria-label":"Full dataset loss against parameter value"});const landscapeDescription=element("p");landscapeDescription.id="landscape-description";landscapePanel.append(landscapeHeading,landscapeChart,landscapeDescription);$("view-data").append(landscapePanel);

const restartButton=element("button","button subtle tiny","↻ Restart");restartButton.id="restart-run";restartButton.title="Restart with a clean training history";restartButton.addEventListener("click",restartRun);$("reset-run").textContent="↺ Reset";$("reset-run").after(restartButton);

// Events are registered once; polling never replaces a focused editing surface.
const selectionStatus=element("span","run-selection-status");selectionStatus.id="run-selection-status";selectionStatus.setAttribute("role","status");selectionStatus.hidden=true;$("view-label").after(selectionStatus);
for(const type of ["click","submit"])document.addEventListener(type,event=>{if(selectionPending&&event.target.closest(".inspector,.transport,#view-timeline,#view-microscope,#view-data,#view-artifacts,#save-run-notes")){event.preventDefault();event.stopImmediatePropagation();toast("Wait for the selected experiment to load before using experiment controls.");}},true);
const navPaths={world:["M4 4h4v4H4z M12 4h4v4h-4z M4 12h4v4H4z M12 12h4v4h-4z"],timeline:["M2 15l4-5 4 2 3-7 5 2","M2 18h16"],microscope:["M12 3a6 6 0 1 0 0 12 6 6 0 0 0 0-12","M4 17l3-4 M8 16h9 M12 6v6 M9 9h6"],data:["M3 3h14v14H3z M3 8h14 M3 12h14 M8 3v14 M12 3v14"],journal:["M4 3h12v14H4z M7 7h6 M7 10h6 M7 13h4"],artifacts:["M3 5l7-3 7 3-7 3z M3 5v9l7 4 7-4V5 M10 8v10"],nodes:["M2 2h5v5H2z M13 13h5v5h-5z M13 2h5v5h-5z M7 4h6 M4 7v8h9"],plugins:["M7 3h6v4h4v6h-4v4H7v-4H3V7h4z"]};
navPaths.transformers=["M2 3h5v5H2z M13 3h5v5h-5z M7 12h6v5H7z M7 5h6 M4 8v6h3 M16 8v6h-3"];
$$("[data-view]").forEach(button=>{const icon=svgElement("svg",{viewBox:"0 0 20 20",width:18,height:18,fill:"none",stroke:"currentColor","stroke-width":1.2,"stroke-linecap":"round","stroke-linejoin":"round","aria-hidden":"true"});(navPaths[button.dataset.view]||[]).forEach(path=>icon.append(svgElement("path",{d:path})));button.querySelector(".nav-icon").replaceChildren(icon);});
$$("[data-view]").forEach(button=>button.addEventListener("click",()=>switchView(button.dataset.view)));
$$("[data-inspector]").forEach(button=>button.addEventListener("click",()=>setInspector(button.dataset.inspector)));
$$(".dialog-close").forEach(button=>button.addEventListener("click",()=>button.closest("dialog").close()));
$$("dialog").forEach(dialog=>dialog.addEventListener("click",event=>{if(event.target===dialog){const bounds=dialog.getBoundingClientRect();if(event.clientX<bounds.left||event.clientX>bounds.right||event.clientY<bounds.top||event.clientY>bounds.bottom)dialog.close();}}));
["new-experiment","sidebar-new","add-experiment","canvas-new"].forEach(id=>$(id).addEventListener("click",showNew));
$("new-form").addEventListener("submit",async event=>{event.preventDefault();const preset=$("preset-options").querySelector("input:checked")?.value||"adamw";const name=$("new-name").value.trim()||`${titleCase(preset)} exploration`;const next=studio.runs.length;const position={x:80+(next%3)*330,y:275+Math.floor(next/3)*235};const result=await act("run.create",{preset,name,position},{button:$("new-form").querySelector("button[type=submit]"),toast:"New experiment created. Press play to explore."});if(result){$("new-dialog").close();if(result.run_id)await selectRun(result.run_id);switchView("world");focusRun(result.run_id);}});
$("text-form").addEventListener("submit",event=>{event.preventDefault();const callback=textCallback;const value=$("text-input").value.trim();$("text-dialog").close();textCallback=null;if(callback)callback(value);});
$("play-run").addEventListener("click",()=>togglePlay());$("step-run").addEventListener("click",()=>act("run.step",{count:1}));
$("run-steps").addEventListener("click",()=>{const count=Number($("step-count").value);if(!Number.isInteger(count)||count<1||count>10000){toast("Choose an integer between 1 and 10,000 steps.","error");return;}act("run.step",{count},{button:$("run-steps")});});
$("run-speed").addEventListener("input",()=>$("speed-label").textContent=`${$("run-speed").value} steps/s`);$("run-speed").addEventListener("change",()=>act("run.speed",{value:Number($("run-speed").value)}));
$("fork-run").addEventListener("click",forkRun);["checkpoint-run","transport-checkpoint","artifact-checkpoint"].forEach(id=>$(id).addEventListener("click",createCheckpoint));
["reset-run","transport-reset"].forEach(id=>$(id).addEventListener("click",resetRun));$("delete-run").addEventListener("click",deleteRun);
$("rename-run").addEventListener("click",()=>{if(run())askText({title:"Name this direction.",value:run().name,confirm:"Save name"},name=>act("run.update",{name},{toast:"Experiment renamed."}));});
async function saveWorkspaceName(){clearTimeout(workspaceNameTimer);const name=$("workspace-name").value.trim();if(!name||!workspaceNameDirty)return;const result=await act("workspace.update",{name});if(result&&$("workspace-name").value.trim()===name)workspaceNameDirty=false;}
$("workspace-name").addEventListener("input",()=>{workspaceNameDirty=true;clearTimeout(workspaceNameTimer);workspaceNameTimer=setTimeout(saveWorkspaceName,700);});$("workspace-name").addEventListener("change",saveWorkspaceName);$("workspace-name").addEventListener("blur",saveWorkspaceName);
$("zoom-in").addEventListener("click",()=>zoomBy(1.2));$("zoom-out").addEventListener("click",()=>zoomBy(1/1.2));$("zoom-reset").addEventListener("click",()=>{const old=camera.zoom;zoomBy(1/old);});$("fit-world").addEventListener("click",fitWorld);
$$("[data-metric]").forEach(button=>button.addEventListener("click",()=>{metric=button.dataset.metric;$$("[data-metric]").forEach(node=>node.classList.toggle("active",node===button));lastChartSignature="";renderTimeline();}));
$("compare-runs").addEventListener("change",()=>{lastChartSignature="";renderTimeline();});$("scrub-step").addEventListener("input",()=>pinRecord(Number($("scrub-step").value)));$("follow-live").addEventListener("click",()=>{if(bitDirty)captureExperimentDraft(selectedId);pinnedStep=null;pinnedRecord=null;pinnedRunId=null;const draft=experimentDrafts.get(selectedId);bitDirty=!!draft?.bitDirty;editBits=bitDirty?draft.bits:null;lastChartSignature="";lastMicroscopeSignature="";renderActiveView();});
$("download-chart").addEventListener("click",downloadChart);
$("apply-bits").addEventListener("click",async()=>{if(editBits===null)return;const result=await act("run.intervene",{parameter_hex:`0x${editBits.toString(16).padStart(8,"0")}`},{toast:"Float32 intervention applied to live state. Playback paused."});if(result){clearAppliedBitDraft();pinnedStep=null;pinnedRecord=null;markDraftsDirty();lastMicroscopeSignature="";renderMicroscope();}});
$("intervention-form").addEventListener("submit",async event=>{event.preventDefault();const value=$("intervene-value").value.trim(),lr=$("intervene-lr").value.trim();const data={};if(value){if(/^0x[0-9a-f]{8}$/i.test(value))data.parameter_hex=value;else if(Number.isFinite(Number(value))){if(Object.is(Number(value),-0))data.parameter_hex="0x80000000";else data.parameter_value=Number(value);}else{toast("Enter a finite decimal parameter or 0x followed by eight hexadecimal digits.","error");return;}}if(lr){if(!Number.isFinite(Number(lr))){toast("Learning rate must be finite.","error");return;}data.learning_rate=Number(lr);}if(!Object.keys(data).length){toast("Enter a parameter value or learning rate to change.");return;}const result=await act("run.intervene",data,{toast:"Manual intervention applied. Playback paused."});if(result){$("intervene-value").value="";$("intervene-lr").value="";pinnedStep=null;clearAppliedBitDraft();markDraftsDirty();lastMicroscopeSignature="";renderMicroscope();}});
$("reload-data").addEventListener("click",()=>{if(datasetDirty)askText({title:"Reload original samples?",description:"Your unapplied CSV edits will be replaced by the selected experiment's actual dataset.",confirm:"Reload dataset",hideInput:true},()=>loadDataset(true));else loadDataset(true);});
$("dataset-csv").addEventListener("input",()=>{datasetDirty=true;clearTimeout(dataPreviewTimer);dataPreviewTimer=setTimeout(drawDataset,220);});
$("apply-dataset").addEventListener("click",async()=>{const result=await act("run.dataset",{csv:$("dataset-csv").value,name:"Studio CSV dataset"},{button:$("apply-dataset"),toast:"Custom data applied to a new experiment branch."});if(result){datasetDirty=false;datasetRunId=null;if(result.run_id)await selectRun(result.run_id);await loadDataset(true);}});
$("download-dataset").addEventListener("click",()=>downloadBlob(new Blob([$("dataset-csv").value],{type:"text/csv"}),`${slug(run()?.name)}-dataset.csv`));
$("import-dataset").addEventListener("change",async event=>{const file=event.target.files[0];if(file){$("dataset-csv").value=await file.text();datasetDirty=true;markDraftsDirty();drawDataset();}event.target.value="";});
$("save-workspace-notes").addEventListener("click",()=>act("workspace.update",{notes:$("workspace-notes").value},{toast:"Workspace notes saved."}));$("save-run-notes").addEventListener("click",()=>act("run.update",{notes:$("run-notes").value},{toast:"Experiment notes saved."}));
$("config-json").addEventListener("input",()=>{configJSONDirty=true;});
$("config-form").addEventListener("submit",async event=>{event.preventDefault();try{if(!configJSONDirty){$$("[data-path]",$("config-fields")).forEach(input=>{if(input.disabled)return;let value=input.type==="checkbox"?input.checked:input.dataset.kind==="number"?Number(input.value):input.dataset.kind==="array"?JSON.parse(input.value):input.value;if(input.dataset.kind==="number"&&(!input.value.trim()||!Number.isFinite(value)))throw new Error(`${input.dataset.path} must contain a finite number`);setConfigPath(input.dataset.path,value);});$("config-json").value=encodeJSON(configDraft,null,2);}const config=parseConfigJSON($("config-json").value);const result=await act("run.configure",{config},{button:$("config-form").querySelector("button[type=submit]")});if(result?.run_id){toast(result.dataset_mode==="retained_custom"?"Configuration branched. Your actual custom dataset was retained.":"Configuration branched with data generated from the selected dataset settings.","success");await selectRun(result.run_id);focusRun(result.run_id);}}catch(error){toast(`Configuration: ${error.message}`,"error");}});
$("use-config-json").addEventListener("click",()=>{try{configDraft=parseConfigJSON($("config-json").value);if(!configDraft||typeof configDraft!=="object"||Array.isArray(configDraft))throw new Error("configuration must be an object");renderConfig();toast("JSON loaded into configuration controls.");}catch(error){toast(error.message,"error");}});
["save-project","artifact-save"].forEach(id=>$(id).addEventListener("click",saveProject));$("open-project").addEventListener("click",()=>$("project-file").click());
$("project-file").addEventListener("change",async event=>{const file=event.target.files[0];event.target.value="";if(!file)return;const text=await file.text();askText({title:"Open this workspace?",description:"The current workspace will be replaced by the imported project. Save a project copy first if you want to keep your current work.",confirm:"Open project",hideInput:true},()=>importProject(text));});
$("artifact-export").addEventListener("click",exportRun);$("verify-run").addEventListener("click",async()=>{const result=await act("run.verify",{},{button:$("verify-run")});if(result){const integrity=result.integrity||run()?.integrity;$("integrity-result").textContent=`${titleCase(integrity?.status)} · ${integrity?.hash||"no records"}`;toast(integrity?.status==="verified"?"Every recorded hash link verified.":"Record verification found a problem.",integrity?.status==="verified"?"success":"error");}});
$("help-open").addEventListener("click",()=>$("help-dialog").showModal());$("command-open").addEventListener("click",showCommands);$("command-search").addEventListener("input",renderCommands);
$("mobile-inspector").addEventListener("click",()=>document.body.classList.toggle("inspector-open"));$("inspector-close").addEventListener("click",()=>document.body.classList.remove("inspector-open"));
$("fullscreen-toggle").addEventListener("click",async()=>{try{if(document.fullscreenElement)await document.exitFullscreen();else await document.documentElement.requestFullscreen();}catch(error){toast(`Fullscreen is unavailable in this browser: ${error.message}`);}});
document.addEventListener("fullscreenchange",()=>{$("fullscreen-toggle").title=document.fullscreenElement?"Exit fullscreen":"Enter fullscreen";$("fullscreen-toggle").setAttribute("aria-label",$("fullscreen-toggle").title);});
document.addEventListener("keydown",event=>{
  const typing=event.target.closest("input,textarea,select,[contenteditable=true]");
  if((event.ctrlKey||event.metaKey)&&event.key.toLowerCase()==="s"){event.preventDefault();saveProject();return;}
  if((event.ctrlKey||event.metaKey)&&event.key.toLowerCase()==="k"){event.preventDefault();if(!$("command-dialog").open)showCommands();return;}
  if(typing||$$("dialog[open]").length||event.altKey||event.ctrlKey||event.metaKey)return;
  const key=event.key.toLowerCase();
  if(["nodes","plugins","transformers"].includes(view)&&!/^[1-9]$/.test(key)&&key!=="?")return;
  if(key===" "){event.preventDefault();togglePlay();}
  else if(key==="s"){event.preventDefault();if(run()&&!isFinished(run()))act("run.step",{count:1});}
  else if(key==="f"){event.preventDefault();forkRun();}
  else if(/^[1-9]$/.test(key))switchView(Object.keys(VIEW_NAMES)[Number(key)-1]);
  else if(key==="?")$("help-dialog").showModal();
});
let resizeTimer;window.addEventListener("resize",()=>{clearTimeout(resizeTimer);resizeTimer=setTimeout(()=>{lastChartSignature="";if(view==="timeline")renderTimeline();if(view==="data")drawDataset();},120);});
window.addEventListener("hashchange",()=>{const next=location.hash.slice(1);if(next!==view&&VIEW_NAMES[next])switchView(next);});
async function initialize(){
  switchView(VIEW_NAMES[location.hash.slice(1)]?location.hash.slice(1):"world");
  try{capabilities=await request("/api/capabilities");$("engine-version").textContent=`PyTorch ${capabilities.torch_version} · v${capabilities.version}`;renderPresets();}
  catch(error){toast(error.message,"error");}
  try{await loadDrafts();}catch(error){toast(`Draft restoration: ${error.message}`,"error");}
  await refresh();
  if(camera.x===0&&camera.y===0&&camera.zoom===1)fitWorld();
  setInterval(()=>{if(!document.hidden)refresh();},450);
  document.addEventListener("visibilitychange",()=>{if(!document.hidden)refresh();});
}
initialize();
window.StudioBridge={request,toast,downloadBlob,switchView,getView:()=>view,getSelectedRun:()=>run(),getRuns:()=>studio.runs,refresh,getEpoch:()=>workspaceEpoch,isWorkspaceReplaced:()=>workspaceReplaced};
window.StudioDrafts={markDirty:markDraftsDirty,flush:flushDrafts,get:()=>studioDrafts,hasPending:()=>draftsDirty};
document.addEventListener("input",event=>{if(event.target.closest("#config-form,#dataset-csv,#workspace-notes,#run-notes,#optimizer-state-json,#graph-plugin-manifest,#graph-properties,.graph-plugin-panel"))markDraftsDirty();});
