/* Original Parameter Studio graph model. Browser and Node share these invariants. */
(function (root, factory) {
  "use strict";
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.GraphModel = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";
  const SAFE=9007199254740991n;
  class ExactInteger {
    constructor(raw){if(!/^-?(0|[1-9]\d*)$/.test(raw)||BigInt(raw)<=SAFE&&BigInt(raw)>=-SAFE)throw new Error("Expected an integer outside JavaScript's exact numeric range.");this.raw=raw;Object.freeze(this);}
    toString(){return this.raw;}
  }
  function clone(value){if(value instanceof ExactInteger)return new ExactInteger(value.raw);if(Array.isArray(value))return value.map(clone);if(value&&typeof value==="object")return Object.fromEntries(Object.entries(value).map(([key,item])=>[key,clone(item)]));return value;}
  function parseJSON(source,transport=false){
    const text=String(source);let cursor=0;
    const fail=message=>{throw new Error(`${message} at JSON character ${cursor+1}.`);};
    const whitespace=()=>{while(/[ \t\r\n]/.test(text[cursor]||"x"))cursor++;};
    function string(){const start=cursor;if(text[cursor++]!== '"')fail("Expected a quoted key");let escaped=false;while(cursor<text.length){const char=text[cursor++];if(!escaped&&char==='"'){try{return JSON.parse(text.slice(start,cursor));}catch(error){fail(error.message);}}if(!escaped&&char==="\\")escaped=true;else escaped=false;}fail("Unterminated JSON string");}
    function value(depth){
      whitespace();if(depth>64)fail("JSON nesting exceeds 64 levels");const char=text[cursor];
      if(char==='"')return string();
      if(char==="{"){cursor++;const result={},keys=new Set();whitespace();if(text[cursor]==="}"){cursor++;return result;}while(cursor<text.length){whitespace();const key=string();if(keys.has(key))fail(`Duplicate JSON key ${JSON.stringify(key)}`);keys.add(key);whitespace();if(text[cursor++]!==":")fail("Expected ':' after the key");const item=value(depth+1);Object.defineProperty(result,key,{value:item,enumerable:true,writable:true,configurable:true});whitespace();const separator=text[cursor++];if(separator==="}")return result;if(separator!==",")fail("Expected ',' or '}'");}fail("Unterminated JSON object");}
      if(char==="["){cursor++;const result=[];whitespace();if(text[cursor]==="]"){cursor++;return result;}while(cursor<text.length){result.push(value(depth+1));whitespace();const separator=text[cursor++];if(separator==="]")return result;if(separator!==",")fail("Expected ',' or ']'");}fail("Unterminated JSON array");}
      for(const [literal,result]of [["true",true],["false",false],["null",null]])if(text.startsWith(literal,cursor)){cursor+=literal.length;return result;}
      const token=text.slice(cursor).match(/^-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?/);
      if(token){cursor+=token[0].length;if(!/[.eE]/.test(token[0])){const integer=BigInt(token[0]);if(integer>SAFE||integer<-SAFE){if(transport)return new ExactInteger(token[0]);fail("Integer literal exceeds JavaScript's exact range; quote its decimal digits as text");}}const number=Number(token[0]);if(!Number.isFinite(number))fail("JSON numbers must be finite");return number;}
      fail("Expected a JSON value");
    }
    const result=value(0);whitespace();if(cursor!==text.length)fail("Unexpected trailing JSON content");return result;
  }
  function stringify(value,space=0){
    const indent=typeof space==="number"?" ".repeat(Math.max(0,Math.min(10,space))):String(space).slice(0,10),seen=new Set();
    function encode(item,depth){
      if(item instanceof ExactInteger)return item.raw;
      if(item===null)return "null";
      if(typeof item==="number"){if(!Number.isFinite(item))throw new Error("JSON numbers must be finite.");if(Object.is(item,-0))return "-0.0";if(Number.isInteger(item)&&!Number.isSafeInteger(item))return BigInt(item).toString();return JSON.stringify(item);}
      if(typeof item==="string"||typeof item==="boolean")return JSON.stringify(item);
      if(item===undefined)return undefined;
      if(!item||typeof item!=="object")throw new Error("Expected a strict JSON value.");
      if(depth>64)throw new Error("JSON nesting exceeds 64 levels.");
      if(seen.has(item))throw new Error("JSON cannot contain circular references.");seen.add(item);
      const padding=indent.repeat(depth),inner=indent.repeat(depth+1);let result;
      if(Array.isArray(item)){const entries=Array.from({length:item.length},(_,index)=>encode(Object.hasOwn(item,index)?item[index]:undefined,depth+1)??"null");result=entries.length?"["+(indent?"\n"+inner:"")+entries.join(indent?",\n"+inner:",")+(indent?"\n"+padding:"")+"]":"[]";}
      else{const entries=Object.entries(item).flatMap(([key,entry])=>{const encoded=encode(entry,depth+1);return encoded===undefined?[]:[JSON.stringify(key)+(indent?": ":":")+encoded];});result=entries.length?"{"+(indent?"\n"+inner:"")+entries.join(indent?",\n"+inner:",")+(indent?"\n"+padding:"")+"}":"{}";}
      seen.delete(item);return result;
    }
    return encode(value,0);
  }
  const uid = prefix => `${prefix}_${globalThis.crypto?.randomUUID?.().replaceAll("-", "").slice(0, 16) || Math.random().toString(36).slice(2)}`;
  const cleanView = view => ({x:Number.isFinite(view?.x)?view.x:0,y:Number.isFinite(view?.y)?view.y:0,zoom:Number.isFinite(view?.zoom)?Math.max(.04,Math.min(2.5,view.zoom)):1});
  function normalize(graph) {
    const result=clone(graph || {});
    result.name=String(result.name || "Untitled workflow");
    result.description=String(result.description || "");
    result.nodes=Array.isArray(result.nodes)?result.nodes:[];
    result.edges=Array.isArray(result.edges)?result.edges:[];
    result.view=cleanView(result.view);
    result.nodes.forEach((node,index)=>{node.params=node.params && typeof node.params==="object"&&!Array.isArray(node.params)?node.params:{};node.position={x:Number.isFinite(node.position?.x)?node.position.x:60+index*270,y:Number.isFinite(node.position?.y)?node.position.y:80};});
    return result;
  }
  const definition = (catalog,type) => (catalog?.types || []).find(entry=>entry.id===type);
  const ports = (def,direction) => Array.isArray(def?.[direction])?def[direction]:[];
  function compatible(source,target) {
    return !source||!target||source==="any"||target==="any"||source===target||(source==="integer"&&target==="number")||(["config","dataset"].includes(source)&&target==="object")||(source==="object"&&["config","dataset"].includes(target))||(source==="string"&&target==="run");
  }
  function defaultParams(def){const params={};[...ports(def,"inputs"),...(def?.parameters||[])].forEach(field=>{if(Object.hasOwn(field,"default"))params[field.name]=clone(field.default);});return params;}
  function hint(graph,catalog,node,port,direction,seen=new Set()){
    const entry=typeof node==="string"?graph?.nodes?.find(item=>item.id===node):node;if(!entry)return {};
    const def=definition(catalog,entry.type),record=ports(def,direction).find(item=>item.name===port);if(record?.type!=="tensor-expression")return {};
    const params={...defaultParams(def),...entry.params},op=entry.type,shape=(ranks,dtypes,width)=>({ranks,dtypes,...(Number.isSafeInteger(width)&&width>0?{width}:{})});
    const float=width=>shape([3],["float32"],width),integer=()=>shape([2],["int64"]),mask=()=>shape([2],["int64","bool"]);
    if(direction==="inputs"){
      if(op==="hf.op_embedding")return integer();
      if(op==="hf.op_positions"||op==="hf.op_attention"&&port==="mask")return mask();
      if(op==="hf.op_rope"&&port==="positions")return integer();
      if(op==="hf.op_custom")return shape([2,3],["float32","int64","bool"]);
      if(op==="hf.graph_model")return float(params.vocab_size);
      if(op==="hf.op_linear")return float(params.in_features);
      if(op==="hf.op_rmsnorm")return float(params.normalized_shape);
      if(op==="hf.op_rope")return float(params.num_heads*params.head_dim);
      if(op==="hf.op_attention")return float((port==="q"?params.num_heads:params.num_key_value_heads)*params.head_dim);
      if(op==="hf.op_swiglu")return float(params.hidden_size);
      return ["hf.op_silu","hf.op_add","hf.op_multiply"].includes(op)?float():{};
    }
    if(op==="hf.tensor_input")return shape([2],["int64","float32","bool"].includes(params.dtype)?[params.dtype]:[]);
    if(op==="hf.op_positions")return integer();
    if(op==="hf.op_embedding")return float(params.embedding_dim);
    if(op==="hf.op_linear")return float(params.out_features);
    if(op==="hf.op_rmsnorm")return float(params.normalized_shape);
    if(op==="hf.op_rope"||op==="hf.op_attention")return float(params.num_heads*params.head_dim);
    if(op==="hf.op_swiglu")return float(params.hidden_size);
    if(["hf.op_silu","hf.op_add","hf.op_multiply"].includes(op)){
      if(!entry.id||seen.has(entry.id))return float();seen.add(entry.id);
      const edge=graph?.edges?.find(item=>item.target===entry.id&&item.target_port===(op==="hf.op_silu"?"x":"a")),upstream=edge&&hint(graph,catalog,edge.source,edge.source_port,"outputs",seen);return float(upstream?.width);
    }
    // Custom programs and plugin recipes can change dtype/rank. Their outputs
    // remain unknown until compilation instead of inferring from example code.
    return {};
  }
  function hintsCompatible(source,target){return ["ranks","dtypes"].every(key=>!source?.[key]?.length||!target?.[key]?.length||source[key].some(value=>target[key].includes(value)));}
  function hintText(value){const parts=[];if(value?.dtypes?.length)parts.push(value.dtypes.join("/"));if(value?.ranks?.length)parts.push(`rank ${value.ranks.join("/")}`);if(value?.width)parts.push(`width ${value.width} (advisory)`);return parts.join(" · ")||"dtype / rank determined at compile";}
  function portCompatibility(graph,catalog,source,sourcePort,target,targetPort){
    const start=typeof source==="string"?graph?.nodes?.find(node=>node.id===source):source,end=typeof target==="string"?graph?.nodes?.find(node=>node.id===target):target,output=ports(definition(catalog,start?.type),"outputs").find(port=>port.name===sourcePort),input=ports(definition(catalog,end?.type),"inputs").find(port=>port.name===targetPort);
    if(!output||!input)return {compatible:false,reason:"Choose an existing output and input port."};
    if(!compatible(output.type,input.type))return {compatible:false,reason:`Cannot connect ${output.type} to ${input.type}. Add a conversion script.`};
    const sourceHint=hint(graph,catalog,start,sourcePort,"outputs"),targetHint=hint(graph,catalog,end,targetPort,"inputs"),valid=hintsCompatible(sourceHint,targetHint);
    return {compatible:valid,sourceHint,targetHint,reason:valid?"":`Incompatible tensor dtype or rank: ${hintText(sourceHint)} → ${hintText(targetHint)}. Choose a matching tensor operation; compilation checks all dimensions.`};
  }
  function connect(graph,catalog,source,sourcePort,target,targetPort) {
    if(source===target)throw new Error("A node cannot connect to itself.");
    const start=graph.nodes.find(node=>node.id===source),end=graph.nodes.find(node=>node.id===target);
    if(!start||!end)throw new Error("Both connected nodes must exist.");
    const output=ports(definition(catalog,start.type),"outputs").find(port=>port.name===sourcePort);
    const input=ports(definition(catalog,end.type),"inputs").find(port=>port.name===targetPort);
    if(!output||!input)throw new Error("Choose an existing output and input port.");
    if(!compatible(output.type,input.type))throw new Error(`Cannot connect ${output.type} to ${input.type}. Add a conversion script.`);
    const compatibility=portCompatibility(graph,catalog,start,sourcePort,end,targetPort);if(!compatibility.compatible)throw new Error(compatibility.reason);
    const adjacency=new Map(graph.nodes.map(node=>[node.id,[]]));
    graph.edges.filter(edge=>!(edge.target===target&&edge.target_port===targetPort)).forEach(edge=>adjacency.get(edge.source)?.push(edge.target));
    const seen=new Set();const pending=[target];
    while(pending.length){const id=pending.pop();if(id===source)throw new Error("This connection would create a cycle. Workflows must be directed acyclic graphs.");if(seen.has(id))continue;seen.add(id);pending.push(...(adjacency.get(id)||[]));}
    const result=normalize(graph);result.edges=result.edges.filter(edge=>!(edge.target===target&&edge.target_port===targetPort));
    result.edges.push({id:uid("edge"),source,source_port:sourcePort,target,target_port:targetPort});return result;
  }
  function add(graph,def,position) {
    if(!def)throw new Error("Choose an available node type.");
    const result=normalize(graph),params=defaultParams(def);
    result.nodes.push({id:uid("node"),type:def.id,title:def.title||def.id,position:{x:Math.round(position.x),y:Math.round(position.y)},params});return result;
  }
  function connectionOrigin(graph,catalog,origin){
    if(!origin||!["inputs","outputs"].includes(origin.direction))throw new Error("Choose an existing input or output port.");
    const node=graph?.nodes?.find(item=>item.id===origin.node),port=ports(definition(catalog,node?.type),origin.direction).find(item=>item.name===origin.port);
    if(!node||!port)throw new Error("The connection's original port is no longer available.");
    return {node,port,direction:origin.direction,opposite:origin.direction==="outputs"?"inputs":"outputs"};
  }
  function compatibleTargets(graph,catalog,origin,query=""){
    const source=connectionOrigin(graph,catalog,origin),search=String(query).trim().toLowerCase();
    return (catalog?.types||[]).flatMap(def=>{
      const candidate={type:def.id,params:defaultParams(def)},candidates=ports(def,source.opposite).filter(port=>source.direction==="outputs"?portCompatibility(graph,catalog,source.node,source.port.name,candidate,port.name).compatible:portCompatibility(graph,catalog,candidate,port.name,source.node,source.port.name).compatible);
      return candidates.filter(port=>!search||`${def.id} ${def.title||""} ${def.category||""} ${def.description||""} ${port.name} ${port.type}`.toLowerCase().includes(search)).map(port=>({definition:def,port,direction:source.opposite,hint:hint(graph,catalog,candidate,port.name,source.opposite)}));
    });
  }
  function addConnected(graph,catalog,origin,type,port,position){
    const source=connectionOrigin(graph,catalog,origin),def=definition(catalog,type);
    if(!compatibleTargets(graph,catalog,origin).some(item=>item.definition.id===type&&item.port.name===port))throw new Error("This node port is no longer compatible. Choose a compatible port.");
    const added=add(graph,def,position),node=added.nodes.at(-1);
    return source.direction==="outputs"?connect(added,catalog,source.node.id,source.port.name,node.id,port):connect(added,catalog,node.id,port,source.node.id,source.port.name);
  }
  const isArchitecture=def=>def?.id==="hf.tensor_input"||def?.id==="hf.graph_model"||def?.id?.startsWith("hf.op_");
  const cardParameters=def=>isArchitecture(def)?(def.parameters||[]).filter(field=>field.options||["integer","number","boolean","string"].includes(field.type)&&field.name!=="code").slice(0,3):[];
  function nodeFootprint(def){return {width:258,height:Math.max(150,125+28*(ports(def,"inputs").length+ports(def,"outputs").length))+(isArchitecture(def)?52+(cardParameters(def).length?8+30*cardParameters(def).length:0):0)};}
  function placeNode(graph,catalog,def,preferred){
    const point={x:Math.round(preferred.x),y:Math.round(preferred.y)},size=nodeFootprint(def),gap=22,step=24;
    const occupied=graph.nodes.map(node=>({...node.position,...nodeFootprint(definition(catalog,node.type))}));
    const free=candidate=>occupied.every(rect=>candidate.x+size.width+gap<=rect.x||rect.x+rect.width+gap<=candidate.x||candidate.y+size.height+gap<=rect.y||rect.y+rect.height+gap<=candidate.y);
    if(free(point))return point;
    let best=null,bestDistance=Infinity;
    // Search a deterministic nearby grid, keeping the nearest free candidate.
    for(let radius=1;radius<=64;radius++){
      const offsets=[];for(let x=-radius;x<=radius;x++)offsets.push({x,y:radius},{x,y:-radius});for(let y=-radius+1;y<radius;y++)offsets.push({x:radius,y},{x:-radius,y});
      offsets.sort((a,b)=>a.x*a.x+a.y*a.y-b.x*b.x-b.y*b.y||b.y-a.y||b.x-a.x);
      for(const offset of offsets){const distance=(offset.x*offset.x+offset.y*offset.y)*step*step;if(distance>=bestDistance)continue;const candidate={x:point.x+offset.x*step,y:point.y+offset.y*step};if(free(candidate)){best=candidate;bestDistance=distance;}}
      if(best&&(radius*step)**2>=bestDistance)return best;
    }
    // A dense or unusually tall graph still gets a non-overlapping new node.
    return best||{x:Math.max(point.x,...occupied.map(rect=>rect.x+rect.width+gap)),y:point.y};
  }
  function remove(graph,ids) {const set=new Set(ids);const result=normalize(graph);result.nodes=result.nodes.filter(node=>!set.has(node.id));result.edges=result.edges.filter(edge=>!set.has(edge.source)&&!set.has(edge.target));return result;}
  function copy(graph,ids) {const set=new Set(ids);return {nodes:clone(graph.nodes.filter(node=>set.has(node.id))),edges:clone(graph.edges.filter(edge=>set.has(edge.source)&&set.has(edge.target)))};}
  function paste(graph,clipboard,offset={x:35,y:35}) {
    if(!clipboard||!Array.isArray(clipboard.nodes)||!Array.isArray(clipboard.edges))throw new Error("Copy one or more nodes first.");
    const result=normalize(graph),ids=new Map(),created=[];
    for(const source of clipboard.nodes){const node=clone(source);node.id=uid("node");ids.set(source.id,node.id);node.position={x:source.position.x+offset.x,y:source.position.y+offset.y};result.nodes.push(node);created.push(node.id);}
    for(const source of clipboard.edges){if(ids.has(source.source)&&ids.has(source.target))result.edges.push({...clone(source),id:uid("edge"),source:ids.get(source.source),target:ids.get(source.target)});}
    return {graph:result,ids:created};
  }
  function parseConfigValue(text) {
    const config=parseJSON(text,true);
    if(!config||typeof config!=="object"||Array.isArray(config))throw new Error("Expected a JSON configuration object.");
    for(const owner of [config,config.dataset]){
      if(!owner||owner.seed===undefined)continue;
      const value=owner.seed;
      if(typeof value==="number"&&!Number.isInteger(value))throw new Error("Configuration seeds must be decimal integers.");
      if(typeof value==="number"&&!Number.isSafeInteger(value))throw new Error("Configuration numeric seeds must be safe integers; quote larger decimal seeds as strings.");
      if(!(value instanceof ExactInteger)&&typeof value!=="number"&&typeof value!=="string")throw new Error("Configuration seeds must be decimal integers.");
      const seed=String(value);if(!/^-?(?:0|[1-9]\d*)$/.test(seed))throw new Error("Configuration seeds must be decimal integers without leading zeros.");owner.seed=seed;
    }
    return config;
  }
  function parseValue(text,type) {
    if(type==="string"||type==="run"||type==="code")return text;
    if(type==="boolean"){if(text===true||text==="true")return true;if(text===false||text==="false")return false;throw new Error("Expected true or false.");}
    if(type==="number"||type==="integer"){const token=String(text).trim();if(!token)throw new Error("Enter a numeric value.");if(!/^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$/.test(token))throw new Error("Expected a finite decimal number.");if(/^[+-]?\d+$/.test(token)){const integer=BigInt(token);if(integer>SAFE||integer<-SAFE)throw new Error("Integer literal exceeds JavaScript's exact range; quote it in JSON, or use decimal/exponent notation for an approximate floating value.");}const value=Number(token);if(!Number.isFinite(value)||(type==="integer"&&!Number.isSafeInteger(value)))throw new Error(`Expected a finite ${type}${type==="integer"?" within JavaScript's exact integer range":""}.`);return value;}
    if(type==="config")return parseConfigValue(text);
    const value=parseJSON(text,true);
    if(type==="array"&&!Array.isArray(value))throw new Error("Expected a JSON array.");
    if(["object","config","dataset"].includes(type)&&(!value||typeof value!=="object"||Array.isArray(value)))throw new Error("Expected a JSON object.");
    if(type==="tensor-expression"&&(!value||typeof value!=="object"||Array.isArray(value)||value.kind!=="tensor-expression"))throw new Error('Expected a tensor-expression JSON object with kind "tensor-expression".');
    return value;
  }
  class History {
    constructor(limit=100){this.limit=limit;this.undoStack=[];this.redoStack=[];}
    push(graph){this.undoStack.push(clone(graph));if(this.undoStack.length>this.limit)this.undoStack.shift();this.redoStack=[];}
    undo(graph){if(!this.undoStack.length)return null;this.redoStack.push(clone(graph));return this.undoStack.pop();}
    redo(graph){if(!this.redoStack.length)return null;this.undoStack.push(clone(graph));return this.redoStack.pop();}
    clear(){this.undoStack=[];this.redoStack=[];}
  }
  function isTransformerJob(job){return !!job?.graph?.nodes?.some(node=>typeof node.type==="string"&&node.type.startsWith("hf."));}
  function transformerObservation(job){
    let model=null,report=null,metrics=null,loss=[];const artifacts=[];let visited=0;
    function walk(value,depth=0){
      if(++visited>20000||depth>8||!value||typeof value!=="object"||value instanceof ExactInteger)return;
      if(Array.isArray(value)){for(const item of value.slice(0,2000))if(item&&typeof item==="object")walk(item,depth+1);return;}
      if(value.kind==="model"&&value.parameter_count!==undefined)model=value;
      if(value.parameter_count!==undefined&&(value.parameters||value.tensors||value.tensor_count!==undefined))report=value;
      if(Array.isArray(value.loss_history)){const values=value.loss_history.map(item=>typeof item==="number"?item:item?.loss).filter(item=>typeof item==="number"&&Number.isFinite(item));if(values.length){loss=values;metrics=value;}}
      if(typeof value.download_url==="string"&&typeof value.sha256==="string"&&value.id&&!artifacts.some(item=>item.id===value.id))artifacts.push(value);
      for(const item of Object.values(value))if(item&&typeof item==="object")walk(item,depth+1);
    }
    const nodes=job?.nodes||job?.node_results||job?.results||{};
    for(const result of Object.values(nodes))walk(result?.outputs??result?.result);
    return {model,report,metrics,loss,artifacts};
  }
  function denseWorkflow(options={}){
    const integer=(value,name,min=1,max=1000000)=>{if(!Number.isSafeInteger(value)||value<min||value>max)throw new Error(`${name} must be an integer between ${min} and ${max.toLocaleString("en-US")}.`);return value;};
    const widths=options.widths??[16,16];if(!Array.isArray(widths)||widths.length>128)throw new Error("Hidden widths must be an array of at most 128 positive integers.");widths.forEach(value=>integer(value,"Hidden width"));
    const input=integer(options.input_features??1,"Input features"),output=integer(options.output_features??1,"Output features"),steps=integer(options.steps??20,"Training steps"),seed=integer(options.seed??42,"Seed",0,2147483647),lr=options.learning_rate??.01;
    if(typeof lr!=="number"||!Number.isFinite(lr)||lr<=0)throw new Error("Learning rate must be a positive finite number.");
    const activation=options.activation??"GELU",normalization=options.normalization??"RMSNorm",optimizer=options.optimizer??"AdamW";
    if(!["GELU","ReLU","Tanh","SiLU"].includes(activation))throw new Error("Choose GELU, ReLU, Tanh, or SiLU.");
    if(!["RMSNorm","LayerNorm","None"].includes(normalization))throw new Error("Choose RMSNorm, LayerNorm, or None.");
    if(!["AdamW","Adam","SGD","RMSprop"].includes(optimizer))throw new Error("Choose a supported optimizer.");
    const layers=[];let previous=input;for(const width of widths){layers.push({type:"Linear",in_features:previous,out_features:width,bias:true});if(normalization!=="None")layers.push({type:normalization,normalized_shape:width,eps:1e-6});layers.push({type:activation});previous=width;}layers.push({type:"Linear",in_features:previous,out_features:output,bias:true});
    const fields=options.fields??{x:[[-1],[-.5],[0],[.5],[1]],labels:[[-2],[-1],[0],[1],[2]]};if(!fields||Array.isArray(fields)||typeof fields!=="object")throw new Error("Training fields must be a JSON object.");
    const node=(id,type,params,x,y)=>({id,type,params,position:{x,y}}),edge=(source,source_port,target,target_port)=>({id:`${source}-${source_port}-${target}-${target_port}`,source,source_port,target,target_port});
    return normalize({name:options.name||"My multi-parameter regression",description:"Editable dense PyTorch architecture, training data, optimizer, actual metrics, tensor inspection, and a portable checkpoint. Execution starts only when explicitly queued.",nodes:[node("architecture","hf.sequential",{layers,seed},0,0),node("data","hf.data",{fields},0,300),node("optimizer","hf.optimizer",{name:optimizer,lr,weight_decay:0,momentum:0},0,560),node("train","hf.train",{steps,batch_size:Math.min(32,Array.isArray(fields.x)?Math.max(1,fields.x.length):3),loss:"mse",clip_grad_norm:1},360,170),node("inspect","hf.inspect",{},720,0),node("checkpoint","hf.checkpoint",{},720,340)],edges:[edge("architecture","model","train","model"),edge("data","data","train","data"),edge("optimizer","optimizer","train","optimizer"),edge("train","model","inspect","model"),edge("train","model","checkpoint","model")],view:{x:20,y:20,zoom:.8}});
  }
  return {clone,uid,cleanView,normalize,definition,ports,compatible,connect,add,remove,copy,paste,parseValue,parseConfigValue,parseJSON,parseTransportJSON:text=>parseJSON(text,true),stringify,ExactInteger,History,isTransformerJob,transformerObservation,denseWorkflow,nodeFootprint,placeNode,connectionOrigin,compatibleTargets,addConnected,isArchitecture,cardParameters,hint,hintsCompatible,hintText,portCompatibility};
});
