/* Accessible sequential decoder builder. No external assets or dependencies. */
"use strict";
window.FPBuilder = (() => {
  const TYPES = {
    transformer: ["Transformer block", "Attention, then feed-forward learning, with residual connections."],
    attention: ["Attention block", "Looks at earlier tokens to connect relevant information."],
    feed_forward: ["Feed-forward block", "Transforms each token independently. Pair it with attention."]
  };
  let nextId = 0;
  const block = type => ({ type, id: "fp-block-" + (++nextId) });
  const initial = () => ({ version: 1, width: 128, heads: 4, context: 128, blocks: [block("transformer"), block("transformer")] });
  function validate(a) {
    for (const [key, min, max] of [["width",64,768],["heads",1,16],["context",32,1024]]) {
      if (!Number.isInteger(a[key]) || a[key] < min || a[key] > max) return `${key}: choose a whole number from ${min} to ${max}.`;
    }
    if (a.width % a.heads || (a.width / a.heads) % 2) return "Width must divide evenly by heads, with an even number of features per head.";
    if (!a.blocks.length || a.blocks.length > 16) return "Add between 1 and 16 blocks.";
    if (a.blocks.some(b => !Object.hasOwn(TYPES, b.type))) return "Choose only the supported block types.";
    if (!a.blocks.some(b => b.type !== "feed_forward")) return "Add an attention or Transformer block so the model can see earlier tokens.";
    return "";
  }
  function serialize(d) {
    if (!d.builderEnabled) return undefined;
    const a = d.architecture, error = validate(a);
    if (error) throw new Error(error);
    return {version:1,width:a.width,heads:a.heads,context:a.context,blocks:a.blocks.map(b=>({type:b.type}))};
  }
  function render(h, d, rerender, preset, vocab) {
    if (!d.architecture) d.architecture = initial();
    const a = d.architecture;
    const section = h("section", {class:"brain-builder stack", "aria-labelledby":"builder-title"});
    section.append(h("h2", {id:"builder-title",text:"Build a brain with blocks"}),
      h("p", {class:"muted",text:"Choose a ready-made brain above, or arrange real decoder blocks yourself. Your custom order and dimensions are used by training."}),
      h("div", {class:"seg",role:"group","aria-label":"Architecture mode"},
        h("button", {type:"button","aria-pressed":String(!d.builderEnabled),onclick:()=>{d.builderEnabled=false;rerender();},text:"Use preset"}),
        h("button", {type:"button","aria-pressed":String(!!d.builderEnabled),onclick:()=>{d.builderEnabled=true;rerender();},text:"Build my own"})));
    if (!d.builderEnabled) return section;
    let drag = null;
    const status = h("p", {class:"small",role:"status","aria-live":"polite"});
    const estimate = h("p", {class:"small muted"});
    const refreshSummary = () => {
      const error = validate(a); status.textContent = error || `${a.blocks.length} blocks. Ready for server validation.`; status.className = "small" + (error ? " builder-error" : "");
      const count = a.blocks.reduce((sum,b)=>sum+(b.type==="transformer"?12:b.type==="attention"?4:8)*a.width*a.width+(b.type==="transformer"?2:1)*a.width,0)+(Number(vocab)||256)*a.width+a.width;
      estimate.textContent = Number.isFinite(count) ? `Approximately ${Math.round(count).toLocaleString()} trainable connections. Memory and speed depend on your data and hardware; preset fit labels do not validate a custom design.` : "Choose valid dimensions to estimate the model size.";
    };
    const field = (key,label,min,max,hint) => h("label",{class:"field"},h("span",{text:label}),h("input",{type:"number",min,max,step:1,value:a[key],oninput:e=>{a[key]=e.target.value===""?NaN:Number(e.target.value);refreshSummary();}}),h("span",{class:"small muted",text:hint}));
    section.append(h("p", {class:"note",text:"This builds a sequential residual decoder, not an arbitrary wiring diagram. Token embedding, positional encoding, final normalization and output stay fixed. All blocks share the same width, head count and context."}),
      h("div",{class:"builder-dimensions"},field("width","Width",64,768,"64–768 features per token."),field("heads","Attention heads",1,16,"1–16; width ÷ heads must be even."),field("context","Context",32,1024,"32–1,024 tokens of look-back.")));
    const list=h("ol",{class:"builder-list","aria-label":"Blocks in execution order"});
    const focus = id => { const target=section.querySelector('[data-focus="'+id+'"]'); if(target)target.focus(); };
    const change = (message,id) => {draw();refreshSummary();status.textContent=message+" "+(validate(a)||`${a.blocks.length} blocks in order.`);if(id)focus(id);};
    const add=(type,index=a.blocks.length)=>{if(a.blocks.length>=16){status.textContent="The limit is 16 blocks. Remove a block first.";return;}const b=block(type);a.blocks.splice(index,0,b);change(TYPES[type][0]+" added.",b.id);};
    const move=(from,to)=>{if(from<0||from>=a.blocks.length)return;to=Math.max(0,Math.min(a.blocks.length-1,to));const b=a.blocks.splice(from,1)[0];a.blocks.splice(to,0,b);change(`Moved ${TYPES[b.type][0]} to position ${to+1}.`,b.id);};
    const acceptDrop=(e,index)=>{e.preventDefault();e.stopPropagation();if(!drag)return;if(drag.type)add(drag.type,index);else{const old=a.blocks.findIndex(b=>b.id===drag.id);move(old,index>old?index-1:index);}drag=null;};
    const palette=h("div",{class:"builder-palette",role:"group","aria-label":"Add a block"},Object.entries(TYPES).map(([type,[name,hint]])=>h("button",{type:"button",class:"builder-tool",draggable:"true",ondragstart:e=>{drag={type};e.dataTransfer.setData("text/plain",type);e.dataTransfer.effectAllowed="copy";},ondragend:()=>{drag=null;},onclick:()=>add(type)},h("b",{text:"Add "+name}),h("small",{text:hint}))));
    function draw(){list.replaceChildren();a.blocks.forEach((b,i)=>{
      const row=h("li",{class:"builder-block",ondragover:e=>{if(drag){e.preventDefault();e.currentTarget.classList.add("drop-target");}},ondragleave:e=>e.currentTarget.classList.remove("drop-target"),ondrop:e=>acceptDrop(e,i)},
        h("button",{type:"button",class:"builder-handle",draggable:"true","data-focus":b.id,"aria-label":`${TYPES[b.type][0]}, position ${i+1}. Drag to move, or use the Move buttons.`,ondragstart:e=>{drag={id:b.id};e.dataTransfer.setData("text/plain",b.id);e.dataTransfer.effectAllowed="move";},ondragend:()=>{drag=null;list.querySelectorAll(".drop-target").forEach(n=>n.classList.remove("drop-target"));},text:String(i+1)}),
        h("div",{class:"builder-block-copy"},h("b",{text:TYPES[b.type][0]}),h("small",{text:TYPES[b.type][1]})),
        h("div",{class:"builder-actions"},h("button",{type:"button",disabled:i===0,"aria-label":`Move block ${i+1} up`,onclick:()=>move(i,i-1),text:"Move up"}),h("button",{type:"button",disabled:i===a.blocks.length-1,"aria-label":`Move block ${i+1} down`,onclick:()=>move(i,i+1),text:"Move down"}),h("button",{type:"button",disabled:a.blocks.length>=16,"aria-label":`Duplicate block ${i+1}`,onclick:()=>add(b.type,i+1),text:"Duplicate"}),h("button",{type:"button","aria-label":`Remove block ${i+1}`,onclick:()=>{a.blocks.splice(i,1);change("Block removed.",a.blocks[Math.min(i,a.blocks.length-1)]?.id);if(!a.blocks.length)palette.querySelector("button").focus();},text:"Remove"})));list.append(row);
    });}
    section.append(h("p",{class:"small muted",text:"Drag a block from the palette into the list, or tap Add. Reorder with drag or Move up / Move down. Every control works with a keyboard and touch."}),palette,
      h("div",{class:"builder-fixed",text:"Text tokens → embedding + position"}),list,
      h("div",{class:"builder-drop",ondragover:e=>{if(drag)e.preventDefault();},ondrop:e=>acceptDrop(e,a.blocks.length),text:"Drop here to append a block"}),
      h("div",{class:"builder-fixed",text:"Final normalization → next-token prediction"}),
      h("div",{class:"btnrow"},h("button",{type:"button",class:"btn quiet",onclick:()=>{a.width=Math.min(768,Math.max(64,preset.n_embd));a.heads=preset.n_head;a.context=Math.min(1024,Math.max(32,preset.block_size));a.blocks=Array.from({length:Math.min(16,preset.n_layer)},()=>block("transformer"));change("Copied selected preset, capped at builder limits.",a.blocks[0].id);rerender();},text:"Copy selected preset"}),h("button",{type:"button",class:"btn quiet",onclick:()=>{a.blocks=[];change("All blocks removed. Add an attention or Transformer block.");palette.querySelector("button").focus();},text:"Clear blocks"})),estimate,status);
    draw();refreshSummary();return section;
  }
  return {render,serialize,validate,initial};
})();
