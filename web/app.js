const state={records:[],league:'all',week:'all',sort:'kickoff'};
const list=document.querySelector('#game-list');
const empty=document.querySelector('#empty-state');
const weekFilter=document.querySelector('#week-filter');
const sortFilter=document.querySelector('#sort-filter');
const dialog=document.querySelector('#detail-dialog');

const signed=n=>n==null?'—':`${n>=0?'+':''}${Number(n).toFixed(1)}`;
const number=n=>n==null?'—':Number(n).toFixed(1);
const date=s=>new Intl.DateTimeFormat(undefined,{weekday:'short',month:'short',day:'numeric',hour:'numeric',minute:'2-digit'}).format(new Date(s));
const forecast=(f,kind='')=>f?`<div class="forecast ${kind}"><strong>${signed(f.spread)} · ${number(f.total)}</strong><span>${number(f.away_score)}–${number(f.home_score)} score</span></div>`:`<div class="forecast"><strong>—</strong><span>not available</span></div>`;

function filtered(){
  let rows=state.records.filter(r=>(state.league==='all'||r.league===state.league)&&(state.week==='all'||String(r.week)===state.week));
  if(state.sort==='difference')rows.sort((a,b)=>Math.abs(b.model_market_difference?.spread||0)-Math.abs(a.model_market_difference?.spread||0));
  else if(state.sort==='confidence')rows.sort((a,b)=>a.unavailable_features.length-b.unavailable_features.length);
  else rows.sort((a,b)=>new Date(a.kickoff)-new Date(b.kickoff));
  return rows;
}
function render(){
  const rows=filtered(); list.replaceChildren(); empty.hidden=rows.length>0;
  rows.forEach(r=>{
    const button=document.createElement('button');button.className='game-row';
    button.innerHTML=`<div class="matchup-id"><span class="league-tag">${r.league.toUpperCase()}</span><div class="teams"><strong>${r.away_team} at ${r.home_team}</strong><small>W${r.week} · ${date(r.kickoff)}</small><span class="quality">${r.confidence}</span></div></div>${forecast(r.independent)}${forecast(r.market)}${forecast(r.calibrated)}<div class="forecast diff"><strong>${signed(r.model_market_difference?.spread)} · ${signed(r.model_market_difference?.total)}</strong><span>spread · total</span></div>`;
    button.addEventListener('click',()=>openDetail(r));list.append(button);
  });
}
function openDetail(r){
  const ev=r.market_evidence; const unavailable=r.unavailable_features.length?r.unavailable_features.join(', '):'None reported';
  document.querySelector('#dialog-content').innerHTML=`<div class="dialog-inner"><p class="dialog-kicker">${r.league.toUpperCase()} · Week ${r.week} · ${date(r.kickoff)}</p><h2>${r.away_team} <span>at</span> ${r.home_team}</h2><div class="detail-grid"><div><span>01 Model</span><strong>${signed(r.independent.spread)} / ${number(r.independent.total)}</strong></div><div><span>02 Market</span><strong>${signed(r.market?.spread)} / ${number(r.market?.total)}</strong></div><div><span>03 Calibrated</span><strong>${signed(r.calibrated?.spread)} / ${number(r.calibrated?.total)}</strong></div><div><span>04 Difference</span><strong>${signed(r.model_market_difference?.spread)} / ${signed(r.model_market_difference?.total)}</strong></div></div><dl class="disclosures"><dt>Model version</dt><dd>${r.model_version}</dd><dt>Information cutoff</dt><dd>${date(r.data_cutoff)}</dd><dt>Market evidence</dt><dd>${ev?`${ev.label}; ${ev.book_count_spread} spread / ${ev.book_count_total} total sources (${ev.providers.join(', ')})`:'No market evidence attached'}</dd><dt>Unavailable inputs</dt><dd>${unavailable}</dd><dt>Interpretation</dt><dd>Difference is the independent model minus market consensus. It is a diagnostic—not a betting recommendation.</dd></dl></div>`;
  dialog.showModal();
}
document.querySelector('.dialog-close').addEventListener('click',()=>dialog.close());
dialog.addEventListener('click',e=>{if(e.target===dialog)dialog.close()});
document.querySelectorAll('[data-league]').forEach(button=>button.addEventListener('click',()=>{document.querySelectorAll('[data-league]').forEach(b=>b.classList.remove('active'));button.classList.add('active');state.league=button.dataset.league;render()}));
weekFilter.addEventListener('change',()=>{state.week=weekFilter.value;render()});
sortFilter.addEventListener('change',()=>{state.sort=sortFilter.value;render()});

fetch('api/v1/predictions.json',{cache:'no-store'}).then(r=>{if(!r.ok)throw new Error(`HTTP ${r.status}`);return r.json()}).then(feed=>{
  if(feed.schema_version!==2)throw new Error('Unsupported feed schema');state.records=feed.records||[];
  [...new Set(state.records.map(r=>r.week))].sort((a,b)=>a-b).forEach(w=>weekFilter.add(new Option(`Week ${w}`,w)));
  document.querySelector('#data-status').textContent=feed.generated_at?`Updated ${date(feed.generated_at)}`:'Awaiting first slate';render();
}).catch(error=>{document.querySelector('#data-status').textContent='Ledger unavailable';empty.hidden=false;empty.querySelector('p:last-child').textContent=`The prediction artifact could not be loaded (${error.message}).`;});
