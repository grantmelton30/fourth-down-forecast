const state={records:[],explorer:null,league:'all',week:'all',sort:'kickoff',group:'all',teamQuery:'',view:'projections',tracker:null,recordLeague:'all',explorerLeague:'nfl',teamLeague:'nfl',team:null};
const $=selector=>document.querySelector(selector);
const $$=selector=>[...document.querySelectorAll(selector)];
const list=$('#game-list'),empty=$('#empty-state'),weekFilter=$('#week-filter'),sortFilter=$('#sort-filter'),groupFilter=$('#group-filter'),teamFilter=$('#team-filter'),dialog=$('#detail-dialog');
const signed=n=>n==null?'—':`${n>=0?'+':''}${Number(n).toFixed(1)}`;
const number=n=>n==null?'—':Number(n).toFixed(1);
const date=s=>s?new Intl.DateTimeFormat(undefined,{weekday:'short',month:'short',day:'numeric',hour:'numeric',minute:'2-digit'}).format(new Date(s)):'TBD';
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[c]));
// `label` renders only on narrow screens. The table header carries the column names on
// desktop, but it is hidden below 900px -- which left four unlabelled numbers with no way
// to tell the model from the market. Each cell now names itself when the header is gone.
const forecast=(f,kind='',missing='not available',label='')=>{
  const tag=label?`<b class="cell-label">${label}</b>`:'';
  return f?`<div class="forecast ${kind}">${tag}<strong>${signed(f.spread)} · ${number(f.total)}</strong><span>${number(f.away_score)}–${number(f.home_score)} score</span></div>`
          :`<div class="forecast">${tag}<strong>—</strong><span>${missing}</span></div>`;
};
const ruleCell=r=>{const t=r.tracked_rule;if(!t)return`<div class="rulecell"><b class="cell-label">Tracked rule</b><span class="badge">—</span><small>rule not evaluated</small></div>`;return`<div class="rulecell ${t.qualifies?'on':''}"><b class="cell-label">Tracked rule</b><span class="badge">${esc(t.label||'—')}</span><small>${esc(t.reason||'')}</small></div>`};
const leagueData=league=>state.explorer?.leagues?.[league]||{ratings:[],schedule:[],summary:{}};
const normalize=value=>String(value||'').toLocaleLowerCase().normalize('NFD').replace(/[\u0300-\u036f]/g,'');
const ratingFor=(league,team)=>leagueData(league).ratings.find(r=>r.team===team);
const groupFor=(league,team)=>ratingFor(league,team)?.group||null;

function filtered(){
  const query=normalize(state.teamQuery.trim());
  let rows=state.records.filter(r=>
    (state.league==='all'||r.league===state.league)
    &&(state.week==='all'||String(r.week)===state.week)
    &&(state.group==='all'||groupFor(r.league,r.home_team)===state.group||groupFor(r.league,r.away_team)===state.group)
    &&(!query||normalize(r.home_team).includes(query)||normalize(r.away_team).includes(query))
  );
  if(state.sort==='difference')rows.sort((a,b)=>Math.abs(b.model_market_difference?.spread||0)-Math.abs(a.model_market_difference?.spread||0));
  else if(state.sort==='confidence')rows.sort((a,b)=>(a.quality_reasons?.length||0)-(b.quality_reasons?.length||0));
  else rows.sort((a,b)=>new Date(a.kickoff)-new Date(b.kickoff));
  return rows;
}
function renderProjections(){
  const rows=filtered();list.replaceChildren();empty.hidden=rows.length>0;
  empty.querySelector('h2').textContent=state.records.length?'No projections match these filters.':'No published slate yet.';
  empty.querySelector('p:last-child').textContent=state.records.length?'Try a different team, conference/division, league, or week.':'The site is ready, but it will not invent picks. Run the prospective model refresh to append evidence to the ledger; this page then updates from the versioned JSON artifact.';
  rows.forEach(r=>{
    const button=document.createElement('button');button.className=`game-row ${r.out_of_distribution?'flagged':''}`;
    const warning=r.out_of_distribution?'<span class="warning-mark" aria-label="Extreme model and market disagreement">!</span>':'';
    button.innerHTML=`<div class="matchup-id"><span class="league-tag">${r.league.toUpperCase()}</span><div class="teams"><strong>${esc(r.away_team)} at ${esc(r.home_team)} ${warning}</strong><small>W${r.week} · ${date(r.kickoff)}</small><span class="quality ${r.confidence}">${esc(r.confidence)}</span></div></div>${forecast(r.independent,'','not available','Model')}${forecast(r.market,'','No line posted yet','Market')}<div class="forecast diff"><b class="cell-label">Difference</b><strong>${signed(r.model_market_difference?.spread)} · ${signed(r.model_market_difference?.total)}</strong><span>spread · total</span></div>${ruleCell(r)}`;
    button.addEventListener('click',()=>openDetail(r));list.append(button);
  });
}

function renderRecord(){
  const t=state.tracker,cards=$('#rule-cards'),list=$('#bet-list'),blank=$('#bet-empty');
  if(!cards)return;
  if(!t){cards.replaceChildren();return}
  const pct=v=>v==null?'—':`${Number(v).toFixed(1)}%`;
  // 'all' shows the pooled bankroll FIRST, then each rule, so the breakout is one glance
  // away rather than one click. A single league shows only that rule.
  const lg=state.recordLeague;
  const shown=lg==='all'
    ? [{...t.combined,total:true},...Object.values(t.rules)]
    : Object.values(t.rules).filter(r=>r.league===lg);
  cards.innerHTML=shown.map(r=>{
    // Colour tracks the only comparison that matters: the break-even rate, not zero.
    const dir=r.win_pct==null?'':(r.win_pct>=t.breakeven?'up':'down');
    const vs=r.win_pct==null?'—':`${r.win_pct>=t.breakeven?'+':''}${(r.win_pct-t.breakeven).toFixed(1)} pts vs break-even`;
    const ci=r.ci_low==null?'not enough settled bets':`${pct(r.ci_low)} to ${pct(r.ci_high)}`;
    return `<article class="rule-card ${dir} ${r.total?'total':''}"><h3>${r.total?'Combined · both rules':`${esc(r.rule)} · ${esc(r.league.toUpperCase())}`}</h3>
      <p class="sub">${esc(r.headline)}</p>
      <div class="big"><strong>${pct(r.win_pct)}</strong><em>${esc(vs)}</em></div>
      <dl><dt>Record</dt><dd>${r.wins}-${r.losses}${r.pushes?` (${r.pushes} push)`:''}</dd>
      <dt>Units @ -110</dt><dd>${r.units>=0?'+':''}${Number(r.units).toFixed(1)}</dd>
      <dt>95% interval</dt><dd>${esc(ci)}</dd>
      <dt>Pending</dt><dd>${r.pending}</dd></dl>
      <p class="status">${esc(r.status)}<br>${esc(r.prereg)}</p></article>`}).join('');
  const bets=(t.bets||[]).filter(b=>lg==='all'||b.league===lg);
  blank.hidden=bets.length>0;
  blank.textContent=(t.bets||[]).length?`No ${lg.toUpperCase()} bets logged yet.`:'No bets logged yet. The trackers append here automatically once the season starts.';
  list.innerHTML=bets.map(b=>{
    const res=(b.result||'').toLowerCase()||'pending';
    const label=res==='pending'?'PENDING':b.result;
    const extra=b.forecast_wind_mph!=null?`${Number(b.forecast_wind_mph).toFixed(0)} mph forecast`
      :(b.edge!=null?`${signed(b.edge)} pt edge`:'');
    return `<div class="bet-row"><div><b>${esc(b.away_team)} at ${esc(b.home_team)}</b>
      <small>W${b.week} · ${date(b.kickoff)}${extra?` · ${esc(extra)}`:''}</small></div>
      <span class="tag">${esc(b.rule)}</span><span>${esc(b.side||'—')}</span>
      <span>${number(b.market_total_at_bet)}</span><span>${number(b.actual_total)}</span>
      <span class="res ${res}">${esc(label)}</span></div>`}).join('');
}
function populateProjectionFilters(){
  const leagues=state.league==='all'?['nfl','ncaa']:[state.league];
  const ratings=leagues.flatMap(league=>leagueData(league).ratings.map(r=>({...r,league})));
  const groups=[...new Set(ratings.map(r=>r.group).filter(Boolean))].sort((a,b)=>a.localeCompare(b));
  const currentGroup=groups.includes(state.group)?state.group:'all';state.group=currentGroup;
  groupFilter.replaceChildren(new Option('All conferences & divisions','all'),...groups.map(group=>new Option(group,group,false,group===currentGroup)));
  groupFilter.value=currentGroup;
  const names=[...new Set(ratings.map(r=>r.team))].sort((a,b)=>a.localeCompare(b));
  $('#projection-teams').replaceChildren(...names.map(name=>{const option=document.createElement('option');option.value=name;return option}));
}
function winProbRow(r){
  const p=r.independent?.home_win_probability;
  if(p==null)return'';
  const homePct=(100*p).toFixed(1),awayPct=(100*(1-p)).toFixed(1);
  return`<dt>Win probability</dt><dd><span class="split-bar"><span class="split-home" style="width:${homePct}%"></span><span class="split-away" style="width:${awayPct}%"></span></span><small>${esc(r.home_team)} ${homePct}% (home) · ${esc(r.away_team)} ${awayPct}% (away)</small></dd>`;
}
function intervalRow(r){
  const lo=r.independent?.interval_80_low,hi=r.independent?.interval_80_high,point=r.independent?.spread;
  if(lo==null||hi==null)return'';
  const DOMAIN=50,toPct=v=>(100*(Math.max(-DOMAIN,Math.min(DOMAIN,v))+DOMAIN)/(2*DOMAIN)).toFixed(1);
  const leftPct=toPct(lo),rightPct=toPct(hi),pointPct=point==null?null:toPct(point);
  return`<dt>80% interval (spread)</dt><dd><span class="range-bar"><span class="range-fill" style="left:${leftPct}%;right:${100-rightPct}%"></span>${pointPct==null?'':`<span class="range-point" style="left:${pointPct}%"></span>`}</span><small>${signed(lo)} to ${signed(hi)}</small></dd>`;
}
function openDetail(r){
  const ev=r.market_evidence,unavailable=r.unavailable_features?.length?r.unavailable_features.join(', '):'None reported';
  const warnings=r.warnings?.length?`<div class="alert-block"><strong>Review required</strong>${r.warnings.map(w=>`<p>${esc(w)}</p>`).join('')}</div>`:'';
  const reasons=r.quality_reasons?.length?r.quality_reasons.join('; '):'No quality blockers reported';
  $('#dialog-content').innerHTML=`<div class="dialog-inner"><p class="dialog-kicker">${r.league.toUpperCase()} · Week ${r.week} · ${date(r.kickoff)}</p><h2>${esc(r.away_team)} <span>at</span> ${esc(r.home_team)}</h2>${warnings}<div class="detail-grid"><div><span>01 Model</span><strong>${signed(r.independent.spread)} / ${number(r.independent.total)}</strong></div><div><span>02 Market</span><strong>${signed(r.market?.spread)} / ${number(r.market?.total)}</strong></div><div><span>03 Difference</span><strong>${signed(r.model_market_difference?.spread)} / ${signed(r.model_market_difference?.total)}</strong></div></div><dl class="disclosures">${winProbRow(r)}${intervalRow(r)}<dt>Quality</dt><dd>${esc(r.confidence)} — ${esc(reasons)}</dd><dt>Model version</dt><dd>${esc(r.model_version)}</dd><dt>Information cutoff</dt><dd>${date(r.data_cutoff)}</dd><dt>Market evidence</dt><dd>${ev?`${esc(ev.label)}; ${ev.book_count_spread} spread / ${ev.book_count_total} total sources (${ev.providers.map(esc).join(', ')})`:'No market evidence attached'}</dd><dt>Calibration permissions</dt><dd>Spread: ${r.calibration_status?.spread?'validated':'not validated'} · Total: ${r.calibration_status?.total?'validated':'not validated'}</dd><dt>Unavailable inputs</dt><dd>${esc(unavailable)}</dd><dt>Tracked rule</dt><dd>${r.tracked_rule?`<strong>${esc(r.tracked_rule.label)}</strong> — ${esc(r.tracked_rule.reason)}<br><small>Pre-registered tracking rule (${esc(r.tracked_rule.rule)}), recorded for a forward record. Not an authorisation to bet: this model's bet gates fail.</small>`:'Not evaluated'}</dd><dt>Interpretation</dt><dd>Difference is the independent model minus the market reference. Large disagreement triggers investigation, never an automatic pick.</dd></dl></div>`;
  dialog.showModal();
}

function setButtonGroup(selector,value,attribute){
  $$(selector).forEach(button=>button.classList.toggle('active',button.dataset[attribute]===value));
}
function showView(view){
  state.view=view;
  $$('.workspace-view').forEach(panel=>panel.classList.toggle('active',panel.id===`${view}-view`));
  setButtonGroup('[data-view]',view,'view');
  if(view==='league')renderLeague();
  if(view==='teams')renderTeam();
}
function renderLeague(){
  const data=leagueData(state.explorerLeague),records=state.records.filter(r=>r.league===state.explorerLeague);
  const marketCount=records.filter(r=>r.market).length,consensus=records.filter(r=>r.market_evidence?.is_consensus).length;
  $('#league-summary').innerHTML=[
    ['Rated teams',data.summary.rated_teams||0],['Season games',data.summary.scheduled_games||0],
    ['Published forecasts',records.length],['Market references',marketCount],
    ['3+ book consensus',consensus],['Pick eligible',records.filter(r=>r.pick_eligible).length],
  ].map(([label,value])=>`<div><span>${label}</span><strong>${value}</strong></div>`).join('');
  $('#rankings-period').textContent=`${data.season||'—'} · Week ${data.week||'—'}`;
  const ratings=data.ratings.slice().sort((a,b)=>(a.net_rank||999)-(b.net_rank||999));
  const maxAbsRating=Math.max(1e-9,...ratings.map(r=>Math.abs(r.net_rating||0)));
  $('#ranking-list').innerHTML=ratings.map(r=>{
    const pct=(100*Math.abs(r.net_rating||0)/maxAbsRating).toFixed(1);
    const dir=(r.net_rating||0)>=0?'pos':'neg';
    return `<button class="ranking-row" data-team-jump="${esc(r.team)}"><span><b>${r.net_rank||'—'}</b><i>${esc(r.team)}</i><small>${esc(r.conference||state.explorerLeague.toUpperCase())}</small></span><strong>${signed(r.net_rating)}</strong><em>#${r.off_rank||'—'}</em><em>#${r.def_rank||'—'}</em><span class="rating-bar-track"><span class="rating-bar-fill ${dir}" style="width:${pct}%"></span></span></button>`;
  }).join('');
  $('#evidence-overview').innerHTML=`<div class="evidence-score"><strong>${records.filter(r=>r.confidence==='established').length}</strong><span>established forecasts</span></div><dl><dt>Incomplete</dt><dd>${records.filter(r=>r.confidence==='incomplete').length}</dd><dt>Extreme-disagreement review</dt><dd>${records.filter(r=>r.out_of_distribution).length}</dd><dt>Spread calibration</dt><dd>${records.some(r=>r.calibration_status?.spread)?'validated':'not validated'}</dd><dt>Total calibration</dt><dd>${records.some(r=>r.calibration_status?.total)?'validated':'not validated'}</dd></dl><p>Quality describes evidence maturity. It is not a pick grade.</p>`;
  $$('[data-team-jump]').forEach(button=>button.addEventListener('click',()=>{state.teamLeague=state.explorerLeague;state.team=button.dataset.teamJump;showView('teams')}));
}
function populateTeams(){
  const data=leagueData(state.teamLeague),select=$('#team-select');
  const names=data.ratings.map(r=>r.team).sort((a,b)=>a.localeCompare(b));
  if(!names.includes(state.team))state.team=names[0]||null;
  select.replaceChildren(...names.map(name=>new Option(name,name,false,name===state.team)));
}
function renderTeam(){
  setButtonGroup('[data-team-league]',state.teamLeague,'teamLeague');populateTeams();
  const data=leagueData(state.teamLeague),rating=data.ratings.find(r=>r.team===state.team)||{};
  const schedule=data.schedule.filter(g=>g.home_team===state.team||g.away_team===state.team).sort((a,b)=>(a.week||0)-(b.week||0));
  const forecasts=state.records.filter(r=>r.league===state.teamLeague&&(r.home_team===state.team||r.away_team===state.team));
  $('#team-identity').innerHTML=`<div><p>${esc(rating.conference||state.teamLeague.toUpperCase())} · ${data.season||''}</p><h2>${esc(state.team||'Choose a team')}</h2></div><div class="team-metrics"><span><small>Overall</small><strong>#${rating.net_rank||'—'}</strong></span><span><small>Offense</small><strong>#${rating.off_rank||'—'}</strong></span><span><small>Defense</small><strong>#${rating.def_rank||'—'}</strong></span><span><small>Power</small><strong>${signed(rating.net_rating)}</strong></span></div>`;
  $('#team-schedule-label').textContent=`${schedule.length} games`;
  $('#team-schedule').innerHTML=schedule.map(g=>{const home=g.home_team===state.team,opponent=home?g.away_team:g.home_team,result=g.completed?(g.home_points!=null?`${g.away_points}–${g.home_points}`:'Final'):'Upcoming';return `<div><span><b>W${g.week}</b><small>${date(g.kickoff)}</small></span><strong>${home?'vs':'at'} ${esc(opponent)}</strong><em>${result}</em></div>`}).join('')||'<p class="panel-empty">No schedule is available.</p>';
  $('#team-forecasts').innerHTML=forecasts.map(r=>`<button data-forecast-id="${r.prediction_id}"><span>W${r.week} · ${esc(r.away_team)} at ${esc(r.home_team)}</span><strong>${signed(r.independent.spread)} / ${number(r.independent.total)}</strong><small>${esc(r.confidence)}</small></button>`).join('')||'<p class="panel-empty">No published forecast for this team yet.</p>';
  $$('[data-forecast-id]').forEach(button=>button.addEventListener('click',()=>openDetail(state.records.find(r=>r.prediction_id===button.dataset.forecastId))));
}

$('.dialog-close').addEventListener('click',()=>dialog.close());
dialog.addEventListener('click',e=>{if(e.target===dialog)dialog.close()});
$$('[data-view]').forEach(button=>button.addEventListener('click',()=>showView(button.dataset.view)));
$$('[data-record]').forEach(button=>button.addEventListener('click',()=>{state.recordLeague=button.dataset.record;setButtonGroup('[data-record]',state.recordLeague,'record');renderRecord()}));
$$('[data-league]').forEach(button=>button.addEventListener('click',()=>{setButtonGroup('[data-league]',button.dataset.league,'league');state.league=button.dataset.league;populateProjectionFilters();renderProjections()}));
$$('[data-explorer-league]').forEach(button=>button.addEventListener('click',()=>{state.explorerLeague=button.dataset.explorerLeague;setButtonGroup('[data-explorer-league]',state.explorerLeague,'explorerLeague');renderLeague()}));
$$('[data-team-league]').forEach(button=>button.addEventListener('click',()=>{state.teamLeague=button.dataset.teamLeague;state.team=null;renderTeam()}));
$('#team-select').addEventListener('change',e=>{state.team=e.target.value;renderTeam()});
weekFilter.addEventListener('change',()=>{state.week=weekFilter.value;renderProjections()});
sortFilter.addEventListener('change',()=>{state.sort=sortFilter.value;renderProjections()});
groupFilter.addEventListener('change',()=>{state.group=groupFilter.value;renderProjections()});
teamFilter.addEventListener('input',()=>{state.teamQuery=teamFilter.value;renderProjections()});
$('#clear-filters').addEventListener('click',()=>{state.league='all';state.week='all';state.group='all';state.teamQuery='';teamFilter.value='';weekFilter.value='all';setButtonGroup('[data-league]','all','league');populateProjectionFilters();renderProjections()});

Promise.all([
  fetch('api/v1/predictions.json',{cache:'no-store'}).then(r=>{if(!r.ok)throw new Error(`predictions HTTP ${r.status}`);return r.json()}),
  fetch('api/v1/explorer.json',{cache:'no-store'}).then(r=>{if(!r.ok)throw new Error(`explorer HTTP ${r.status}`);return r.json()}),
  // The record is additive: an older deploy without tracker.json still renders everything
  // else rather than failing the whole page on one missing artifact.
  fetch('api/v1/tracker.json',{cache:'no-store'}).then(r=>r.ok?r.json():null).catch(()=>null),
]).then(([feed,explorer,tracker])=>{
  if(feed.schema_version!==2)throw new Error('Unsupported prediction schema');
  state.records=feed.records||[];state.explorer=explorer;
  [...new Set(state.records.map(r=>r.week))].sort((a,b)=>a-b).forEach(w=>weekFilter.add(new Option(`Week ${w}`,w)));
  $('#data-status').textContent=feed.generated_at?`Updated ${date(feed.generated_at)}`:'Awaiting first slate';
  state.tracker=tracker;populateProjectionFilters();renderProjections();renderLeague();renderTeam();renderRecord();
}).catch(error=>{$('#data-status').textContent='Data unavailable';empty.hidden=false;empty.querySelector('p:last-child').textContent=`The publication artifacts could not be loaded (${error.message}).`;});
