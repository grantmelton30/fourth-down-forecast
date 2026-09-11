const state={records:[],explorer:null,league:'all',status:'upcoming',week:'all',sort:'kickoff',group:'all',teamQuery:'',view:'projections',tracker:null,recordLeague:'all',explorerLeague:'nfl',teamLeague:'nfl',team:null};
const $=selector=>document.querySelector(selector);
const $$=selector=>[...document.querySelectorAll(selector)];
const list=$('#game-list'),empty=$('#empty-state'),weekFilter=$('#week-filter'),sortFilter=$('#sort-filter'),groupFilter=$('#group-filter'),teamFilter=$('#team-filter'),dialog=$('#detail-dialog');

// Signed figures use a true minus sign: a hyphen is easy to lose at 13px, and the sign is
// the reading that matters most on this page. Rounding happens before the sign is chosen,
// so -0.04 reads "+0.0" rather than "-0.0".
const fmt=(n,places)=>{
  if(n==null||!Number.isFinite(Number(n)))return null;
  const scale=10**places,v=Math.round(Number(n)*scale)/scale;
  return`${v>=0?'+':'−'}${Math.abs(v).toFixed(places)}`;
};
const signed=n=>fmt(n,1)??'—';
// Net ratings sit around ±0.1. Formatting them with `signed` rendered every team in the
// power table as "+0.1" -- Ohio State at 0.0858 and Texas Tech at 0.0597 alike -- so the
// column carried no information at all.
const rating=n=>fmt(n,3)??'—';
const number=n=>n==null?'—':Number(n).toFixed(1);
const date=s=>s?new Intl.DateTimeFormat(undefined,{weekday:'short',month:'short',day:'numeric',hour:'numeric',minute:'2-digit'}).format(new Date(s)):'TBD';
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[c]));

// Below a field goal, a model-market gap sits inside the noise of a single key number, so it
// is drawn neutral rather than as a signal. Hue never carries the sign alone: the +/- and
// the triangle say the same thing for readers who cannot separate red from green.
const SIGNAL_POINTS=3;
const tone=n=>n==null||Math.abs(n)<SIGNAL_POINTS?'flat':(n>0?'pos':'neg');
const direction=n=>n==null||Math.abs(n)<0.5?'flat':(n>0?'up':'dn');
const diffFigure=(spread,total,sep=' · ')=>`<span class="${tone(spread)}"><i class="dir dir-${direction(spread)}" aria-hidden="true"></i>${signed(spread)}</span>${sep}<span class="${tone(total)}">${signed(total)}</span>`;

// `label` renders only on narrow screens. The table header carries the column names on
// desktop, but it is hidden below 760px -- which left four unlabelled numbers with no way
// to tell the model from the market. Each cell now names itself when the header is gone.
const forecast=(f,kind='',missing='not available',label='')=>{
  const tag=label?`<b class="cell-label">${label}</b>`:'';
  return f?`<div class="forecast ${kind}">${tag}<strong>${signed(f.spread)} · ${number(f.total)}</strong><span>${number(f.away_score)}–${number(f.home_score)} score</span></div>`
          :`<div class="forecast">${tag}<strong>—</strong><span>${missing}</span></div>`;
};
const diffCell=d=>`<div class="forecast diff"><b class="cell-label">Difference</b><strong>${diffFigure(d?.spread,d?.total)}</strong><span>spread · total</span></div>`;
// The bar reaches its end at a 24-point spread difference; anything larger is clamped.
const DISAGREE_SPAN=24;
const disagreeCell=d=>{
  const s=d?.spread;
  const fill=s==null?'':`<i class="fill ${tone(s)}" style="${s>=0?'left':'right'}:50%;width:${Math.min(50,50*Math.abs(s)/DISAGREE_SPAN).toFixed(1)}%"></i>`;
  const label=s==null?'No market spread to compare':`Spread difference ${signed(s)} points`;
  return`<div class="disagree"><b class="cell-label">Spread disagreement</b><span class="track" role="img" aria-label="${esc(label)}"><i class="tick"></i>${fill}</span></div>`;
};
const ruleCell=r=>{const t=r.tracked_rule;if(!t)return`<div class="rulecell"><b class="cell-label">Tracked rule</b><span class="badge">—</span><small>rule not evaluated</small></div>`;return`<div class="rulecell ${t.qualifies?'on':''}"><b class="cell-label">Tracked rule</b><span class="badge">${esc(t.label||'—')}</span><small title="${esc(t.reason||'')}">${esc(t.reason||'')}</small></div>`};
const leagueData=league=>state.explorer?.leagues?.[league]||{ratings:[],schedule:[],summary:{}};
const normalize=value=>String(value||'').toLocaleLowerCase().normalize('NFD').replace(/[\u0300-\u036f]/g,'');
const ratingFor=(league,team)=>leagueData(league).ratings.find(r=>r.team===team);
const groupFor=(league,team)=>ratingFor(league,team)?.group||null;

// The feed carries every game the model has ever projected, oldest first, so an
// unfiltered Kickoff sort opened this tab on games a fortnight old. Kickoff time is the
// only status the feed carries -- it has no final scores -- so "settled" here means
// "has kicked off", which includes a game still in progress.
const hasKickedOff=(record,now)=>{
  const kickoff=new Date(record.kickoff).getTime();
  return Number.isFinite(kickoff)&&kickoff<now;
};

function matchesFilters(r,query){
  return (state.league==='all'||r.league===state.league)
    &&(state.week==='all'||String(r.week)===state.week)
    &&(state.group==='all'||groupFor(r.league,r.home_team)===state.group||groupFor(r.league,r.away_team)===state.group)
    &&(!query||normalize(r.home_team).includes(query)||normalize(r.away_team).includes(query));
}
function filtered(){
  const query=normalize(state.teamQuery.trim());
  const now=Date.now();
  // The Games counts respect every other filter, so each button says how many rows
  // clicking it would show.
  const base=state.records.filter(r=>matchesFilters(r,query));
  const settled=base.filter(r=>hasKickedOff(r,now)).length;
  $('#count-upcoming').textContent=base.length-settled;
  $('#count-settled').textContent=settled;
  $('#count-all').textContent=base.length;
  const rows=base.filter(r=>state.status==='all'||(state.status==='settled')===hasKickedOff(r,now));
  if(state.sort==='difference')rows.sort((a,b)=>Math.abs(b.model_market_difference?.spread||0)-Math.abs(a.model_market_difference?.spread||0));
  else if(state.sort==='confidence')rows.sort((a,b)=>(a.quality_reasons?.length||0)-(b.quality_reasons?.length||0));
  else rows.sort((a,b)=>new Date(a.kickoff)-new Date(b.kickoff));
  return rows;
}
function renderProjections(){
  const rows=filtered();list.replaceChildren();empty.hidden=rows.length>0;
  $('#showing-count').innerHTML=`${rows.length}<small> / ${state.records.length}</small>`;
  // An empty Upcoming view is the ordinary state between slates, not a fault, and the
  // default now hides settled games -- so say where they went rather than listing
  // filters the reader has not touched.
  const settledAvailable=state.status==='upcoming'&&state.records.some(r=>hasKickedOff(r,Date.now()));
  empty.querySelector('h2').textContent=!state.records.length?'No published slate yet.'
    :settledAvailable?'No upcoming games.':'No projections match these filters.';
  empty.querySelector('p:last-child').textContent=!state.records.length
    ?'The site is ready, but it will not invent picks. Run the prospective model refresh to append evidence to the ledger; this page then updates from the versioned JSON artifact.'
    :settledAvailable?'Every projected game has kicked off. Switch Games to Settled to review them, or wait for the next slate to publish.'
    :'Try a different team, conference/division, league, or week.';
  rows.forEach(r=>{
    const button=document.createElement('button');button.type='button';button.className=`game-row ${r.out_of_distribution?'flagged':''}`;
    const warning=r.out_of_distribution?'<span class="warning-mark" aria-label="Extreme model and market disagreement">!</span>':'';
    button.innerHTML=`<div class="matchup-id"><span class="league-tag ${esc(r.league)}">${esc(r.league.toUpperCase())}</span><div class="teams"><strong>${esc(r.away_team)} <em>at</em> ${esc(r.home_team)}${warning}</strong><small>W${esc(r.week)} · ${date(r.kickoff)}<span class="quality">${esc(r.confidence)}</span></small></div></div>${forecast(r.independent,'','not available','Model')}${forecast(r.market,'','No line posted yet','Market')}${diffCell(r.model_market_difference)}${disagreeCell(r.model_market_difference)}${ruleCell(r)}`;
    button.addEventListener('click',()=>openDetail(r));list.append(button);
  });
}

// Wilson score interval: the same estimator scripts/build_tracker.py uses for each cohort,
// so the pooled headline and the cohort cards cannot disagree about method.
const wilson=(wins,losses)=>{
  const n=wins+losses;if(!n)return null;
  const z=1.96,p=wins/n,d=1+z*z/n,centre=(p+z*z/(2*n))/d,half=z*Math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d;
  return{pct:100*p,low:100*(centre-half),high:100*(centre+half)};
};
function renderRecord(){
  const t=state.tracker,cards=$('#rule-cards'),betList=$('#bet-list'),blank=$('#bet-empty'),head=$('#record-headline');
  if(!cards)return;
  if(!t){head.replaceChildren();cards.replaceChildren();betList.replaceChildren();blank.hidden=false;blank.textContent='Record unavailable — the tracker could not be loaded.';return}
  const pct=v=>v==null?'—':`${Number(v).toFixed(1)}%`;
  const be=Number(t.breakeven);
  const lg=state.recordLeague;
  const rules=Object.entries(t.rules||{}).map(([key,r])=>({...r,rule:r.rule||key})).filter(r=>lg==='all'||r.league===lg);
  const active=rules.map(r=>({...r,legacy:false}));
  const legacy=rules.flatMap(r=>(r.cohorts||[]).filter(c=>c.cohort!==r.cohort).map(c=>({...c,rule:r.rule,league:r.league,headline:c.headline||r.headline,legacy:true})));

  // After a protocol change the active cohorts are legitimately empty, so they cannot be
  // the scoreboard: that read 0-0 while 30 selections had settled. The headline pools every
  // cohort of the selected rules and the split sits beneath it -- nothing hidden or dropped.
  const pooled=rules.flatMap(r=>r.cohorts||[]).reduce((a,c)=>({wins:a.wins+(c.wins||0),losses:a.losses+(c.losses||0),pushes:a.pushes+(c.pushes||0),logged:a.logged+(c.logged||0),units:a.units+(Number(c.units)||0)}),{wins:0,losses:0,pushes:0,logged:0,units:0});
  const settledN=pooled.wins+pooled.losses,ci=wilson(pooled.wins,pooled.losses);
  const names=esc(rules.map(r=>r.rule).join(' + ')||'No rules');
  if(!ci){
    head.innerHTML=`<div><span class="rail-label">${names} · every selection recorded to date</span><div class="big-line"><strong class="flat">—</strong><span>${pooled.logged?`${pooled.logged} logged, none settled yet`:'No selections recorded yet'}</span></div><p>Break-even at −110 is ${be.toFixed(2)}%.</p></div>`;
  }else{
    const winning=ci.pct>=be,headTone=winning?'pos':'neg';
    const straddles=ci.low<be&&be<ci.high;
    head.innerHTML=`<div><span class="rail-label">${names} · every selection recorded to date</span>
      <div class="big-line"><strong class="${headTone}"><i class="dir dir-${winning?'up':'dn'}" aria-hidden="true"></i>${pct(ci.pct)}</strong><span>${pooled.wins}–${pooled.losses}${pooled.pushes?` · ${pooled.pushes} push`:''}</span><span class="${headTone}">${signed(ci.pct-be)} pts vs break-even</span><span>${signed(pooled.units)} units @ −110</span></div>
      <p>${settledN} settled of ${pooled.logged} logged · 95% interval ${pct(ci.low)}–${pct(ci.high)} · break-even ${be.toFixed(2)}%</p></div>
      ${straddles?`<aside class="caveat"><strong>Too few settled bets to call this either way</strong><p>The 95% interval still contains the ${be.toFixed(2)}% break-even rate, so this record cannot yet tell a losing rule from a break-even one.</p></aside>`:''}`;
  }

  const card=c=>{
    const settled=(c.wins||0)+(c.losses||0);
    const cardTone=!settled?'flat':(c.win_pct>=be?'pos':'neg');
    const ciText=c.ci_low==null?'':` · CI ${Number(c.ci_low).toFixed(1)}–${Number(c.ci_high).toFixed(1)}`;
    const priced=c.priced_bets?` · recorded-price units ${Number(c.priced_units).toFixed(2)} (${c.priced_bets} priced)`:'';
    const body=settled
      ?`<div class="card-big"><strong class="${cardTone}">${pct(c.win_pct)}</strong><span>${c.wins}–${c.losses}${c.pushes?` · ${c.pushes} push`:''}</span></div><p class="card-sub">${settled} settled · ${signed(c.units)} units${ciText}${c.pending?` · ${c.pending} pending`:''}${priced}</p>`
      // No settled bets is a different claim from a 0% win rate; never render one as the other.
      :`<div class="card-big"><strong class="flat">—</strong></div><p class="card-sub">${c.logged?`${c.logged} logged, none settled yet.`:'No selections recorded under this protocol yet.'}</p>`;
    return`<article class="rule-card ${c.legacy?'legacy':'active'}"><header><h3>${esc(c.cohort||c.rule)}</h3><span class="card-tag">${c.legacy?'Historical':'Active'}</span></header><p class="sub">${esc(c.rule||'')}${c.headline?` · ${esc(c.headline)}`:''}</p>${body}<p class="status">${c.legacy?'Does not advance the active checkpoint.':esc(c.status||'')}</p></article>`;
  };
  cards.innerHTML=[...active,...legacy].map(card).join('');

  const bets=(t.bets||[]).filter(b=>lg==='all'||b.league===lg);
  const settledBets=bets.filter(b=>b.result).sort((a,b)=>new Date(b.kickoff)-new Date(a.kickoff));
  const pendingBets=bets.filter(b=>!b.result).sort((a,b)=>new Date(a.kickoff)-new Date(b.kickoff));
  blank.hidden=bets.length>0;
  blank.textContent=(t.bets||[]).length?`No ${lg.toUpperCase()} bets logged yet.`:'No bets logged yet. The trackers append here automatically once the season starts.';
  const betRow=b=>{
    const res=(b.result||'').toLowerCase()||'pending';
    const label=res==='pending'?'PENDING':b.result;
    const wind=b.forecast_wind_mph!=null;
    const gap=wind?`${Number(b.forecast_wind_mph).toFixed(0)} mph`:signed(b.edge);
    const quote=`${b.book||'Book not recorded'} · ${b.price==null?'price unavailable':b.price}`;
    const basis=`${b.line_basis||'reference'} line · ${b.quote_observed_at?`quote ${date(b.quote_observed_at)}`:'quote time unavailable'}`;
    return`<div class="bet-row"><span class="wk">W${esc(b.week)}</span><div class="sel"><b>${esc(b.away_team)} <em>at</em> ${esc(b.home_team)}</b><small title="${esc(basis)}">${date(b.kickoff)} · ${esc(quote)}</small></div><span class="tag" title="${esc(b.cohort||'')}">${esc(b.cohort||b.rule||'')}</span><span class="side" data-label="Side">${esc(b.side||'—')}</span><span class="n" data-label="Line">${number(b.market_total_at_bet)}</span><span class="n" data-label="${wind?'Wind':'Gap'}">${esc(gap)}</span><span class="n" data-label="Close">${number(b.market_total_close)}</span><span class="n actual" data-label="Actual">${number(b.actual_total)}</span><span class="res ${res}">${esc(label)}</span></div>`;
  };
  betList.innerHTML=settledBets.map(betRow).join('')
    +(settledBets.length&&pendingBets.length?`<div class="bet-divider">Pending · ${pendingBets.length} awaiting a result</div>`:'')
    +pendingBets.map(betRow).join('');
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
  const lo=r.independent?.interval_80_low,hi=r.independent?.interval_80_high,point=r.independent?.spread,market=r.market?.spread;
  if(lo==null||hi==null)return'';
  // Symmetric about a pick'em, sized to the widest value in play so a ten-point NFL game is
  // not a sliver in a ±50 axis. Spread is positive when the home team is favoured, so the
  // home side is on the right.
  const reach=Math.max(Math.abs(lo),Math.abs(hi),Math.abs(market??0),Math.abs(point??0));
  const DOMAIN=Math.max(14,Math.ceil((reach+2)/7)*7);
  const toPct=v=>(100*(Math.max(-DOMAIN,Math.min(DOMAIN,v))+DOMAIN)/(2*DOMAIN)).toFixed(1);
  const leftPct=toPct(lo),rightPct=toPct(hi);
  const pointMark=point==null?'':`<span class="range-point" style="left:${toPct(point)}%" title="Model ${signed(point)}"></span>`;
  const marketMark=market==null?'':`<span class="range-market" style="left:${toPct(market)}%" title="Market ${signed(market)}"></span>`;
  return`<dt>80% interval (spread)</dt><dd><span class="range-bar"><span class="range-fill" style="left:${leftPct}%;right:${100-rightPct}%"></span>${marketMark}${pointMark}</span><span class="range-labels"><span>${esc(r.away_team)} by ${DOMAIN}</span><span>${signed(lo)} to ${signed(hi)}</span><span>${esc(r.home_team)} by ${DOMAIN}</span></span><small>Light marker is the model${market==null?'':', amber is the market'}.</small></dd>`;
}
function openDetail(r){
  if(!r)return;
  const ev=r.market_evidence,unavailable=r.unavailable_features?.length?r.unavailable_features.join(', '):'None reported';
  const warnings=r.warnings?.length?`<div class="alert-block"><strong>Review required</strong>${r.warnings.map(w=>`<p>${esc(w)}</p>`).join('')}</div>`:'';
  const reasons=r.quality_reasons?.length?r.quality_reasons.join('; '):'No quality blockers reported';
  const d=r.model_market_difference,rule=r.tracked_rule;
  const ruleBlock=rule?`<div class="callout ${rule.qualifies?'on':''}"><strong>Tracked rule${rule.rule?` ${esc(rule.rule)}`:''} — ${rule.qualifies?'fires':'does not fire'}</strong><p>${esc(rule.reason||rule.label||'')}</p><small>A pre-registered rule, recorded for a forward record. Not an authorisation to bet: this model's bet gates fail.</small></div>`:'';
  $('#dialog-content').innerHTML=`<div class="dialog-inner"><p class="dialog-kicker"><span class="league-tag ${esc(r.league)}">${esc(r.league.toUpperCase())}</span>Week ${esc(r.week)} · ${date(r.kickoff)}</p><h2 id="dialog-title">${esc(r.away_team)} <span>at</span> ${esc(r.home_team)}</h2>${warnings}<div class="detail-grid"><div><span class="label">Model</span><strong>${signed(r.independent?.spread)} / ${number(r.independent?.total)}</strong><small>${number(r.independent?.away_score)}–${number(r.independent?.home_score)} score</small></div><div><span class="label">Market</span><strong>${signed(r.market?.spread)} / ${number(r.market?.total)}</strong><small>${esc(ev?.label||'No line posted yet')}</small></div><div><span class="label">Difference</span><strong>${diffFigure(d?.spread,d?.total,' / ')}</strong><small>model minus market</small></div></div>${ruleBlock}<dl class="disclosures">${winProbRow(r)}${intervalRow(r)}<dt>Quality</dt><dd>${esc(r.confidence)} — ${esc(reasons)}</dd><dt>Model version</dt><dd>${esc(r.model_version)}</dd><dt>Information cutoff</dt><dd>${date(r.data_cutoff)}</dd><dt>Forecast generated</dt><dd>${date(r.generated_at)}</dd><dt>Market observed</dt><dd>${ev?.observed_at?date(ev.observed_at):'Observation time unavailable'}</dd><dt>Market evidence</dt><dd>${ev?`${esc(ev.label)}; ${ev.book_count_spread} spread / ${ev.book_count_total} total sources (${ev.providers.map(esc).join(', ')})`:'No market evidence attached'}</dd><dt>Experimental blend</dt><dd>Spread: ${r.calibration_status?.spread?'eligible':'unavailable'} · Total: ${r.calibration_status?.total?'eligible':'unavailable'}. Positive fitted weights do not establish probability calibration or a betting edge.</dd><dt>Unavailable inputs</dt><dd>${esc(unavailable)}</dd><dt>Interpretation</dt><dd>Difference is the independent model minus the market reference. Large disagreement triggers investigation, never an automatic pick.</dd></dl></div>`;
  dialog.showModal();
}

function setButtonGroup(selector,value,attribute){
  $$(selector).forEach(button=>{
    const on=button.dataset[attribute]===value;
    button.classList.toggle('active',on);
    if(attribute==='view'){if(on)button.setAttribute('aria-current','page');else button.removeAttribute('aria-current')}
    else button.setAttribute('aria-pressed',String(on));
  });
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
  const flagged=records.filter(r=>r.out_of_distribution).length,eligible=records.filter(r=>r.pick_eligible).length;
  const count=n=>Number(n||0).toLocaleString();
  $('#league-summary').innerHTML=[
    ['Rated teams',data.summary.rated_teams,''],['Season games',data.summary.scheduled_games,''],
    ['Published forecasts',records.length,''],['Market references',marketCount,''],
    ['Flagged for review',flagged,flagged?'warn':''],['Pick eligible',eligible,eligible?'':'muted'],
  ].map(([label,value,cls])=>`<div class="${cls}"><span>${label}</span><strong>${count(value)}</strong></div>`).join('');
  $('#rankings-period').textContent=`${data.season||'—'} · week ${data.week||'—'} · ${count(data.summary.completed_games)} games rated`;
  const ratings=data.ratings.slice().sort((a,b)=>(a.net_rank||999)-(b.net_rank||999));
  const maxAbsRating=Math.max(1e-9,...ratings.map(r=>Math.abs(r.net_rating||0)));
  $('#ranking-list').innerHTML=ratings.map(r=>{
    const pct=(100*Math.abs(r.net_rating||0)/maxAbsRating).toFixed(1);
    const dir=(r.net_rating||0)>=0?'pos':'neg';
    return `<button type="button" class="ranking-row" data-team-jump="${esc(r.team)}"><span class="rk">${esc(r.net_rank||'—')}</span><span class="team"><i>${esc(r.team)}</i><small>${esc(r.conference||state.explorerLeague.toUpperCase())}</small></span><span class="rating-bar-track" aria-hidden="true"><span class="rating-bar-fill ${dir}" style="width:${pct}%"></span></span><strong>${rating(r.net_rating)}</strong><em>#${esc(r.off_rank||'—')}</em><em>#${esc(r.def_rank||'—')}</em></button>`;
  }).join('')||'<p class="panel-empty">No ratings are available.</p>';

  const levels=records.reduce((acc,r)=>{const key=r.confidence||'unknown';acc[key]=(acc[key]||0)+1;return acc},{});
  if(!('established' in levels))levels.established=0;
  const ordered=Object.entries(levels).sort((a,b)=>(a[0]==='established')-(b[0]==='established')||b[1]-a[1]);
  const total=Math.max(1,records.length);
  $('#evidence-model').textContent=records[0]?.model_version||'';
  $('#evidence-overview').innerHTML=`<div class="maturity">${ordered.map(([level,n])=>`<div class="maturity-row ${n?'':'zero'}"><div><span>${esc(level.replace(/-/g,' '))}</span><b>${n}</b></div><span class="maturity-bar"><i style="width:${(100*n/total).toFixed(1)}%"></i></span></div>`).join('')}</div>`
    +`<p class="evidence-note">Quality describes how much history a forecast rests on, not whether it is right.${levels.established?'':' No forecast has reached &ldquo;established&rdquo; yet.'}</p>`
    +`<dl class="evidence-list"><dt>Extreme-disagreement review</dt><dd>${flagged}</dd><dt>3+ book consensus</dt><dd>${consensus}</dd><dt>Spread blend</dt><dd>${records.some(r=>r.calibration_status?.spread)?'experimental':'unavailable'}</dd><dt>Total blend</dt><dd>${records.some(r=>r.calibration_status?.total)?'experimental':'unavailable'}</dd></dl>`
    +(records.length&&!eligible?`<div class="caveat"><strong>Picks are blocked</strong><p>No forecast on this slate cleared the publication gates. A failing gate is a measurement, not a fault, and blend eligibility is not calibration or a pick grade.</p></div>`:'');
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
  const data=leagueData(state.teamLeague),teamRating=data.ratings.find(r=>r.team===state.team)||{};
  const schedule=data.schedule.filter(g=>g.home_team===state.team||g.away_team===state.team).sort((a,b)=>(a.week||0)-(b.week||0));
  const forecasts=state.records.filter(r=>r.league===state.teamLeague&&(r.home_team===state.team||r.away_team===state.team)).sort((a,b)=>new Date(a.kickoff)-new Date(b.kickoff));
  $('#team-identity').innerHTML=`<div class="team-name"><p>${esc(teamRating.conference||state.teamLeague.toUpperCase())} · ${esc(data.season||'')}</p><h2>${esc(state.team||'Choose a team')}</h2></div><div class="team-metrics"><span><small>Overall</small><strong>#${esc(teamRating.net_rank||'—')}</strong></span><span><small>Offense</small><strong>#${esc(teamRating.off_rank||'—')}</strong></span><span><small>Defense</small><strong>#${esc(teamRating.def_rank||'—')}</strong></span><span><small>Net rating</small><strong>${rating(teamRating.net_rating)}</strong></span></div>`;
  $('#team-schedule-label').textContent=`${schedule.length} games`;
  $('#team-schedule').innerHTML=schedule.map(g=>{
    const home=g.home_team===state.team,opponent=home?g.away_team:g.home_team;
    let result='No line yet';
    if(g.completed)result=g.home_points!=null?`${g.away_points}–${g.home_points} final`:'Final';
    // Lines arrive home-relative (positive when the home side is favoured). Shown from this
    // team's side in betting notation, so a favourite always reads negative.
    else if(g.spread_line!=null)result=`${signed(home?-g.spread_line:g.spread_line)}${g.total_line!=null?` · O/U ${number(g.total_line)}`:''}`;
    return `<div class="${g.completed?'done':''}"><b>W${esc(g.week)}</b><strong><em>${home?'vs':'at'}</em> ${esc(opponent)}</strong><small>${date(g.kickoff)}</small><em class="result">${esc(result)}</em></div>`;
  }).join('')||'<p class="panel-empty">No schedule is available.</p>';
  $('#team-forecasts').innerHTML=forecasts.map(r=>`<button type="button" data-forecast-id="${esc(r.prediction_id)}"><span class="tf-match">W${esc(r.week)} · ${esc(r.away_team)} <em>at</em> ${esc(r.home_team)}</span><span class="tf-line"><span>Model ${signed(r.independent?.spread)} · ${number(r.independent?.total)}</span><span>Diff ${diffFigure(r.model_market_difference?.spread,r.model_market_difference?.total)}</span></span><small>${esc(r.confidence)} · ${date(r.kickoff)}</small></button>`).join('')||'<p class="panel-empty">No published forecast for this team yet.</p>';
  $$('[data-forecast-id]').forEach(button=>button.addEventListener('click',()=>openDetail(state.records.find(r=>r.prediction_id===button.dataset.forecastId))));
}

// Theme: follow the operating system until the visitor chooses; after that the choice sticks.
const THEME_KEY='fd-theme';
const savedTheme=()=>{try{const t=localStorage.getItem(THEME_KEY);return t==='light'||t==='dark'?t:null}catch(_){return null}};
function applyTheme(theme,remember){
  document.documentElement.setAttribute('data-theme',theme);
  $$('[data-theme-choice]').forEach(button=>button.setAttribute('aria-pressed',String(button.dataset.themeChoice===theme)));
  if(remember){try{localStorage.setItem(THEME_KEY,theme)}catch(_){/* storage blocked: the choice lasts this page view */}}
}
$$('[data-theme-choice]').forEach(button=>button.addEventListener('click',()=>applyTheme(button.dataset.themeChoice,true)));
const lightQuery=window.matchMedia?window.matchMedia('(prefers-color-scheme: light)'):null;
lightQuery?.addEventListener?.('change',e=>{if(!savedTheme())applyTheme(e.matches?'light':'dark',false)});
applyTheme(document.documentElement.getAttribute('data-theme')==='light'?'light':'dark',false);

$('.dialog-close').addEventListener('click',()=>dialog.close());
dialog.addEventListener('click',e=>{if(e.target===dialog)dialog.close()});
$$('[data-view]').forEach(button=>button.addEventListener('click',()=>showView(button.dataset.view)));
$$('[data-record]').forEach(button=>button.addEventListener('click',()=>{state.recordLeague=button.dataset.record;setButtonGroup('[data-record]',state.recordLeague,'record');renderRecord()}));
$$('[data-league]').forEach(button=>button.addEventListener('click',()=>{setButtonGroup('[data-league]',button.dataset.league,'league');state.league=button.dataset.league;populateProjectionFilters();renderProjections()}));
$$('[data-status]').forEach(button=>button.addEventListener('click',()=>{state.status=button.dataset.status;setButtonGroup('[data-status]',state.status,'status');renderProjections()}));
$$('[data-explorer-league]').forEach(button=>button.addEventListener('click',()=>{state.explorerLeague=button.dataset.explorerLeague;setButtonGroup('[data-explorer-league]',state.explorerLeague,'explorerLeague');renderLeague()}));
$$('[data-team-league]').forEach(button=>button.addEventListener('click',()=>{state.teamLeague=button.dataset.teamLeague;state.team=null;renderTeam()}));
$('#team-select').addEventListener('change',e=>{state.team=e.target.value;renderTeam()});
weekFilter.addEventListener('change',()=>{state.week=weekFilter.value;renderProjections()});
sortFilter.addEventListener('change',()=>{state.sort=sortFilter.value;renderProjections()});
groupFilter.addEventListener('change',()=>{state.group=groupFilter.value;renderProjections()});
teamFilter.addEventListener('input',()=>{state.teamQuery=teamFilter.value;renderProjections()});
$('#clear-filters').addEventListener('click',()=>{state.league='all';state.status='upcoming';state.week='all';state.group='all';state.teamQuery='';teamFilter.value='';weekFilter.value='all';setButtonGroup('[data-league]','all','league');setButtonGroup('[data-status]','upcoming','status');populateProjectionFilters();renderProjections()});

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
  $('#gate-chip').hidden=!state.records.length||state.records.some(r=>r.pick_eligible);
  state.tracker=tracker;populateProjectionFilters();renderProjections();renderLeague();renderTeam();renderRecord();
}).catch(error=>{$('#data-status').textContent='Data unavailable';empty.hidden=false;empty.querySelector('p:last-child').textContent=`The publication artifacts could not be loaded (${error.message}).`;});
