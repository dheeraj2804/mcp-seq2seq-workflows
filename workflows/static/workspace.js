const $ = id => document.getElementById(id);
const node = (tag, text) => { const el = document.createElement(tag); el.textContent = text; return el; };
let currentRun = null;
let busy = false;
let conversationId = null;
let documentCatalog = [];
let activeJob = null;
let currentEvents = null;
let providerReadiness = null;
async function api(url, options) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) { const error = new Error(data.error || 'Request failed.'); error.data = data; throw error; }
  return data;
}
function render(data) {
  currentRun = data;
  $('empty').hidden = true;
  $('result').hidden = false;
  $('error').hidden = !['failed','cancelled'].includes(data.status);
  const evidenceLabels = {insufficient:'Insufficient evidence · the app could not establish an answer',model_supported:'Model reports supporting evidence · verify the citations',extractive:'Extracted source text',unverified:'Source support has not been verified'};
  $('evidence-badge').textContent = evidenceLabels[data.evidence_status] || ''; $('evidence-badge').hidden = !data.evidence_status; $('evidence-badge').dataset.status = data.evidence_status || '';
  $('error').textContent = data.error || '';
  $('result-heading').textContent = data.status === 'cancelled' ? 'Workflow cancelled' : data.status === 'failed' ? 'Workflow needs attention' : `Completed · ${data.answer_kind === 'extractive_summary' ? 'Extractive summary' : data.provider === 'local' ? 'Local AI explanation' : data.provider}`;
  $('result-meta').textContent = `${data.elapsed_ms ?? '—'} ms · run ${data.id.slice(0, 8)}`;
  const retrievalWarnings = [...new Set((data.sources || []).map(source => source.retrieval_warning).filter(Boolean))];
  $('answer-notice').textContent = [data.answer_notice || (data.status === 'cancelled' ? 'Cancelled by you. No draft answer was saved.' : 'Saved result from an earlier version.'), ...retrievalWarnings].join(' ');
  let answerText = data.answer || '';
  (data.sources || []).forEach((source, index) => { answerText = answerText.replaceAll('[' + source.id + ']', '[' + (index+1) + ']'); });
  renderCitedAnswer(answerText || 'The run did not produce an answer. Its recorded trace is available below.', data.sources || []);
  conversationId = data.conversation_id || null;
  loadConversation(data).catch(() => { $('conversation-status').textContent = 'Conversation history unavailable'; });
  $('feedback-form').reset(); $('feedback-status').textContent = '';
  $('answer-feedback').hidden = data.status !== 'completed';
  $('normalization').textContent = data.normalization ? `${data.normalization.mode} → ${data.normalization.query}` : 'Normalization details unavailable';
  $('raw').textContent = JSON.stringify(data, null, 2);
  $('sources').replaceChildren();
  for (const [sourceIndex, source] of (data.sources || []).entries()) {
    const item = node('div', ''); item.className = 'source';
    item.append(node('strong', `[${sourceIndex+1}] ${source.title}`));
    const details = node('details', ''); details.append(node('summary', `Read passage${source.score !== undefined ? ' · relevance score ' + source.score : ''}`), node('p', source.text));
    const inspect = node('button', source.page ? `View source · page ${source.page}` : 'View cited passage'); inspect.type = 'button'; inspect.className = 'source-open'; inspect.addEventListener('click', () => openSource(sourceIndex + 1));
    item.append(inspect, details); $('sources').append(item);
  }
  if (!(data.sources || []).length) $('sources').append(node('p', 'No evidence was retrieved.'));
  $('trace').replaceChildren();
  for (const step of data.trace || []) {
    const row = node('div', ''); row.className = 'trace';
    const status = node('span', `${step.status || step.mode || 'ok'}${step.attempt ? ' · attempt ' + step.attempt : ''}${step.ms !== undefined ? ' · ' + step.ms + ' ms' : ''}${step.error ? ' · ' + step.error : ''}`);
    if (step.status === 'error') status.className = 'failed';
    row.append(node('span', step.step.replaceAll('_', ' ')), status); $('trace').append(row);
  }
}
async function loadLibrary() {
  const data = await api('/api/documents');
  documentCatalog = data.documents;
  $('document-count').textContent = data.documents.length;
  const comparison = $('compare-document').value; $('compare-document').replaceChildren(new Option('Choose a second document',''));
  $('evaluation-documents').replaceChildren();
  for (const doc of data.documents) { $('compare-document').add(new Option(doc.title,doc.id)); $('evaluation-documents').add(new Option(doc.title,doc.id)); }
  if (data.documents.some(d=>d.id===comparison)) $('compare-document').value=comparison;
  $('library-list').replaceChildren();
  const scope = $('document-scope').value;
  $('document-scope').replaceChildren(new Option('All documents', ''));
  for (const doc of data.documents) $('document-scope').add(new Option(doc.title, doc.id));
  if (data.documents.some(d => d.id === scope)) $('document-scope').value = scope;
  for (const doc of data.documents) {
    const item = node('div', ''); item.className = 'library-item';
    item.append(node('span', doc.title), node('small', `${doc.metadata?.format?.toUpperCase() || doc.kind}${doc.metadata?.ocr_pages?.length ? ' · OCR' : ''} · ${doc.characters.toLocaleString()} chars`)); $('library-list').append(item);
  }
}
let historyCursor = null;
let displayedRuns = 0;
let historyLoading = false;
async function loadHistory(reset = true) {
  if (historyLoading) return;
  historyLoading = true;
  $('show-more-history').disabled = true;
  try {
    const url = '/api/workflows?limit=5' + (!reset && historyCursor ? '&before=' + historyCursor : '');
    const data = await api(url);
    $('run-count').textContent = data.total;
    if (reset) { $('history-list').replaceChildren(); displayedRuns = 0; }
    for (const run of data.runs) {
      const button = node('button', ''); button.type = 'button'; button.className = 'history-row';
      const label = node('span', '');
      const duration = run.elapsed_ms == null ? '—' : run.elapsed_ms >= 1000 ? (run.elapsed_ms / 1000).toFixed(1) + ' s' : Math.round(run.elapsed_ms) + ' ms';
      const provider = {local:'Local AI', demo:'Extractive summary', claude:'Claude', openai:'OpenAI'}[run.provider] || run.provider;
      label.append(node('strong', run.query), node('small', `${new Date(run.created).toLocaleString()} · ${provider} · ${duration}`));
      const status = node('span', run.status === 'completed' ? 'Completed' : run.status === 'cancelled' ? 'Cancelled' : 'Needs attention'); status.className = 'run-status ' + run.status;
      button.append(label, status);
      button.addEventListener('click', async () => {
        if (busy) return;
        try { const saved = await api('/api/workflows/' + encodeURIComponent(run.id)); render(saved); if(saved.query) $('query').value = saved.query; restoreDocumentScope(saved.document_id); $('output').scrollIntoView({behavior:'smooth', block:'start'}); }
        catch (error) { $('error').textContent = error.message; $('error').hidden = false; }
      });
      $('history-list').append(button);
    }
    displayedRuns += data.runs.length;
    historyCursor = data.next_cursor;
    $('show-more-history').hidden = !data.has_more;
    $('history-summary').textContent = `Showing ${displayedRuns} of ${data.total} workflows`;
    if (!data.total) $('history-list').append(node('p', 'No workflows yet. Ask your first question above.'));
  } finally { historyLoading = false; $('show-more-history').disabled = false; }
}
$('show-more-history').addEventListener('click', () => loadHistory(false).catch(() => { $('history-summary').textContent = 'Could not load more workflows. Please try again.'; }));
for (const id of ['library-nav', 'manage-library']) $(id).addEventListener('click', () => { $('knowledge').open = true; });
async function refreshHealth(preferred = null) {
  const data = await api('/health');
  $('health').textContent = `${data.model_loaded ? 'Seq2Seq ready' : 'Identity normalization'} · ${data.transport} tools`;
  $('model-state').textContent = data.model_loaded ? 'Seq2Seq' : 'Identity';
  $('enable-ocr').disabled = !data.ocr?.available;
  if (!data.ocr?.available) $('ocr-help').textContent = 'OCR setup is required: install Tesseract and the project OCR dependencies.';
  const localOption = $('provider').querySelector('option[value="local"]');
  localOption.disabled = !data.local_model?.ready;
  localOption.textContent = data.local_model?.ready ? `Local AI · ${data.local_model.model}` : 'Local AI · setup required';
  providerReadiness = data.providers || {};
  for (const name of ['claude', 'openai']) {
    const option = $('provider').querySelector(`option[value="${name}"]`);
    option.disabled = false;
    option.textContent = (name === 'claude' ? 'Claude' : 'OpenAI') + (!data.providers?.[name] ? ' · connect API key' : '');
  }
  $('provider').querySelector('option[value="auto"]').disabled = false;
  const choice = preferred || $('provider').value;
  const selected = Array.from($('provider').options).find(option => option.value === choice);
  $('provider').value = selected && !selected.disabled ? choice : data.local_model?.ready ? 'local' : 'demo';
}
refreshHealth().catch(() => { $('health').textContent = 'Health check unavailable'; });
loadLibrary().catch(() => { $('library-list').textContent = 'Could not load the library.'; });
loadHistory().catch(() => { $('history-list').textContent = 'Could not load recent runs.'; });
$('refresh-history').addEventListener('click', () => loadHistory().catch(() => { $('history-list').textContent = 'Could not load recent runs.'; }));
document.querySelectorAll('[data-query]').forEach(button => button.addEventListener('click', () => { $('query').value = button.dataset.query; $('query').focus(); }));
$('query').addEventListener('keydown', event => { if ((event.ctrlKey || event.metaKey) && event.key === 'Enter' && !busy) $('form').requestSubmit(); });
$('form').addEventListener('submit', async event => {
  event.preventDefault(); const query=$('query').value.trim(); if(!query || busy)return;
  if(!await ensureProviderReady())return;
  const first=$('document-scope').value; const second=$('compare-document').value;
  if($('compare-mode').checked && (!first || !second || first===second)) { $('error').textContent='Choose two different documents to compare.'; $('error').hidden=false;return; }
  busy=true;
  for(const id of ['run','document-scope','compare-document','compare-mode','new-conversation','provider'])$(id).disabled=true;
  $('run').textContent='Running…';$('cancel-workflow').hidden=false;$('cancel-workflow').disabled=false;
  $('output').setAttribute('aria-busy','true');$('error').hidden=true;$('result').hidden=true;$('empty').hidden=false;
  $('result-heading').textContent='Workflow in progress';$('empty').textContent='Retrieving evidence…';
  $('live-progress').replaceChildren();$('trace').replaceChildren();$('sources').replaceChildren();$('raw').textContent='';
  const body={query,provider:$('provider').value,document_id:first || null,conversation_id:conversationId};
  if($('compare-mode').checked)body.document_ids=[first,second];
  try { render(await streamWorkflow(body)); }
  catch(error) { $('error').textContent=error.message;$('error').hidden=false;$('empty').hidden=true;$('result-heading').textContent='Workflow needs attention'; }
  finally {
    busy=false;activeJob=null;currentEvents?.close();currentEvents=null;
    for(const id of ['run','document-scope','compare-document','compare-mode','new-conversation','provider'])$(id).disabled=false;
    $('run').textContent='Run workflow ↗';$('cancel-workflow').hidden=true;$('output').setAttribute('aria-busy','false');loadHistory().catch(()=>{});
  }
});
async function streamWorkflow(body) {
  const started=await api('/api/workflow-jobs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  activeJob=started.job_id;
  return new Promise((resolve,reject)=>{
    const events=new EventSource('/api/workflow-jobs/'+activeJob+'/events');currentEvents=events;let result=null;
    const timeout=setTimeout(()=>{events.close();reject(new Error('The live connection timed out. Reopen the latest saved run or stop the workflow.'));},240000);
    events.onmessage=message=>{
      const event=JSON.parse(message.data);
      if(event.type==='step') { const line=node('div',event.data.step.replaceAll('_',' ')+' · '+(event.data.status || event.data.mode || ''));$('live-progress').append(line);while($('live-progress').children.length>8)$('live-progress').firstChild.remove(); }
      if(event.type==='draft') {
        $('empty').hidden=true;$('result').hidden=false;$('answer').textContent=event.data.text;
        $('result-meta').textContent='Streaming draft';$('answer-notice').textContent='Draft answer · citations and evidence support are checked when generation finishes.';
        $('evidence-badge').hidden=true;$('answer-feedback').hidden=true;$('conversation-history').hidden=true;
      }
      if(event.type==='result')result=event.data;
      if(event.type==='error') {clearTimeout(timeout);events.close();reject(new Error(event.data.error));}
      if(event.type==='done') {clearTimeout(timeout);events.close();result ? resolve(result) : reject(new Error('The workflow ended without a result.'));}
    };
    events.onerror=()=>{$('live-progress').textContent='Live connection interrupted. Reconnecting…';};
  });
}
$('cancel-workflow').addEventListener('click',async()=>{
  if(!activeJob)return;$('cancel-workflow').disabled=true;
  try{await api('/api/workflow-jobs/'+activeJob,{method:'DELETE'});$('live-progress').textContent='Stopping the workflow…';}
  catch(error){$('live-progress').textContent=error.message;$('cancel-workflow').disabled=false;}
});
function restoreDocumentScope(scope) {
  const comparison=Array.isArray(scope);$('compare-mode').checked=comparison;$('compare-controls').hidden=!comparison;
  $('document-scope').value=comparison ? scope[0] : scope || '';
  $('compare-document').value=comparison ? scope[1] : '';
}
$('compare-mode').addEventListener('change',()=>{$('compare-controls').hidden=!$('compare-mode').checked;if(!busy)newConversation(false);});
$('compare-document').addEventListener('change',()=>{if(!busy)newConversation(false);});
$('doc-file').addEventListener('change', event => {
  const file = event.target.files[0];
  $('import-status').dataset.error = 'false';
  if (!file) { $('doc-text').disabled = false; $('clear-file').hidden = true; return; }
  if (!/\.(pdf|docx|txt|md|markdown)$/i.test(file.name) || file.size > 20 * 1024 * 1024) {
    $('import-status').textContent = 'Choose a PDF, DOCX, TXT, or Markdown file up to 20 MB.';
    $('import-status').dataset.error = 'true'; event.target.value = ''; $('doc-text').disabled = false; $('clear-file').hidden = true; return;
  }
  if (!$('doc-title').value) $('doc-title').value = file.name.replace(/\.[^.]+$/, '').slice(0,160);
  $('doc-text').disabled = true; $('clear-file').hidden = false;
  $('import-status').textContent = `${file.name} · ${(file.size / 1024 / 1024).toFixed(2)} MB selected. Choose Add to library to extract its text. Clear the file selection to paste text instead.`;
});
$('clear-file').addEventListener('click', () => { $('doc-file').value = ''; $('doc-text').disabled = false; $('clear-file').hidden = true; $('import-status').textContent = ''; });
$('document-form').addEventListener('submit', async event => {
  event.preventDefault();
  const file = $('doc-file').files[0];
  if (!file && $('doc-text').value.trim().length < 20) {
    $('import-status').textContent = 'Choose a file or paste at least 20 characters.'; $('import-status').dataset.error = 'true'; return;
  }
  $('import-button').disabled = true; $('doc-file').disabled = true;
  $('import-status').textContent = file ? 'Uploading and extracting text… Large or scanned files can take up to three minutes.' : 'Adding document…';
  $('import-status').dataset.error = 'false';
  try {
    let options;
    if (file) {
      const form = new FormData(); form.append('file', file); form.append('title', $('doc-title').value); form.append('ocr', String($('enable-ocr').checked));
      options = {method:'POST', body:form};
    } else {
      options = {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({title:$('doc-title').value, text:$('doc-text').value})};
    }
    const imported = await api('/api/documents', options);
    $('document-form').reset(); $('doc-text').disabled = false; $('clear-file').hidden = true;
    $('import-status').textContent = `Added ${imported.characters.toLocaleString()} searchable characters${imported.metadata?.pages ? ' from ' + imported.metadata.pages + ' pages' : ''}. You can now ask questions about this document. ${(imported.metadata?.warnings || []).join(' ')}`;
    await loadLibrary();
  } catch (error) { $('import-status').textContent = error.message; $('import-status').dataset.error = 'true'; }
  finally { $('import-button').disabled = false; $('doc-file').disabled = false; }
});
$('download').addEventListener('click', () => {
  if (!currentRun) return;
  const url = URL.createObjectURL(new Blob([JSON.stringify(currentRun, null, 2)], {type:'application/json'}));
  const link = document.createElement('a'); link.href = url; link.download = `workflow-${currentRun.id}.json`; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
});

// Credentials are write-only and are never kept in localStorage or sessionStorage.
let providerConfigurations = {};
let settingsBusy = false;
const settingsDialog = $('provider-settings');
function settingsMessage(text, error = false) {
  $('settings-message').textContent = text;
  $('settings-message').dataset.error = String(error);
}
function populateProviderForm() {
  const config = providerConfigurations[$('settings-provider').value] || {};
  $('settings-key').value = '';
  $('settings-key').placeholder = config.has_key ? 'Key configured · leave blank to keep it' : 'Paste your API key';
  $('settings-model').value = config.model || '';
  $('provider-models').replaceChildren();
  $('settings-config-status').textContent = config.configured ? `Configured · ${config.source === 'session' ? 'this app session' : 'environment settings'}` : config.has_key ? 'Key configured · choose a model' : 'Not connected';
  $('disconnect-provider').disabled = !config.has_key;
  settingsMessage('');
}
async function loadProviderSettings() {
  const data = await api('/api/settings/providers');
  providerConfigurations = data.providers;
  populateProviderForm();
}
function setSettingsBusy(value) {
  settingsBusy = value;
  for (const id of ['load-provider-models', 'test-provider', 'save-provider', 'settings-provider', 'settings-key', 'settings-model', 'disconnect-provider']) $(id).disabled = value;
  if (!value) $('disconnect-provider').disabled = !providerConfigurations[$('settings-provider').value]?.has_key;
}
async function settingsAction(action) {
  if (settingsBusy) return;
  const provider = $('settings-provider').value;
  const body = {model:$('settings-model').value.trim()};
  if ($('settings-key').value.trim()) body.api_key = $('settings-key').value.trim();
  if (action !== 'disconnect' && !body.api_key && !providerConfigurations[provider]?.has_key) { settingsMessage('Enter an API key first.', true); return; }
  if (['save','test'].includes(action) && !body.model) { settingsMessage('Choose a model or enter its exact ID.', true); return; }
  setSettingsBusy(true);
  settingsMessage(action === 'models' ? 'Loading available models…' : action === 'test' ? 'Testing the selected model…' : action === 'save' ? 'Saving for this session…' : 'Disconnecting…');
  const headers = {'Content-Type':'application/json', 'X-Settings-Token':document.querySelector('meta[name="settings-token"]').content};
  try {
    const url = '/api/settings/providers/' + provider + (['models','test'].includes(action) ? '/' + action : '');
    const result = await api(url, {method:action === 'disconnect' ? 'DELETE' : 'POST', headers, ...(action === 'disconnect' ? {} : {body:JSON.stringify(body)})});
    if (action === 'models') {
      $('provider-models').replaceChildren();
      for (const model of result.models) { const option = document.createElement('option'); option.value = model; $('provider-models').append(option); }
      settingsMessage(`Loaded ${result.models.length} model IDs. Choose a text-generation model from the Model field, or type its exact ID.`);
      $('settings-model').focus();
    } else if (action === 'test') settingsMessage(result.message);
    else {
      providerConfigurations = result.providers;
      populateProviderForm();
      settingsMessage(action === 'save' ? 'Saved for this app session. The provider is now selected for your next workflow.' : 'Provider disconnected for this session. Environment settings, if any, are unchanged.');
      await refreshHealth(action === 'save' ? provider : null);
    }
  } catch (error) { settingsMessage(error.message, true); }
  finally { setSettingsBusy(false); }
}
async function openProviderSettings(provider = null, guidance = '') {
  if (provider) $('settings-provider').value = provider;
  if (!settingsDialog.open) settingsDialog.showModal();
  try { await loadProviderSettings(); if (guidance) settingsMessage(guidance); }
  catch { settingsMessage('Could not load provider settings. Please refresh the page.', true); }
}
async function ensureProviderReady() {
  const selected = $('provider').value;
  if (!['openai', 'claude', 'auto'].includes(selected)) return true;
  if (!providerReadiness) {
    try { await refreshHealth(selected); }
    catch { await openProviderSettings(selected === 'auto' ? 'openai' : selected, 'Configure your provider before running a cloud workflow.'); return false; }
  }
  const needsSetup = selected === 'auto' ? !providerReadiness.openai && !providerReadiness.claude : !providerReadiness[selected];
  if (needsSetup) {
    await openProviderSettings(selected === 'auto' ? 'openai' : selected, 'Enter your API key and choose a model, then save for this session to enable cloud answers.');
    return false;
  }
  return true;
}
$('provider').addEventListener('change', () => { ensureProviderReady(); });
$('open-settings').addEventListener('click', () => openProviderSettings());
$('close-settings').addEventListener('click', () => settingsDialog.close());
settingsDialog.addEventListener('close', () => { $('settings-key').value = ''; $('provider-models').replaceChildren(); });
$('settings-provider').addEventListener('change', populateProviderForm);
$('load-provider-models').addEventListener('click', () => settingsAction('models'));
$('test-provider').addEventListener('click', () => settingsAction('test'));
$('disconnect-provider').addEventListener('click', () => settingsAction('disconnect'));
$('provider-settings-form').addEventListener('submit', event => { event.preventDefault(); settingsAction('save'); });

let activeSource = null;
let pdfRenderer = null;
let activeSourceIndex = null;
let activeEvaluation = null;
function renderCitedAnswer(text, sources) {
  const output = $('answer'); output.replaceChildren(); let offset = 0;
  for (const match of text.matchAll(/\[(\d+)\]/g)) {
    output.append(document.createTextNode(text.slice(offset, match.index)));
    const index = Number(match[1]);
    if (index >= 1 && index <= sources.length) {
      const citation = node('button', match[0]); citation.type = 'button'; citation.className = 'citation'; citation.title = 'View supporting source: ' + sources[index-1].title;
      citation.addEventListener('click', () => openSource(index)); output.append(citation);
    } else output.append(document.createTextNode(match[0]));
    offset = match.index + match[0].length;
  }
  output.append(document.createTextNode(text.slice(offset)));
}
async function openSource(index) {
  if (!currentRun) return;
  activeSourceIndex = index;
  $('source-heading').textContent = 'Cited evidence'; $('source-message').textContent = 'Loading the cited passage…';
  $('source-page-text').replaceChildren(); $('source-passage').textContent = '';
  $('source-pdf').hidden = true; $('source-pdf').removeAttribute('src'); $('attach-original-form').hidden = true;
  if (!$('source-dialog').open) $('source-dialog').showModal();
  try {
    const data = await api('/api/workflows/' + encodeURIComponent(currentRun.id) + '/sources/' + index);
    activeSource = data;
    $('source-heading').textContent = data.title;
    $('source-page-label').textContent = data.page ? `Page ${data.page}${data.page_inferred ? ' · recovered from a legacy citation' : ''}` : 'Text document';
    $('source-message').textContent = data.pdf_url ? 'Original PDF and extracted text. Matching text is highlighted when its position can be recovered; verify technical details against the page.' : 'Showing indexed text. ' + (data.format === 'pdf' ? 'Attach the original PDF below to enable the page preview.' : '');
    if(data.ocr_page)$('source-message').textContent += ` OCR-recognized page; recognition confidence ${data.ocr_confidence ?? 'unknown'}/100. This is text-recognition confidence, not factual accuracy.`;
    $('source-passage').textContent = data.passage;
    if (data.highlight) {
      const [start,end] = data.highlight;
      $('source-page-text').append(document.createTextNode(data.page_text.slice(0,start)), node('mark', data.page_text.slice(start,end)), document.createTextNode(data.page_text.slice(end)));
    } else { $('source-page-text').textContent = data.page_text; $('source-message').textContent += ' Exact highlighting was unavailable; the retrieved passage is shown below.'; }
    if (data.highlight) { const mark = $('source-page-text').querySelector('mark'); if (mark) $('source-page-text').scrollTop = Math.max(0, mark.offsetTop - 30); }
    if (data.pdf_url) {
      $('source-pdf').hidden = false; $('source-pdf-link').href = data.pdf_url + '#page=' + (data.page || 1);
      pdfRenderer = await import('/static/source-pdf.js');
      if (!$('source-dialog').open || activeSourceIndex !== index) return;
      try { await pdfRenderer.renderSourcePdf(data.pdf_url, data.page || 1, data.passage); }
      catch { $('source-render-status').textContent = 'Preview unavailable. Use Open PDF separately or read the highlighted extracted text.'; }
    }
    $('attach-original-form').hidden = !!data.pdf_url || data.format !== 'pdf';
  } catch (error) { $('source-message').textContent = error.message; }
}
$('close-source').addEventListener('click', () => $('source-dialog').close());
$('source-dialog').addEventListener('close', () => { pdfRenderer?.disposeSourcePdf(); $('source-pdf').removeAttribute('src'); $('source-original-file').value = ''; });
$('attach-original-form').addEventListener('submit', async event => {
  event.preventDefault(); const file = $('source-original-file').files[0]; if (!file || !activeSource) return;
  if (file.size > 20*1024*1024) { $('source-message').textContent = 'Choose a PDF up to 20 MB.'; return; }
  const button = event.submitter; button.disabled = true; $('source-message').textContent = 'Verifying the original PDF…';
  try { const form = new FormData(); form.append('file', file); await api('/api/documents/' + activeSource.document_id + '/original', {method:'POST',body:form}); await openSource(activeSourceIndex); }
  catch (error) { $('source-message').textContent = error.message; }
  finally { button.disabled = false; }
});
async function loadConversation(run) {
  if (!run.conversation_id) { $('conversation-status').textContent = 'Standalone saved workflow'; $('conversation-history').hidden = true; return; }
  const data = await api('/api/conversations/' + run.conversation_id);
  if (conversationId !== run.conversation_id) return;
  $('conversation-status').textContent = `Conversation · ${data.messages.length} turn${data.messages.length === 1 ? '' : 's'} · follow-ups use the latest turn`;
  const previous = data.messages.filter(message => message.id !== run.id);
  $('conversation-history').hidden = !previous.length;
  $('conversation-turns').replaceChildren();
  for (const message of previous) { const turn = node('div',''); turn.className = 'conversation-turn'; turn.append(node('strong', message.query), node('p', message.answer || 'This turn did not complete.')); $('conversation-turns').append(turn); }
}
function newConversation(clearQuestion = true) {
  conversationId = null; currentRun = null;
  $('conversation-status').textContent = 'New conversation'; $('conversation-history').hidden = true;
  $('result').hidden = true; $('empty').hidden = false; $('empty').textContent = 'Ask a question to start a new conversation.';
  $('error').hidden = true; $('result-heading').textContent = 'Answer & evidence';
  if (clearQuestion) { $('query').value = ''; $('query').focus(); }
}
$('new-conversation').addEventListener('click', () => { if (!busy) newConversation(); });
$('document-scope').addEventListener('change', () => { if (!busy) newConversation(false); });
$('feedback-form').addEventListener('submit', async event => {
  event.preventDefault(); if (!currentRun) return;
  const button = event.submitter; button.disabled = true;
  try { await api('/api/workflows/' + currentRun.id + '/feedback', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({helpful:$('feedback-helpful').value === 'yes',supported:$('feedback-supported').value === 'yes',note:$('feedback-note').value})}); $('feedback-status').textContent = 'Review saved. Thank you.'; }
  catch (error) { $('feedback-status').textContent = error.message; }
  finally { button.disabled = false; }
});
function showEvaluation(result) {
  activeEvaluation = result; $('evaluation-results').replaceChildren(); $('export-evaluation').hidden = !result;
  if (!result) return;
  $('evaluation-results').append(node('p', `${result.dataset} · ${result.questions} questions · ${result.corpus_passages} indexed passages`));
  const table = node('table',''); table.className = 'evaluation-table'; const head = node('tr','');
  for (const title of ['Measure','Keyword','Hybrid semantic','Hybrid + Seq2Seq']) head.append(node('th',title)); table.append(head);
  for (const [key,label,percent] of [['recall_at_3','Document recall @3',true],['mrr','Mean reciprocal rank',false],['no_hit_accuracy','No-hit accuracy',true],['median_ms','Median retrieval latency',false],['p95_ms','P95 retrieval latency',false]]) {
    const row = node('tr',''); row.append(node('td',label));
    for (const mode of ['original','hybrid','seq2seq']) { const value = result.metrics[mode]?.[key]; row.append(node('td', value == null ? 'N/A' : percent ? (value*100).toFixed(1)+'%' : key.endsWith('_ms') ? value.toFixed(2)+' ms' : value.toFixed(3))); } table.append(row);
  }
  $('evaluation-results').append(table, node('p',result.scope),node('p',result.limitations));
  const details = node('details',''); details.append(node('summary','Inspect all labeled questions'));
  for (const item of result.cases) { const row = node('p', item.query + '\nExpected: ' + (item.expected_documents.join(', ') || 'no relevant document') + '\nOriginal: ' + item.original.retrieved.join(', ') + '\nHybrid: ' + (item.hybrid?.retrieved || []).join(', ') + '\nSeq2Seq: ' + item.seq2seq.retrieved.join(', ')); row.className = 'evaluation-case'; details.append(row); }
  $('evaluation-results').append(details);
}
async function loadEvaluationDashboard() {
  const data = await api('/api/evaluations'); showEvaluation(data.latest);
  const feedback = data.feedback;
  $('evaluation-feedback').textContent = feedback.rated_answers ? `${feedback.rated_answers} rated answer(s) · ${(feedback.helpful_rate*100).toFixed(0)}% marked helpful · ${(feedback.supported_rate*100).toFixed(0)}% marked supported by sources` : 'No answers rated yet. Review an answer to add human feedback.';
}
$('open-evaluations').addEventListener('click', async () => { $('evaluation-dialog').showModal(); try { await loadEvaluationSets(); await loadEvaluationDashboard(); } catch(error) { $('evaluation-status').textContent = error.message; } });
$('close-evaluations').addEventListener('click', () => $('evaluation-dialog').close());
$('run-evaluation').addEventListener('click', async () => {
  $('run-evaluation').disabled = true; $('evaluation-status').textContent = 'Evaluating the current library…';
  try { const result = await api('/api/evaluations', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({dataset_id:$('evaluation-set').value})}); showEvaluation(result); $('evaluation-status').textContent = 'Evaluation complete · ' + new Date(result.created).toLocaleString(); }
  catch (error) { $('evaluation-status').textContent = error.message; }
  finally { $('run-evaluation').disabled = false; }
});
$('export-evaluation').addEventListener('click', () => {
  if (!activeEvaluation) return;
  const url = URL.createObjectURL(new Blob([JSON.stringify(activeEvaluation,null,2)], {type:'application/json'}));
  const link = document.createElement('a'); link.href = url; link.download = 'retrieval-evaluation-' + activeEvaluation.id + '.json'; link.click(); setTimeout(() => URL.revokeObjectURL(url),1000);
});

let customCases=[];
async function loadEvaluationSets(selected=null) {
  const data=await api('/api/evaluation-sets');const previous=selected || $('evaluation-set').value;
  $('evaluation-set').replaceChildren();for(const set of data.sets)$('evaluation-set').add(new Option(`${set.name} · ${set.questions} questions`,set.id));
  if(data.sets.some(set=>set.id===previous))$('evaluation-set').value=previous;
}
function showCustomCases() {
  $('evaluation-draft-cases').replaceChildren();
  customCases.forEach((item,index)=>{const row=node('div','');row.className='evaluation-draft-case';
    const labels=item.expected_documents.map(id=>documentCatalog.find(doc=>doc.id===id)?.title || id).join(', ') || 'No relevant evidence expected';
    row.append(node('span',item.query+' — '+labels));const remove=node('button','Remove');remove.type='button';remove.addEventListener('click',()=>{customCases.splice(index,1);showCustomCases();});row.append(remove);$('evaluation-draft-cases').append(row);
  });
}
$('evaluation-unanswerable').addEventListener('change',()=>{$('evaluation-documents').disabled=$('evaluation-unanswerable').checked;});
$('add-evaluation-case').addEventListener('click',()=>{
  const query=$('evaluation-question').value.trim();const expected=Array.from($('evaluation-documents').selectedOptions).map(option=>option.value);
  if(!query || (!$('evaluation-unanswerable').checked && !expected.length)){$('evaluation-status').textContent='Enter a question and select relevant documents, or mark it unanswerable.';return;}
  if(customCases.length>=100){$('evaluation-status').textContent='A set supports up to 100 questions.';return;}
  customCases.push({query,expected_documents:$('evaluation-unanswerable').checked ? [] : expected});$('evaluation-question').value='';showCustomCases();
});
$('custom-evaluation-form').addEventListener('submit',async event=>{
  event.preventDefault();const button=event.submitter;button.disabled=true;
  try{const result=await api('/api/evaluation-sets',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name:$('evaluation-name').value,questions:customCases})});await loadEvaluationSets(result.id);customCases=[];showCustomCases();$('evaluation-status').textContent='Custom set saved. Run retrieval evaluation to test it.';}
  catch(error){$('evaluation-status').textContent=error.message;}
  finally{button.disabled=false;}
});
