import * as pdfjs from './pdfjs/pdf.min.mjs';
pdfjs.GlobalWorkerOptions.workerSrc = new URL('./pdfjs/pdf.worker.min.mjs', import.meta.url).href;
let generation = 0;
let currentPdf = null;
let currentRender = null;
export async function disposeSourcePdf() {
  generation++;
  if (currentRender) { currentRender.cancel(); currentRender = null; }
  if (currentPdf) { const pdf = currentPdf; currentPdf = null; await pdf.destroy(); }
}
export async function renderSourcePdf(url, pageNumber, passage = '') {
  await disposeSourcePdf();
  const request = generation;
  const status = document.getElementById('source-render-status');
  const overlay = document.getElementById('source-pdf-highlights'); overlay.replaceChildren();
  status.textContent = 'Rendering original PDF page…';
  const task = pdfjs.getDocument({url, standardFontDataUrl:'/static/pdfjs/standard_fonts/', isEvalSupported:false});
  const pdf = await task.promise;
  if (request !== generation) { await pdf.destroy(); return; }
  currentPdf = pdf;
  const page = await pdf.getPage(pageNumber);
  if (request !== generation) return;
  const canvas = document.getElementById('source-pdf-canvas');
  const container = document.getElementById('source-pdf');
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.max(280, container.clientWidth - 24);
  const viewport = page.getViewport({scale:width/page.getViewport({scale:1}).width});
  canvas.width = Math.floor(viewport.width*ratio); canvas.height = Math.floor(viewport.height*ratio);
  canvas.style.width = viewport.width+'px'; canvas.style.height = viewport.height+'px';
  document.getElementById('source-canvas-wrap').style.width = viewport.width+'px';
  canvas.setAttribute('aria-label', 'Original PDF, page '+pageNumber);
  currentRender = page.render({canvasContext:canvas.getContext('2d'),viewport,transform:ratio===1 ? null : [ratio,0,0,ratio,0,0]});
  await currentRender.promise;
  if (request !== generation) return;
  currentRender = null;
  const content = await page.getTextContent();
  if (request !== generation) return;
  const compact = value => value.normalize('NFKC').replace(/[^\p{L}\p{N}]/gu,'').toLowerCase();
  const items = content.items.filter(item => typeof item.str === 'string');
  const strings = items.map(item => compact(item.str)); const all = strings.join('');
  let quote = compact(passage.replace(/\[Page \d+\]/g,'')); let start = all.indexOf(quote);
  if (start < 0) { quote = quote.slice(0,200); start = all.indexOf(quote); }
  if (start < 0) { quote = quote.slice(0,50); start = all.indexOf(quote); }
  let offset = 0; let firstTop = null;
  if (start >= 0 && quote.length >= 20) {
    const end = start + quote.length;
    for (let index=0; index<items.length; index++) {
      const item = items[index]; const next = offset + strings[index].length;
      if (next > start && offset < end && strings[index]) {
        const [a,b,c,d,e,f] = viewport.transform;
        const x = a*item.transform[4] + c*item.transform[5] + e;
        const baseline = b*item.transform[4] + d*item.transform[5] + f;
        const height = Math.max(5,Math.hypot(item.transform[2],item.transform[3])*viewport.scale);
        const top = baseline - height;
        const rect = document.createElement('span'); rect.style.cssText = `left:${x}px;top:${top}px;width:${Math.max(3,item.width*viewport.scale)}px;height:${height*1.2}px`;
        overlay.append(rect); if(firstTop===null) firstTop=top;
      }
      offset=next;
    }
  }
  status.textContent = 'Original PDF · page '+pageNumber+' of '+pdf.numPages + (firstTop !== null ? ' · matching text highlighted' : '');
  container.scrollTop = firstTop === null ? 0 : Math.max(0, document.getElementById('source-canvas-wrap').offsetTop + firstTop - 35);

}
