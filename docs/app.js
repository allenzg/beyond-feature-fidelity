'use strict';
const navLinks = [...document.querySelectorAll('nav a')];
const sections = navLinks.map(link => document.querySelector(link.getAttribute('href')));
let scrollQueued = false;
function updateNavigation() {
  const current = [...sections].reverse().find(section => section.getBoundingClientRect().top <= 155);
  navLinks.forEach(link => {
    const active = current && link.hash === '#' + current.id;
    link.classList.toggle('active', Boolean(active));
    if (active) link.setAttribute('aria-current', 'location'); else link.removeAttribute('aria-current');
  });
  const range = document.documentElement.scrollHeight - innerHeight;
  document.querySelector('.reading-progress').style.width = (range > 0 ? scrollY / range * 100 : 0) + '%';
  scrollQueued = false;
}
addEventListener('scroll', () => { if (!scrollQueued) { scrollQueued = true; requestAnimationFrame(updateNavigation); } }, { passive: true });
addEventListener('resize', updateNavigation);
updateNavigation();
const modelSelect = document.querySelector('#model-select');
const benchmarkSelect = document.querySelector('#benchmark-select');
fetch('results.json').then(response => { if (!response.ok) throw new Error('Results unavailable'); return response.json(); }).then(results => {
  function render() {
    const model = modelSelect.value;
    const index = benchmarkSelect.selectedIndex;
    const rows = [...document.querySelectorAll('.bar-row')];
    const entries = Object.entries(results[model]);
    entries.forEach(([condition, scores], i) => {
      rows[i].querySelector('.bar').style.width = Number(scores[index]) + '%';
      rows[i].querySelector('b').textContent = scores[index];
    });
    document.querySelector('#chart').setAttribute('aria-label', model + ' ' + benchmarkSelect.value + ': ' + entries.map(([name,scores]) => name + ' ' + scores[index]).join(', '));
    document.querySelector('#chart-note').textContent = model + ' · ' + benchmarkSelect.value + ' · scores in percent · zero-based axis. Benchmark-specific scoring follows the paper.';
  }
  modelSelect.addEventListener('change', render);
  benchmarkSelect.addEventListener('change', render);
  render();
}).catch(() => {
  modelSelect.disabled = true;
  benchmarkSelect.disabled = true;
  document.querySelector('#chart-note').textContent = 'Showing LLaVA-1.5 / GQA. Interactive data could not load; all scores remain available in the tables below.';
});
document.querySelectorAll('#results table').forEach(table => {
  const rows = [...table.querySelectorAll('tbody tr')];
  for (let col = 1; col <= 9; col++) {
    const best = Math.max(...rows.map(row => Number(row.children[col].textContent)));
    rows.forEach(row => { if (Number(row.children[col].textContent) === best) row.children[col].classList.add('best-score'); });
  }
});
document.querySelector('#copy-citation').addEventListener('click', async () => {
  const text = document.querySelector('#citation-text').textContent;
  const status = document.querySelector('#copy-status');
  try {
    await navigator.clipboard.writeText(text);
    status.textContent = 'Citation copied.';
  } catch {
    const selection = getSelection();
    const range = document.createRange();
    range.selectNodeContents(document.querySelector('#citation-text'));
    selection.removeAllRanges(); selection.addRange(range);
    status.textContent = 'Citation selected. Press Ctrl+C or Command+C to copy, or download the .bib file below.';
  }
});
